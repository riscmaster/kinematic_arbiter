#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace kinematic_arbiter {
namespace ros2 {

KinematicArbiterNode::KinematicArbiterNode()
    : Node("kinematic_arbiter") {

  // Declare and get parameters
  this->declare_parameter("publish_rate", 20.0);
  this->declare_parameter("max_delay_window", 0.5);
  this->declare_parameter("world_frame_id", "map");
  this->declare_parameter("body_frame_id", "base_link");

  publish_rate_ = this->get_parameter("publish_rate").as_double();
  max_delay_window_ = this->get_parameter("max_delay_window").as_double();
  world_frame_id_ = this->get_parameter("world_frame_id").as_string();
  body_frame_id_ = this->get_parameter("body_frame_id").as_string();

  // Set up TF2 transform listener and buffer
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize the filter wrapper
  filter_wrapper_ = std::make_unique<wrapper::FilterWrapper>();
  filter_wrapper_->setMaxDelayWindow(max_delay_window_);

  // Create publishers for state estimates
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "state/pose", 10);
  velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "state/velocity", 10);
  accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelWithCovarianceStamped>(
      "state/acceleration", 10);

  // Set up the publishing timer
  publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
      std::bind(&KinematicArbiterNode::publishEstimates, this));

  // Look for position sensor parameters
  this->declare_parameter("position_sensors", std::vector<std::string>());
  auto position_sensors = this->get_parameter("position_sensors").as_string_array();

  for (const auto& sensor_name : position_sensors) {
    std::string topic_param = "sensors." + sensor_name + ".topic";
    this->declare_parameter(topic_param, "");
    std::string topic = this->get_parameter(topic_param).as_string();

    if (!topic.empty()) {
      // Register the sensor with the filter
      std::string sensor_id = filter_wrapper_->registerPositionSensor(sensor_name);

      // Create the subscription pair
      createSensorPair<geometry_msgs::msg::PointStamped>(
          "position", sensor_id, topic, position_subs_);
    }
  }

  // Services
  // register_sensor_srv_ = this->create_service<kinematic_arbiter_interfaces::srv::RegisterSensor>(
  //     "register_sensor",
  //     std::bind(&KinematicArbiterNode::registerSensorCallback, this,
  //               std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(this->get_logger(), "Kinematic Arbiter node initialized");
}

void KinematicArbiterNode::positionCallback(
    const geometry_msgs::msg::PointStamped::SharedPtr msg,
    const std::string& sensor_id) {

  RCLCPP_DEBUG(this->get_logger(), "Received position from sensor %s", sensor_id.c_str());

  // Get sensor transform from TF tree
  if (!updateSensorTransform(sensor_id, msg->header.frame_id, msg->header.stamp)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to get transform for sensor %s from frame %s to %s",
                sensor_id.c_str(), msg->header.frame_id.c_str(), body_frame_id_.c_str());
    // Continue processing even if transform failed - we'll use the last known transform
  }

  // Process the measurement
  bool success = filter_wrapper_->processPosition(sensor_id, *msg);

  if (!success) {
    RCLCPP_WARN(this->get_logger(), "Failed to process position measurement from sensor %s",
                sensor_id.c_str());
    return;
  }

  // Get the expected measurement for this sensor
  auto expected = filter_wrapper_->getExpectedPosition(sensor_id);

  // Find the subscription to publish the expected measurement
  for (const auto& sub : position_subs_) {
    if (sub.sensor_id == sensor_id && sub.expected_pub) {
      // Publish expected measurement
      auto expected_pub = std::static_pointer_cast<rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>>(
          sub.expected_pub);
      expected_pub->publish(expected);
      break;
    }
  }
}

bool KinematicArbiterNode::updateSensorTransform(
    const std::string& sensor_id,
    const std::string& frame_id,
    const rclcpp::Time& time) {

  try {
    // Look up transform from sensor frame to body frame
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped = tf_buffer_->lookupTransform(
        body_frame_id_,                 // target frame (body)
        frame_id,                       // source frame (sensor)
        time,                          // time
        rclcpp::Duration::from_seconds(0.1)  // timeout
    );

    // Convert to Eigen Isometry3d
    Eigen::Isometry3d sensor_to_body_transform = tf2::transformToEigen(transform_stamped);

    // Set the transform in the sensor model
    filter_wrapper_->setSensorTransform(sensor_id, sensor_to_body_transform);

    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Transform lookup failed: %s", ex.what());
    return false;
  }
}

void KinematicArbiterNode::publishEstimates() {
  // Get current time
  rclcpp::Time current_time = this->now();

  // Predict to current time
  filter_wrapper_->predictTo(current_time);

  // Get and publish state estimates
  auto pose_estimate = filter_wrapper_->getPoseEstimate(current_time);
  auto velocity_estimate = filter_wrapper_->getVelocityEstimate(current_time);
  auto accel_estimate = filter_wrapper_->getAccelerationEstimate(current_time);

  // Set the frame_id
  pose_estimate.header.frame_id = world_frame_id_;
  velocity_estimate.header.frame_id = body_frame_id_;
  accel_estimate.header.frame_id = body_frame_id_;

  // Publish
  pose_pub_->publish(pose_estimate);
  velocity_pub_->publish(velocity_estimate);
  accel_pub_->publish(accel_estimate);
}

template<typename MsgType>
void KinematicArbiterNode::createSensorPair(
    const std::string& sensor_type,
    const std::string& sensor_id,
    const std::string& topic,
    std::vector<SensorSubscription>& subscription_list) {

  SensorSubscription sub;
  sub.sensor_id = sensor_id;
  sub.topic = topic;

  // Create subscription with updated callback signature (using shared_ptr<const MsgType>)
  if constexpr (std::is_same_v<MsgType, geometry_msgs::msg::PointStamped>) {
    sub.subscription = this->create_subscription<MsgType>(
        topic, 10,
        [this, sensor_id](const std::shared_ptr<const MsgType> msg) {
          this->positionCallback(std::const_pointer_cast<MsgType>(msg), sensor_id);
        });

    // Create publisher for expected measurements
    sub.expected_pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        topic + "/expected", 10);
  }
  // Add more sensor types here when they are implemented in the wrapper

  subscription_list.push_back(sub);

  RCLCPP_INFO(this->get_logger(), "Created %s sensor '%s' on topic '%s'",
              sensor_type.c_str(), sensor_id.c_str(), topic.c_str());
}

// void KinematicArbiterNode::registerSensorCallback(
//     const kinematic_arbiter_interfaces::srv::RegisterSensor::Request::SharedPtr request,
//     kinematic_arbiter_interfaces::srv::RegisterSensor::Response::SharedPtr response) {
//   // Implementation for dynamic sensor registration via service
// }

} // namespace ros2
} // namespace kinematic_arbiter
