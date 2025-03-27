#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <tf2_eigen/tf2_eigen.hpp>
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

  // Initialize subscription vectors for each sensor type
  position_subs_ = {};
  pose_subs_ = {};
  velocity_subs_ = {};
  imu_subs_ = {};

  // Initialize all sensor types
  initializeSensors<geometry_msgs::msg::PointStamped>("position_sensors", position_subs_,
      &wrapper::FilterWrapper::registerPositionSensor);

  initializeSensors<geometry_msgs::msg::PoseStamped>("pose_sensors", pose_subs_,
      &wrapper::FilterWrapper::registerPoseSensor);

  initializeSensors<geometry_msgs::msg::TwistStamped>("velocity_sensors", velocity_subs_,
      &wrapper::FilterWrapper::registerBodyVelocitySensor);

  initializeSensors<sensor_msgs::msg::Imu>("imu_sensors", imu_subs_,
      &wrapper::FilterWrapper::registerImuSensor);

  RCLCPP_INFO(this->get_logger(), "Kinematic Arbiter node initialized with:");
  RCLCPP_INFO(this->get_logger(), "  - %zu position sensors", position_subs_.size());
  RCLCPP_INFO(this->get_logger(), "  - %zu pose sensors", pose_subs_.size());
  RCLCPP_INFO(this->get_logger(), "  - %zu velocity sensors", velocity_subs_.size());
  RCLCPP_INFO(this->get_logger(), "  - %zu IMU sensors", imu_subs_.size());
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
  expected.header.frame_id = world_frame_id_;
  expected.header.stamp = this->now();

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

void KinematicArbiterNode::poseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg,
    const std::string& sensor_id) {

  RCLCPP_DEBUG(this->get_logger(), "Received pose from sensor %s", sensor_id.c_str());

  // Get sensor transform from TF tree
  if (!updateSensorTransform(sensor_id, msg->header.frame_id, msg->header.stamp)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to get transform for sensor %s from frame %s to %s",
                sensor_id.c_str(), msg->header.frame_id.c_str(), body_frame_id_.c_str());
  }

  // Process the measurement
  bool success = filter_wrapper_->processPose(sensor_id, *msg);

  if (!success) {
    RCLCPP_WARN(this->get_logger(), "Failed to process pose measurement from sensor %s",
                sensor_id.c_str());
    return;
  }

  // We could implement expected pose publishing similar to position if needed
}

void KinematicArbiterNode::velocityCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg,
    const std::string& sensor_id) {

  RCLCPP_DEBUG(this->get_logger(), "Received velocity from sensor %s", sensor_id.c_str());

  // Get sensor transform from TF tree
  if (!updateSensorTransform(sensor_id, msg->header.frame_id, msg->header.stamp)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to get transform for sensor %s from frame %s to %s",
                sensor_id.c_str(), msg->header.frame_id.c_str(), body_frame_id_.c_str());
  }

  // Process the measurement
  bool success = filter_wrapper_->processBodyVelocity(sensor_id, *msg);

  if (!success) {
    RCLCPP_WARN(this->get_logger(), "Failed to process velocity measurement from sensor %s",
                sensor_id.c_str());
  }
}

void KinematicArbiterNode::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg,
    const std::string& sensor_id) {

  RCLCPP_DEBUG(this->get_logger(), "Received IMU from sensor %s", sensor_id.c_str());

  // Get sensor transform from TF tree
  if (!updateSensorTransform(sensor_id, msg->header.frame_id, msg->header.stamp)) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to get transform for sensor %s from frame %s to %s",
                sensor_id.c_str(), msg->header.frame_id.c_str(), body_frame_id_.c_str());
  }

  // Process the measurement
  bool success = filter_wrapper_->processImu(sensor_id, *msg);

  if (!success) {
    RCLCPP_WARN(this->get_logger(), "Failed to process IMU measurement from sensor %s",
                sensor_id.c_str());
  }
}

bool KinematicArbiterNode::updateSensorTransform(
    const std::string& sensor_id,
    const std::string& frame_id,
    const rclcpp::Time& time) {

  // Skip transform lookup if sensor frame is the same as body frame
  if (frame_id == body_frame_id_) {
    // Create identity transform
    geometry_msgs::msg::TransformStamped identity_transform;
    identity_transform.header.stamp = time;
    identity_transform.header.frame_id = body_frame_id_;
    identity_transform.child_frame_id = frame_id;
    identity_transform.transform.rotation.w = 1.0;  // Identity quaternion

    // Set identity transform in filter wrapper
    return filter_wrapper_->setSensorTransform(sensor_id, identity_transform);
  }

  try {
    // Look up transform from sensor frame to body frame
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped = tf_buffer_->lookupTransform(
        body_frame_id_,                 // target frame (body)
        frame_id,                       // source frame (sensor)
        time,                          // time
        rclcpp::Duration::from_seconds(0.1)  // timeout
    );

    // Set the transform in the filter wrapper
    if (!filter_wrapper_->setSensorTransform(sensor_id, transform_stamped)) {
      RCLCPP_WARN(this->get_logger(), "Failed to set transform for sensor %s", sensor_id.c_str());
      return false;
    }

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

  // Set the frame IDs for the messages
  pose_estimate.header.frame_id = world_frame_id_;
  velocity_estimate.header.frame_id = body_frame_id_;
  accel_estimate.header.frame_id = body_frame_id_;

  // Publish
  pose_pub_->publish(pose_estimate);
  velocity_pub_->publish(velocity_estimate);
  accel_pub_->publish(accel_estimate);
}

template<typename MsgType>
void KinematicArbiterNode::initializeSensors(
    const std::string& param_name,
    std::vector<SensorSubscription>& subscription_list,
    std::string (wrapper::FilterWrapper::*register_func)(const std::string&)) {

  // Look for sensor parameters
  this->declare_parameter(param_name, std::vector<std::string>());
  auto sensors = this->get_parameter(param_name).as_string_array();

  // Extract the sensor type from the parameter name (remove '_sensors' suffix)
  std::string sensor_type = param_name.substr(0, param_name.find("_sensors"));

  for (const auto& sensor_name : sensors) {
    std::string topic_param = "sensors." + sensor_name + ".topic";
    this->declare_parameter(topic_param, "");
    std::string topic = this->get_parameter(topic_param).as_string();

    if (!topic.empty()) {
      // Register the sensor with the filter using the provided register function
      std::string sensor_id = (filter_wrapper_.get()->*register_func)(sensor_name);

      // Create the subscription pair
      createSensorPair<MsgType>(sensor_type, sensor_id, topic, subscription_list);
    }
  }
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

  // Create subscription based on sensor type
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
  else if constexpr (std::is_same_v<MsgType, geometry_msgs::msg::PoseStamped>) {
    sub.subscription = this->create_subscription<MsgType>(
        topic, 10,
        [this, sensor_id](const std::shared_ptr<const MsgType> msg) {
          this->poseCallback(std::const_pointer_cast<MsgType>(msg), sensor_id);
        });
  }
  else if constexpr (std::is_same_v<MsgType, geometry_msgs::msg::TwistStamped>) {
    sub.subscription = this->create_subscription<MsgType>(
        topic, 10,
        [this, sensor_id](const std::shared_ptr<const MsgType> msg) {
          this->velocityCallback(std::const_pointer_cast<MsgType>(msg), sensor_id);
        });
  }
  else if constexpr (std::is_same_v<MsgType, sensor_msgs::msg::Imu>) {
    sub.subscription = this->create_subscription<MsgType>(
        topic, 10,
        [this, sensor_id](const std::shared_ptr<const MsgType> msg) {
          this->imuCallback(std::const_pointer_cast<MsgType>(msg), sensor_id);
        });
  }

  // Add the subscription to the list
  subscription_list.push_back(sub);

  RCLCPP_INFO(this->get_logger(), "Registered %s sensor '%s' on topic '%s'",
              sensor_type.c_str(), sensor_id.c_str(), topic.c_str());
}

} // namespace ros2
} // namespace kinematic_arbiter
