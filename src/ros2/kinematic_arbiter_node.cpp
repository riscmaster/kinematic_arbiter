#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"

namespace kinematic_arbiter {
namespace ros2 {

KinematicArbiterNode::KinematicArbiterNode()
    : Node("kinematic_arbiter") {

  // Declare and get parameters
  this->declare_parameter("publish_rate", 20.0);
  this->declare_parameter("max_delay_window", 0.5);
  this->declare_parameter("frame_id", "odom");

  publish_rate_ = this->get_parameter("publish_rate").as_double();
  max_delay_window_ = this->get_parameter("max_delay_window").as_double();
  frame_id_ = this->get_parameter("frame_id").as_string();

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
  pose_estimate.header.frame_id = frame_id_;
  velocity_estimate.header.frame_id = frame_id_;
  accel_estimate.header.frame_id = frame_id_;

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
        [this, sensor_id](const std::shared_ptr<const MsgType> msg) {  // Fixed: updated callback signature
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
