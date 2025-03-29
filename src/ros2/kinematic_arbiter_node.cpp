#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"

namespace kinematic_arbiter {
namespace ros2 {

KinematicArbiterNode::KinematicArbiterNode()
    : Node("kinematic_arbiter") {

  // Declare and get parameters
  this->declare_parameter("publish_rate", 50.0);
  this->declare_parameter("max_delay_window", 1.0);
  this->declare_parameter("world_frame_id", "map");
  this->declare_parameter("body_frame_id", "base_link");

  publish_rate_ = this->get_parameter("publish_rate").as_double();
  double max_delay_window = this->get_parameter("max_delay_window").as_double();
  world_frame_id_ = this->get_parameter("world_frame_id").as_string();
  body_frame_id_ = this->get_parameter("body_frame_id").as_string();

  // Set up TF components
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Create publishers
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "state/pose", 10);
  velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "state/velocity", 10);
  accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelWithCovarianceStamped>(
      "state/acceleration", 10);

  // Create state model parameters (using defaults for now)
  // In a real implementation, these should be configured via parameters
  ::kinematic_arbiter::models::RigidBodyStateModel::Params model_params;

  // Create filter wrapper
  filter_wrapper_ = std::make_unique<FilterWrapper>(
      this,
      tf_buffer_,
      model_params,
      body_frame_id_,
      world_frame_id_);

  // Set max delay window
  filter_wrapper_->setMaxDelayWindow(max_delay_window);

  // Configure sensors
  configureSensors();

  // Create timer for publishing estimates
  publish_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&KinematicArbiterNode::publishEstimates, this));

  RCLCPP_INFO(this->get_logger(),
      "KinematicArbiterNode initialized with world frame '%s' and body frame '%s'",
      world_frame_id_.c_str(), body_frame_id_.c_str());
}

void KinematicArbiterNode::configureSensors() {
  configurePositionSensors();
  configurePoseSensors();
  configureVelocitySensors();
  configureImuSensors();
}

void KinematicArbiterNode::configurePositionSensors() {
  if (!this->has_parameter("position_sensors")) {
    return;
  }

  auto sensors = this->get_parameter("position_sensors").as_string_array();
  for (const auto& config : sensors) {
    // Parse "sensor_id:topic:frame_id"
    std::stringstream ss(config);
    std::string sensor_id, topic, frame_id;

    std::getline(ss, sensor_id, ':');
    std::getline(ss, topic, ':');
    std::getline(ss, frame_id);

    if (sensor_id.empty() || topic.empty() || frame_id.empty()) {
      RCLCPP_WARN(this->get_logger(),
          "Invalid position sensor config: '%s'. Format should be 'sensor_id:topic:frame_id'",
          config.c_str());
      continue;
    }

    if (filter_wrapper_->addPositionSensor(sensor_id, topic, frame_id)) {
      RCLCPP_INFO(this->get_logger(), "Configured position sensor '%s'", sensor_id.c_str());
    }
  }
}

void KinematicArbiterNode::configurePoseSensors() {
  if (!this->has_parameter("pose_sensors")) {
    return;
  }

  auto sensors = this->get_parameter("pose_sensors").as_string_array();
  for (const auto& config : sensors) {
    // Parse "sensor_id:topic:frame_id"
    std::stringstream ss(config);
    std::string sensor_id, topic, frame_id;

    std::getline(ss, sensor_id, ':');
    std::getline(ss, topic, ':');
    std::getline(ss, frame_id);

    if (sensor_id.empty() || topic.empty() || frame_id.empty()) {
      RCLCPP_WARN(this->get_logger(),
          "Invalid pose sensor config: '%s'. Format should be 'sensor_id:topic:frame_id'",
          config.c_str());
      continue;
    }

    if (filter_wrapper_->addPoseSensor(sensor_id, topic, frame_id)) {
      RCLCPP_INFO(this->get_logger(), "Configured pose sensor '%s'", sensor_id.c_str());
    }
  }
}

void KinematicArbiterNode::configureVelocitySensors() {
  if (!this->has_parameter("velocity_sensors")) {
    return;
  }

  auto sensors = this->get_parameter("velocity_sensors").as_string_array();
  for (const auto& config : sensors) {
    // Parse "sensor_id:topic:frame_id"
    std::stringstream ss(config);
    std::string sensor_id, topic, frame_id;

    std::getline(ss, sensor_id, ':');
    std::getline(ss, topic, ':');
    std::getline(ss, frame_id);

    if (sensor_id.empty() || topic.empty() || frame_id.empty()) {
      RCLCPP_WARN(this->get_logger(),
          "Invalid velocity sensor config: '%s'. Format should be 'sensor_id:topic:frame_id'",
          config.c_str());
      continue;
    }

    if (filter_wrapper_->addVelocitySensor(sensor_id, topic, frame_id)) {
      RCLCPP_INFO(this->get_logger(), "Configured velocity sensor '%s'", sensor_id.c_str());
    }
  }
}

void KinematicArbiterNode::configureImuSensors() {
  if (!this->has_parameter("imu_sensors")) {
    return;
  }

  auto sensors = this->get_parameter("imu_sensors").as_string_array();
  for (const auto& config : sensors) {
    // Parse "sensor_id:topic:frame_id"
    std::stringstream ss(config);
    std::string sensor_id, topic, frame_id;

    std::getline(ss, sensor_id, ':');
    std::getline(ss, topic, ':');
    std::getline(ss, frame_id);

    if (sensor_id.empty() || topic.empty() || frame_id.empty()) {
      RCLCPP_WARN(this->get_logger(),
          "Invalid IMU sensor config: '%s'. Format should be 'sensor_id:topic:frame_id'",
          config.c_str());
      continue;
    }

    if (filter_wrapper_->addImuSensor(sensor_id, topic, frame_id)) {
      RCLCPP_INFO(this->get_logger(), "Configured IMU sensor '%s'", sensor_id.c_str());
    }
  }
}

void KinematicArbiterNode::publishEstimates() {
  // Get current ROS time
  auto current_time = this->now();

  // Predict filter to current time
  filter_wrapper_->predictTo(current_time);

  // Check if filter is initialized
  if (!filter_wrapper_->isInitialized()) {
    RCLCPP_DEBUG(this->get_logger(), "Filter not yet initialized, skipping publication");
    return;
  }

  // Get and publish pose
  auto pose_msg = filter_wrapper_->getPoseEstimate(current_time);
  pose_pub_->publish(pose_msg);

  // Get and publish velocity
  auto velocity_msg = filter_wrapper_->getVelocityEstimate(current_time);
  velocity_pub_->publish(velocity_msg);

  // Get and publish acceleration
  auto accel_msg = filter_wrapper_->getAccelerationEstimate(current_time);
  accel_pub_->publish(accel_msg);
}

} // namespace ros2
} // namespace kinematic_arbiter
