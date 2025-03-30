#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace kinematic_arbiter {
namespace ros2 {

KinematicArbiterNode::KinematicArbiterNode()
    : Node("kinematic_arbiter", rclcpp::NodeOptions().allow_undeclared_parameters(true)
                                                    .automatically_declare_parameters_from_overrides(true)) {

  // Print node name and namespace for debugging
  RCLCPP_INFO(this->get_logger(), "Node name: '%s', namespace: '%s'",
              this->get_name(), this->get_namespace());

  // Dump all available parameters at root level for debugging
  auto all_params = this->list_parameters({}, 0);
  RCLCPP_INFO(this->get_logger(), "Root parameters available (%zu):", all_params.names.size());
  for (const auto& name : all_params.names) {
    RCLCPP_INFO(this->get_logger(), "  %s", name.c_str());
  }

  // List all parameter prefixes
  auto all_prefixes = this->list_parameters({}, 1).prefixes;
  RCLCPP_INFO(this->get_logger(), "Parameter prefixes (%zu):", all_prefixes.size());
  for (const auto& prefix : all_prefixes) {
    RCLCPP_INFO(this->get_logger(), "  %s", prefix.c_str());
  }

  // Get basic parameters
  publish_rate_ = this->get_parameter("publish_rate").as_double();
  double max_delay_window = this->get_parameter("max_delay_window").as_double();
  world_frame_id_ = this->get_parameter("world_frame_id").as_string();
  body_frame_id_ = this->get_parameter("body_frame_id").as_string();

  // Get process noise window parameter (use get_parameter instead of declare_parameter)
  int process_noise_window = 500; // Default value
  if (this->has_parameter("process_noise_window")) {
    process_noise_window = this->get_parameter("process_noise_window").as_int();
  }
  RCLCPP_INFO(this->get_logger(), "Process noise window: %d", process_noise_window);

  // Get the topic names
  std::string pose_topic = this->get_parameter("pose_state_topic").as_string();
  std::string velocity_topic = this->get_parameter("velocity_state_topic").as_string();
  std::string acceleration_topic = this->get_parameter("acceleration_state_topic").as_string();

  // Set up TF components
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Create publishers
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      pose_topic, 10);
  velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
      velocity_topic, 10);
  accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelWithCovarianceStamped>(
      acceleration_topic, 10);

  // Create state model parameters
  ::kinematic_arbiter::models::RigidBodyStateModel::Params model_params;
  model_params.process_noise_window = process_noise_window;

  // Get initial process noise values with defaults
  model_params.position_uncertainty_per_second =
      this->get_parameter_or("position_uncertainty_per_second", 0.01);
  model_params.orientation_uncertainty_per_second =
      this->get_parameter_or("orientation_uncertainty_per_second", 0.01);
  model_params.linear_velocity_uncertainty_per_second =
      this->get_parameter_or("linear_velocity_uncertainty_per_second", 0.1);
  model_params.angular_velocity_uncertainty_per_second =
      this->get_parameter_or("angular_velocity_uncertainty_per_second", 0.1);
  model_params.linear_acceleration_uncertainty_per_second =
      this->get_parameter_or("linear_acceleration_uncertainty_per_second", 1.0);
  model_params.angular_acceleration_uncertainty_per_second =
      this->get_parameter_or("angular_acceleration_uncertainty_per_second", 1.0);

  RCLCPP_INFO(this->get_logger(), "Initial process noise settings:");
  RCLCPP_INFO(this->get_logger(), "  Position: %.4f", model_params.position_uncertainty_per_second);
  RCLCPP_INFO(this->get_logger(), "  Orientation: %.4f", model_params.orientation_uncertainty_per_second);
  RCLCPP_INFO(this->get_logger(), "  Linear velocity: %.4f", model_params.linear_velocity_uncertainty_per_second);
  RCLCPP_INFO(this->get_logger(), "  Angular velocity: %.4f", model_params.angular_velocity_uncertainty_per_second);
  RCLCPP_INFO(this->get_logger(), "  Linear acceleration: %.4f", model_params.linear_acceleration_uncertainty_per_second);
  RCLCPP_INFO(this->get_logger(), "  Angular acceleration: %.4f", model_params.angular_acceleration_uncertainty_per_second);

  // Create filter wrapper
  filter_wrapper_ = std::make_unique<FilterWrapper>(
      this,
      tf_buffer_,
      model_params,
      body_frame_id_,
      world_frame_id_);

  // Set max delay window
  filter_wrapper_->setMaxDelayWindow(max_delay_window);

  // Configure sensors - now using the direct approach with loaded parameters
  configureSensorsFromLoadedParams();

  // Create timer for publishing estimates
  publish_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&KinematicArbiterNode::publishEstimates, this));

  RCLCPP_INFO(this->get_logger(), "KinematicArbiterNode initialized with world frame '%s' and body frame '%s'",
             world_frame_id_.c_str(), body_frame_id_.c_str());

  // Add these lines to set up true state subscribers
  try {
    // Create subscribers for true pose and velocity if topics are specified
    if (this->has_parameter("truth_pose_topic")) {
      std::string truth_pose_topic = this->get_parameter("truth_pose_topic").as_string();
      if (!truth_pose_topic.empty()) {
        RCLCPP_INFO(this->get_logger(), "Subscribing to true pose topic: %s", truth_pose_topic.c_str());
        truth_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            truth_pose_topic, 10,
            std::bind(&KinematicArbiterNode::truthPoseCallback, this, std::placeholders::_1));
      }
    }

    if (this->has_parameter("truth_velocity_topic")) {
      std::string truth_velocity_topic = this->get_parameter("truth_velocity_topic").as_string();
      if (!truth_velocity_topic.empty()) {
        RCLCPP_INFO(this->get_logger(), "Subscribing to true velocity topic: %s", truth_velocity_topic.c_str());
        truth_velocity_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            truth_velocity_topic, 10,
            std::bind(&KinematicArbiterNode::truthVelocityCallback, this, std::placeholders::_1));
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN(this->get_logger(), "Error setting up true state subscribers: %s", e.what());
  }

  // Create reset service
  reset_service_ = this->create_service<std_srvs::srv::Trigger>(
      "~/reset",
      std::bind(&KinematicArbiterNode::handleResetService, this,
                std::placeholders::_1, std::placeholders::_2));
  RCLCPP_INFO(this->get_logger(), "Created reset service on '%s/reset'", this->get_namespace());
}

// Simplify the sensor configuration method to focus only on param loading
void KinematicArbiterNode::configureSensorsFromLoadedParams() {
  try {
    // Process position sensors
    if (this->has_parameter("position_sensors.position_sensor.topic")) {
      std::string topic = this->get_parameter("position_sensors.position_sensor.topic").as_string();
      std::string frame_id = this->get_parameter("position_sensors.position_sensor.frame_id").as_string();

      // Get validation parameters with defaults if not specified
      double p2m_ratio = 2.0;
      std::string mediation_action = "force_accept";

      if (this->has_parameter("position_sensors.position_sensor.p2m_noise_ratio")) {
        p2m_ratio = this->get_parameter("position_sensors.position_sensor.p2m_noise_ratio").as_double();
      }

      if (this->has_parameter("position_sensors.position_sensor.mediation_action")) {
        mediation_action = this->get_parameter("position_sensors.position_sensor.mediation_action").as_string();
      }

      RCLCPP_INFO(this->get_logger(), "Adding position sensor: topic='%s', frame_id='%s', p2m_ratio=%.2f, action='%s'",
                 topic.c_str(), frame_id.c_str(), p2m_ratio, mediation_action.c_str());
      filter_wrapper_->addPositionSensor("position_sensor", topic, frame_id, p2m_ratio, mediation_action);
    }

    // Process pose sensors
    if (this->has_parameter("pose_sensors.pose_sensor.topic")) {
      std::string topic = this->get_parameter("pose_sensors.pose_sensor.topic").as_string();
      std::string frame_id = this->get_parameter("pose_sensors.pose_sensor.frame_id").as_string();

      // Get validation parameters
      double p2m_ratio = 2.0;
      std::string mediation_action = "force_accept";

      if (this->has_parameter("pose_sensors.pose_sensor.p2m_noise_ratio")) {
        p2m_ratio = this->get_parameter("pose_sensors.pose_sensor.p2m_noise_ratio").as_double();
      }

      if (this->has_parameter("pose_sensors.pose_sensor.mediation_action")) {
        mediation_action = this->get_parameter("pose_sensors.pose_sensor.mediation_action").as_string();
      }

      RCLCPP_INFO(this->get_logger(), "Adding pose sensor: topic='%s', frame_id='%s', p2m_ratio=%.2f, action='%s'",
                 topic.c_str(), frame_id.c_str(), p2m_ratio, mediation_action.c_str());
      filter_wrapper_->addPoseSensor("pose_sensor", topic, frame_id, p2m_ratio, mediation_action);
    }

    // Process velocity sensors
    if (this->has_parameter("velocity_sensors.velocity_sensor.topic")) {
      std::string topic = this->get_parameter("velocity_sensors.velocity_sensor.topic").as_string();
      std::string frame_id = this->get_parameter("velocity_sensors.velocity_sensor.frame_id").as_string();

      // Get validation parameters
      double p2m_ratio = 2.0;
      std::string mediation_action = "force_accept";

      if (this->has_parameter("velocity_sensors.velocity_sensor.p2m_noise_ratio")) {
        p2m_ratio = this->get_parameter("velocity_sensors.velocity_sensor.p2m_noise_ratio").as_double();
      }

      if (this->has_parameter("velocity_sensors.velocity_sensor.mediation_action")) {
        mediation_action = this->get_parameter("velocity_sensors.velocity_sensor.mediation_action").as_string();
      }

      RCLCPP_INFO(this->get_logger(), "Adding velocity sensor: topic='%s', frame_id='%s', p2m_ratio=%.2f, action='%s'",
                 topic.c_str(), frame_id.c_str(), p2m_ratio, mediation_action.c_str());
      filter_wrapper_->addVelocitySensor("velocity_sensor", topic, frame_id, p2m_ratio, mediation_action);
    }

    // Process IMU sensors
    if (this->has_parameter("imu_sensors.imu_sensor.topic")) {
      std::string topic = this->get_parameter("imu_sensors.imu_sensor.topic").as_string();
      std::string frame_id = this->get_parameter("imu_sensors.imu_sensor.frame_id").as_string();

      // Get validation parameters
      double p2m_ratio = 2.0;
      std::string mediation_action = "force_accept";

      if (this->has_parameter("imu_sensors.imu_sensor.p2m_noise_ratio")) {
        p2m_ratio = this->get_parameter("imu_sensors.imu_sensor.p2m_noise_ratio").as_double();
      }

      if (this->has_parameter("imu_sensors.imu_sensor.mediation_action")) {
        mediation_action = this->get_parameter("imu_sensors.imu_sensor.mediation_action").as_string();
      }

      RCLCPP_INFO(this->get_logger(), "Adding IMU sensor: topic='%s', frame_id='%s', p2m_ratio=%.2f, action='%s'",
                 topic.c_str(), frame_id.c_str(), p2m_ratio, mediation_action.c_str());
      filter_wrapper_->addImuSensor("imu_sensor", topic, frame_id, p2m_ratio, mediation_action);
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Error configuring sensors: %s", e.what());
  }
}

void KinematicArbiterNode::publishEstimates() {

  // Check if filter is initialized
  if (!filter_wrapper_->isInitialized()) {
    static int init_message_count = 0;
    if (init_message_count++ % 10 == 0) {  // Only log every 10th message to avoid spam
      RCLCPP_INFO(this->get_logger(), "Filter not yet initialized");
    }
    return;
  }

  // Log first successful publication
  static bool first_publication = true;
  if (first_publication) {
    RCLCPP_INFO(this->get_logger(), "Filter initialized! Publishing first state estimate");
    auto state = filter_wrapper_->getPoseEstimate();
    RCLCPP_INFO(this->get_logger(), "Position: [%.3f, %.3f, %.3f]",
                state.pose.pose.position.x, state.pose.pose.position.y, state.pose.pose.position.z);
    first_publication = false;
  }

  // Get and publish pose
  auto pose_msg = filter_wrapper_->getPoseEstimate();
  pose_pub_->publish(pose_msg);

  // Get and publish velocity
  auto velocity_msg = filter_wrapper_->getVelocityEstimate();
  velocity_pub_->publish(velocity_msg);

  // Get and publish acceleration
  auto accel_msg = filter_wrapper_->getAccelerationEstimate();
  accel_pub_->publish(accel_msg);
}

void KinematicArbiterNode::truthPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  // Just store the latest message - metrics will be handled externally
  RCLCPP_DEBUG(this->get_logger(), "Received true pose at time: %f",
               rclcpp::Time(msg->header.stamp).seconds());
  filter_wrapper_->setPoseEstimate(msg);
}

void KinematicArbiterNode::truthVelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  // Just store the latest message - metrics will be handled externally
  RCLCPP_DEBUG(this->get_logger(), "Received true velocity at time: %f",
               rclcpp::Time(msg->header.stamp).seconds());
  filter_wrapper_->setVelocityEstimate(msg);
}

void KinematicArbiterNode::handleResetService(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {

  RCLCPP_INFO(this->get_logger(), "Received reset request");

  // Use the existing simple reset method
  filter_wrapper_->reset();

  response->success = true;
  response->message = "Filter reset successful";
  RCLCPP_INFO(this->get_logger(), "Filter reset successful");
}

} // namespace ros2
} // namespace kinematic_arbiter
