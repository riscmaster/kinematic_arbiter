#include "kinematic_arbiter/ros2/filter_wrapper.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"

namespace kinematic_arbiter {
namespace ros2 {

FilterWrapper::FilterWrapper(
    rclcpp::Node* node,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer,
    const ::kinematic_arbiter::models::RigidBodyStateModel::Params& model_params,
    const std::string& body_frame_id,
    const std::string& world_frame_id)
  : node_(node),
    tf_buffer_(tf_buffer),
    body_frame_id_(body_frame_id),
    world_frame_id_(world_frame_id) {

  // Create state model
  auto state_model = std::make_shared<::kinematic_arbiter::models::RigidBodyStateModel>(model_params);

  // Create the filter
  filter_= std::make_shared<MediatedKalmanFilter>(state_model);

  RCLCPP_INFO(node_->get_logger(),
      "Created FilterWrapper with body frame '%s' and world frame '%s'",
      body_frame_id_.c_str(), world_frame_id_.c_str());

  time_manager_ = std::make_shared<utils::TimeManager>();
  time_manager_->setReferenceTime(node_->now());
}

void FilterWrapper::setMaxDelayWindow(double seconds) {
  filter_->SetMaxDelayWindow(seconds);
}

bool FilterWrapper::addPositionSensor(
    const std::string& sensor_name,
    const std::string& topic,
    const std::string& sensor_frame_id,
    double p2m_noise_ratio,
    const std::string& mediation_action) {

  try {
    auto handler = std::make_shared<PositionSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        time_manager_,
        sensor_name,
        topic,
        sensor_frame_id,
        world_frame_id_,  // Use world frame as reference
        body_frame_id_);

    // Set validation parameters
    handler->setValidationParams(p2m_noise_ratio, stringToMediationAction(mediation_action));

    position_handlers_.push_back(handler);
    RCLCPP_INFO(node_->get_logger(), "Added position sensor '%s' on topic '%s'",
                sensor_name.c_str(), topic.c_str());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to add position sensor '%s': %s",
                sensor_name.c_str(), e.what());
    return false;
  }
}

bool FilterWrapper::addPoseSensor(
    const std::string& sensor_name,
    const std::string& topic,
    const std::string& sensor_frame_id,
    double p2m_noise_ratio,
    const std::string& mediation_action) {

  try {
    auto handler = std::make_shared<PoseSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        time_manager_,
        sensor_name,
        topic,
        sensor_frame_id,
        world_frame_id_,  // Use world frame as reference
        body_frame_id_);

    // Set validation parameters
    handler->setValidationParams(p2m_noise_ratio, stringToMediationAction(mediation_action));

    pose_handlers_.push_back(handler);
    RCLCPP_INFO(node_->get_logger(), "Added pose sensor '%s' on topic '%s'",
                sensor_name.c_str(), topic.c_str());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to add pose sensor '%s': %s",
                sensor_name.c_str(), e.what());
    return false;
  }
}

bool FilterWrapper::addVelocitySensor(
    const std::string& sensor_name,
    const std::string& topic,
    const std::string& sensor_frame_id,
    double p2m_noise_ratio,
    const std::string& mediation_action) {

  try {
    auto handler = std::make_shared<VelocitySensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        time_manager_,
        sensor_name,
        topic,
        sensor_frame_id,
        sensor_frame_id,  // Velocities use sensor frame as reference
        body_frame_id_);

    // Set validation parameters
    handler->setValidationParams(p2m_noise_ratio, stringToMediationAction(mediation_action));

    velocity_handlers_.push_back(handler);
    RCLCPP_INFO(node_->get_logger(), "Added velocity sensor '%s' on topic '%s'",
                sensor_name.c_str(), topic.c_str());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to add velocity sensor '%s': %s",
                sensor_name.c_str(), e.what());
    return false;
  }
}

bool FilterWrapper::addImuSensor(
    const std::string& sensor_name,
    const std::string& topic,
    const std::string& sensor_frame_id,
    double p2m_noise_ratio,
    const std::string& mediation_action) {

  try {
    auto handler = std::make_shared<ImuSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        time_manager_,
        sensor_name,
        topic,
        sensor_frame_id,
        sensor_frame_id,  // IMU measurements are in sensor frame
        body_frame_id_);

    // Set validation parameters
    handler->setValidationParams(p2m_noise_ratio, stringToMediationAction(mediation_action));

    imu_handlers_.push_back(handler);
    RCLCPP_INFO(node_->get_logger(), "Added IMU sensor '%s' on topic '%s'",
                sensor_name.c_str(), topic.c_str());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to add IMU sensor '%s': %s",
                sensor_name.c_str(), e.what());
    return false;
  }
}

geometry_msgs::msg::PoseWithCovarianceStamped FilterWrapper::getPoseEstimate() {

  double time_sec = filter_->GetCurrentTime();

  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::PoseWithCovarianceStamped msg;
  msg.header.stamp = time_manager_->filterTimeToRosTime(time_sec);
  msg.header.frame_id = world_frame_id_;

  // Position
  msg.pose.pose.position.x = state(StateIndex::Position::X);
  msg.pose.pose.position.y = state(StateIndex::Position::Y);
  msg.pose.pose.position.z = state(StateIndex::Position::Z);

  // Orientation (quaternion)
  msg.pose.pose.orientation.w = state(StateIndex::Quaternion::W);
  msg.pose.pose.orientation.x = state(StateIndex::Quaternion::X);
  msg.pose.pose.orientation.y = state(StateIndex::Quaternion::Y);
  msg.pose.pose.orientation.z = state(StateIndex::Quaternion::Z);

  // Fill covariance (position and orientation blocks)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Position covariance (top-left 3x3 block)
      msg.pose.covariance[i*6 + j] = covariance(StateIndex::Position::X + i, StateIndex::Position::X + j);

      // Orientation covariance (bottom-right 3x3 block)
      // Note: This is an approximation as we're mapping quaternion to Euler angle representation
      msg.pose.covariance[(i+3)*6 + (j+3)] = covariance(StateIndex::Quaternion::X + i, StateIndex::Quaternion::X + j);
    }
  }

  return msg;
}

geometry_msgs::msg::TwistWithCovarianceStamped FilterWrapper::getVelocityEstimate() {

  double time_sec = filter_->GetCurrentTime();


  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::TwistWithCovarianceStamped msg;
  msg.header.stamp = time_manager_->filterTimeToRosTime(time_sec);
  msg.header.frame_id = body_frame_id_;  // Velocities are in body frame

  // Linear velocity
  msg.twist.twist.linear.x = state(StateIndex::LinearVelocity::X);
  msg.twist.twist.linear.y = state(StateIndex::LinearVelocity::Y);
  msg.twist.twist.linear.z = state(StateIndex::LinearVelocity::Z);

  // Angular velocity
  msg.twist.twist.angular.x = state(StateIndex::AngularVelocity::X);
  msg.twist.twist.angular.y = state(StateIndex::AngularVelocity::Y);
  msg.twist.twist.angular.z = state(StateIndex::AngularVelocity::Z);

  // Fill covariance
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Linear velocity covariance (top-left 3x3 block)
      msg.twist.covariance[i*6 + j] = covariance(StateIndex::LinearVelocity::X + i, StateIndex::LinearVelocity::X + j);

      // Angular velocity covariance (bottom-right 3x3 block)
      msg.twist.covariance[(i+3)*6 + (j+3)] = covariance(StateIndex::AngularVelocity::X + i, StateIndex::AngularVelocity::X + j);
    }
  }

  return msg;
}

geometry_msgs::msg::AccelWithCovarianceStamped FilterWrapper::getAccelerationEstimate() {

  double time_sec = filter_->GetCurrentTime();

  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::AccelWithCovarianceStamped msg;
  msg.header.stamp = time_manager_->filterTimeToRosTime(time_sec);
  msg.header.frame_id = body_frame_id_;  // Accelerations are in body frame

  // Linear acceleration
  msg.accel.accel.linear.x = state(StateIndex::LinearAcceleration::X);
  msg.accel.accel.linear.y = state(StateIndex::LinearAcceleration::Y);
  msg.accel.accel.linear.z = state(StateIndex::LinearAcceleration::Z);

  // Angular acceleration
  msg.accel.accel.angular.x = state(StateIndex::AngularAcceleration::X);
  msg.accel.accel.angular.y = state(StateIndex::AngularAcceleration::Y);
  msg.accel.accel.angular.z = state(StateIndex::AngularAcceleration::Z);

  // Fill covariance
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Linear acceleration covariance (top-left 3x3 block)
      msg.accel.covariance[i*6 + j] = covariance(StateIndex::LinearAcceleration::X + i, StateIndex::LinearAcceleration::X + j);

      // Angular acceleration covariance (bottom-right 3x3 block)
      msg.accel.covariance[(i+3)*6 + (j+3)] = covariance(StateIndex::AngularAcceleration::X + i, StateIndex::AngularAcceleration::X + j);
    }
  }

  return msg;
}

void FilterWrapper::setPoseEstimate(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  auto state = filter_->GetStateEstimate();
  state(StateIndex::Position::X) = msg->pose.position.x;
  state(StateIndex::Position::Y) = msg->pose.position.y;
  state(StateIndex::Position::Z) = msg->pose.position.z;
  state(StateIndex::Quaternion::W) = msg->pose.orientation.w;
  state(StateIndex::Quaternion::X) = msg->pose.orientation.x;
  state(StateIndex::Quaternion::Y) = msg->pose.orientation.y;
  state(StateIndex::Quaternion::Z) = msg->pose.orientation.z;
  filter_->SetStateEstimate(state, time_manager_->rosTimeToFilterTime(msg->header.stamp));
}

void FilterWrapper::setVelocityEstimate(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  auto state = filter_->GetStateEstimate();
  state(StateIndex::LinearVelocity::X) = msg->twist.linear.x;
  state(StateIndex::LinearVelocity::Y) = msg->twist.linear.y;
  state(StateIndex::LinearVelocity::Z) = msg->twist.linear.z;
  state(StateIndex::AngularVelocity::X) = msg->twist.angular.x;
  state(StateIndex::AngularVelocity::Y) = msg->twist.angular.y;
  state(StateIndex::AngularVelocity::Z) = msg->twist.angular.z;
  filter_->SetStateEstimate(state, time_manager_->rosTimeToFilterTime(msg->header.stamp));
}

bool FilterWrapper::isInitialized() const {
  // Check if the filter itself is ready (has fully initialized state)
  bool filter_ready = filter_->IsInitialized();

  // Check if time manager is initialized
  bool time_ready = time_manager_->isInitialized();

  // Both need to be ready for the wrapper to be considered initialized
  return filter_ready && time_ready;
}

// Helper method to convert string to MediationAction enum
kinematic_arbiter::core::MediationAction kinematic_arbiter::ros2::FilterWrapper::stringToMediationAction(
    const std::string& action_str) const {
  using namespace kinematic_arbiter::core;

  if (action_str == "force_accept") {
    return MediationAction::ForceAccept;
  } else if (action_str == "adjust_covariance") {
    return MediationAction::AdjustCovariance;
  } else if (action_str == "reject") {
    return MediationAction::Reject;
  } else {
    RCLCPP_WARN(node_->get_logger(), "Unknown mediation action '%s', defaulting to 'force_accept'",
                action_str.c_str());
    return MediationAction::ForceAccept;
  }
}

} // namespace ros2
} // namespace kinematic_arbiter
