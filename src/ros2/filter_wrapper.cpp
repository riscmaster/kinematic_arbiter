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
}

void FilterWrapper::setMaxDelayWindow(double seconds) {
  filter_->SetMaxDelayWindow(seconds);
}

bool FilterWrapper::addPositionSensor(
    const std::string& sensor_name,
    const std::string& topic,
    const std::string& sensor_frame_id) {

  try {
    auto handler = std::make_shared<PositionSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        sensor_name,
        topic,
        sensor_frame_id,
        world_frame_id_,  // Use world frame as reference
        body_frame_id_);

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
    const std::string& sensor_frame_id) {

  try {
    auto handler = std::make_shared<PoseSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        sensor_name,
        topic,
        sensor_frame_id,
        world_frame_id_,  // Use world frame as reference
        body_frame_id_);

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
    const std::string& sensor_frame_id) {

  try {
    auto handler = std::make_shared<VelocitySensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        sensor_name,
        topic,
        sensor_frame_id,
        sensor_frame_id,  // Velocities use sensor frame as reference
        body_frame_id_);

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
    const std::string& sensor_frame_id) {

  try {
    auto handler = std::make_shared<ImuSensorHandler>(
        node_,
        filter_,
        tf_buffer_,
        sensor_name,
        topic,
        sensor_frame_id,
        sensor_frame_id,  // IMU measurements are in sensor frame
        body_frame_id_);

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

geometry_msgs::msg::PoseWithCovarianceStamped FilterWrapper::getPoseEstimate(
    const rclcpp::Time& time) {

  double time_sec = utils::rosTimeToSeconds(time);
  if (time_sec == 0.0) {
    time_sec = filter_->GetCurrentTime();
  }

  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::PoseWithCovarianceStamped msg;
  msg.header.stamp = utils::doubleTimeToRosTime(time_sec);
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

geometry_msgs::msg::TwistWithCovarianceStamped FilterWrapper::getVelocityEstimate(
    const rclcpp::Time& time) {

  double time_sec = utils::rosTimeToSeconds(time);
  if (time_sec == 0.0) {
    time_sec = filter_->GetCurrentTime();
  }

  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::TwistWithCovarianceStamped msg;
  msg.header.stamp = utils::doubleTimeToRosTime(time_sec);
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

geometry_msgs::msg::AccelWithCovarianceStamped FilterWrapper::getAccelerationEstimate(
    const rclcpp::Time& time) {

  double time_sec = utils::rosTimeToSeconds(time);
  if (time_sec == 0.0) {
    time_sec = filter_->GetCurrentTime();
  }

  // Get state and covariance at specified time
  auto state = filter_->GetStateEstimate(time_sec);
  auto covariance = filter_->GetStateCovariance();

  // Create message
  geometry_msgs::msg::AccelWithCovarianceStamped msg;
  msg.header.stamp = utils::doubleTimeToRosTime(time_sec);
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

void FilterWrapper::predictTo(const rclcpp::Time& time) {
  double time_sec = utils::rosTimeToSeconds(time);
  filter_->PredictNewReference(time_sec);
}

bool FilterWrapper::isInitialized() const {
  return filter_->IsInitialized();
}

} // namespace ros2
} // namespace kinematic_arbiter
