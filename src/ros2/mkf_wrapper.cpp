#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace wrapper {
using namespace kinematic_arbiter::core;
using namespace kinematic_arbiter::sensors;
// Global constant to offset times to avoid negative values
// Large enough to ensure all test times are positive
const double TIME_OFFSET = 1000000.0;  // 1 million seconds

// Constructor
FilterWrapper::FilterWrapper(const kinematic_arbiter::models::RigidBodyStateModel::Params& model_params) {
  // Create state model
  auto state_model = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>(model_params);

  // Create the filter
  filter_ = std::make_shared<kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>>(state_model);
}

std::string FilterWrapper::registerSensor(const std::string& name, kinematic_arbiter::core::SensorType type) {
  // Check if a sensor with this name already exists with the same type
  for (const auto& pair : sensors_) {
    if (pair.first == name && pair.second.type == type) {
      // Return existing ID if already registered with same name and type
      return name;
    }
  }

  // Create appropriate sensor model based on type
  std::shared_ptr<kinematic_arbiter::core::MeasurementModelInterface> sensor;

  switch (type) {
    case kinematic_arbiter::core::SensorType::Position:
      sensor = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();
      break;
    case kinematic_arbiter::core::SensorType::Pose:
      sensor = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();
      break;
    case kinematic_arbiter::core::SensorType::BodyVelocity:
      sensor = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();
      break;
    case kinematic_arbiter::core::SensorType::Imu:
      sensor = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();
      break;
    default:
      throw std::invalid_argument("Unsupported sensor type");
  }

  // Add the sensor to the filter
  size_t idx = filter_->AddSensor(sensor);

  // Use name directly as the ID, optionally make it unique if needed
  std::string unique_id = name;
  if (sensors_.find(unique_id) != sensors_.end()) {
    // If same ID exists but different type, make it unique
    unique_id = name + "_" + kinematic_arbiter::core::SensorTypeToString(type);
  }

  // Store the sensor info
  sensors_[unique_id] = {idx, type};

  return unique_id;
}

void FilterWrapper::setMaxDelayWindow(double seconds) {
  filter_->SetMaxDelayWindow(seconds);
}

/**
 * @brief Process position measurement
 */
bool FilterWrapper::processPosition(const std::string& sensor_id, const geometry_msgs::msg::PointStamped& msg) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != SensorType::Position) {
    std::cerr << "Invalid sensor ID or type mismatch: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  double timestamp = rosTimeToSeconds(msg.header.stamp);

  // Convert ROS message to Eigen vector
  Eigen::Vector3d position(msg.point.x, msg.point.y, msg.point.z);

  // Use non-templated ProcessMeasurementByIndex
  return filter_->ProcessMeasurementByIndex(info.index, position, timestamp);
}

/**
 * @brief Process pose measurement
 */
bool FilterWrapper::processPose(const std::string& sensor_id, const geometry_msgs::msg::PoseStamped& msg) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != SensorType::Pose) {
    std::cerr << "Invalid sensor ID or type mismatch: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  double timestamp = rosTimeToSeconds(msg.header.stamp);

  PoseSensorModel::Vector pose;
  pose[PoseSensorModel::MeasurementIndex::X] = msg.pose.position.x;
  pose[PoseSensorModel::MeasurementIndex::Y] = msg.pose.position.y;
  pose[PoseSensorModel::MeasurementIndex::Z] = msg.pose.position.z;
  pose[PoseSensorModel::MeasurementIndex::QW] = msg.pose.orientation.w;
  pose[PoseSensorModel::MeasurementIndex::QX] = msg.pose.orientation.x;
  pose[PoseSensorModel::MeasurementIndex::QY] = msg.pose.orientation.y;
  pose[PoseSensorModel::MeasurementIndex::QZ] = msg.pose.orientation.z;
  // Use non-templated ProcessMeasurementByIndex
  return filter_->ProcessMeasurementByIndex(info.index, pose, timestamp);
}

/**
 * @brief Process body velocity measurement
 */
bool FilterWrapper::processBodyVelocity(const std::string& sensor_id, const geometry_msgs::msg::TwistStamped& msg) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != SensorType::BodyVelocity) {
    std::cerr << "Invalid sensor ID or type mismatch: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  double timestamp = rosTimeToSeconds(msg.header.stamp);

  BodyVelocitySensorModel::Vector velocity;
  velocity[BodyVelocitySensorModel::MeasurementIndex::VX] = msg.twist.linear.x;
  velocity[BodyVelocitySensorModel::MeasurementIndex::VY] = msg.twist.linear.y;
  velocity[BodyVelocitySensorModel::MeasurementIndex::VZ] = msg.twist.linear.z;
  velocity[BodyVelocitySensorModel::MeasurementIndex::WX] = msg.twist.angular.x;
  velocity[BodyVelocitySensorModel::MeasurementIndex::WY] = msg.twist.angular.y;
  velocity[BodyVelocitySensorModel::MeasurementIndex::WZ] = msg.twist.angular.z;

  // Use non-templated ProcessMeasurementByIndex
  return filter_->ProcessMeasurementByIndex(info.index, velocity, timestamp);
}

  /**
 * @brief Process IMU measurement
 */
bool FilterWrapper::processImu(const std::string& sensor_id, const sensor_msgs::msg::Imu& msg) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != SensorType::Imu) {
    std::cerr << "Invalid sensor ID or type mismatch: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  double timestamp = rosTimeToSeconds(msg.header.stamp);

  ImuSensorModel::Vector imu;
  imu[ImuSensorModel::MeasurementIndex::AX] = msg.linear_acceleration.x;
  imu[ImuSensorModel::MeasurementIndex::AY] = msg.linear_acceleration.y;
  imu[ImuSensorModel::MeasurementIndex::AZ] = msg.linear_acceleration.z;
  imu[ImuSensorModel::MeasurementIndex::GX] = msg.angular_velocity.x;
  imu[ImuSensorModel::MeasurementIndex::GY] = msg.angular_velocity.y;
  imu[ImuSensorModel::MeasurementIndex::GZ] = msg.angular_velocity.z;

  // Use non-templated ProcessMeasurementByIndex
  return filter_->ProcessMeasurementByIndex(info.index, imu, timestamp);
}

/**
 * @brief Get expected position measurement for visualization
 */
geometry_msgs::msg::PoseWithCovarianceStamped FilterWrapper::getExpectedPosition(const std::string& sensor_id) {
  geometry_msgs::msg::PoseWithCovarianceStamped result;

  // Use filter's current time if available, otherwise use system time
  if (filter_->IsInitialized()) {
    result.header.stamp = doubleTimeToRosTime(filter_->GetCurrentTime());
  } else {
    result.header.stamp = rclcpp::Clock().now();
  }

  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != SensorType::Position) {
    std::cerr << "Invalid sensor ID or type mismatch: " << sensor_id << std::endl;
    return result;
  }

  const SensorInfo& info = it->second;

  // Use non-templated GetExpectedMeasurementByIndex
  MeasurementModelInterface::DynamicVector expected;
  if (!filter_->GetExpectedMeasurementByIndex(info.index, expected)) {
    std::cerr << "Failed to get expected measurement for sensor: " << sensor_id << std::endl;
    return result;
  }

  // Use non-templated GetSensorCovarianceByIndex
  MeasurementModelInterface::DynamicCovariance cov;
  filter_->GetSensorCovarianceByIndex(info.index, cov);

  // Fill position part of the message (ensure expected has 3 elements)
  if (expected.size() == 3) {
    result.pose.pose.position.x = expected(0);
    result.pose.pose.position.y = expected(1);
    result.pose.pose.position.z = expected(2);
  }

  // Fill orientation with identity quaternion (position sensors don't have orientation)
  result.pose.pose.orientation.w = 1.0;

  // Fill position covariance (3x3 block in the upper left corner)
  if (cov.rows() >= 3 && cov.cols() >= 3) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        result.pose.covariance[i*6 + j] = cov(i, j);
      }
    }
  }

  return result;
}

geometry_msgs::msg::PoseWithCovarianceStamped FilterWrapper::getPoseEstimate(const rclcpp::Time& time) {
  double timestamp = rosTimeToSeconds(time);
  StateVector state = filter_->GetStateEstimate(timestamp);
  StateMatrix covariance = filter_->GetStateCovariance(timestamp);

  geometry_msgs::msg::PoseWithCovarianceStamped msg;
  msg.header.stamp = time;
  msg.header.frame_id = "map"; // Should be configurable

  // Extract position
  Eigen::Vector3d position = state.segment<3>(StateIndex::Position::Begin());

  // Extract orientation
  Eigen::Quaterniond orientation(
      state(StateIndex::Quaternion::W),
      state(StateIndex::Quaternion::X),
      state(StateIndex::Quaternion::Y),
      state(StateIndex::Quaternion::Z)
  );

  // Convert to message types using tf2_eigen
  msg.pose.pose.position = tf2::toMsg(position);
  msg.pose.pose.orientation = tf2::toMsg(orientation);

  // Extract position and orientation covariance
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // Position covariance (top-left 3x3 block)
      int pi = StateIndex::Position::X + i;
      int pj = StateIndex::Position::X + j;
      msg.pose.covariance[i*6 + j] = covariance(pi, pj);

      // Orientation covariance is more complex due to quaternion representation
      // This is a simplified approach; a proper conversion would involve
      // transforming from quaternion space to rotation vector space
      // For now, just copy the quaternion elements' covariance
      int qi = StateIndex::Quaternion::X + i;
      int qj = StateIndex::Quaternion::X + j;
      msg.pose.covariance[(i+3)*6 + (j+3)] = covariance(qi, qj);
    }
  }

  return msg;
}

geometry_msgs::msg::TwistWithCovarianceStamped FilterWrapper::getVelocityEstimate(const rclcpp::Time& time) {
  double timestamp = rosTimeToSeconds(time);
  StateVector state = filter_->GetStateEstimate(timestamp);
  StateMatrix covariance = filter_->GetStateCovariance(timestamp);

  geometry_msgs::msg::TwistWithCovarianceStamped msg;
  msg.header.stamp = time;
  msg.header.frame_id = "base_link"; // Should be configurable

  // Extract linear and angular velocity
  Eigen::Vector3d lin_velocity = state.segment<3>(StateIndex::LinearVelocity::Begin());
  Eigen::Vector3d ang_velocity = state.segment<3>(StateIndex::AngularVelocity::Begin());

  // Convert to message types using tf2_eigen
  tf2::toMsg(lin_velocity, msg.twist.twist.linear);
  tf2::toMsg(ang_velocity, msg.twist.twist.angular);

  // Extract velocity covariance
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // Linear velocity covariance (top-left 3x3 block)
      int li = StateIndex::LinearVelocity::X + i;
      int lj = StateIndex::LinearVelocity::X + j;
      msg.twist.covariance[i*6 + j] = covariance(li, lj);

      // Angular velocity covariance (bottom-right 3x3 block)
      int ai = StateIndex::AngularVelocity::X + i;
      int aj = StateIndex::AngularVelocity::X + j;
      msg.twist.covariance[(i+3)*6 + (j+3)] = covariance(ai, aj);
    }
  }

  return msg;
}

geometry_msgs::msg::AccelWithCovarianceStamped FilterWrapper::getAccelerationEstimate(const rclcpp::Time& time) {
  double timestamp = rosTimeToSeconds(time);
  StateVector state = filter_->GetStateEstimate(timestamp);
  StateMatrix covariance = filter_->GetStateCovariance(timestamp);

  geometry_msgs::msg::AccelWithCovarianceStamped msg;
  msg.header.stamp = time;
  msg.header.frame_id = "base_link"; // Should be configurable

  // Extract linear and angular acceleration
  Eigen::Vector3d lin_accel = state.segment<3>(StateIndex::LinearAcceleration::Begin());
  Eigen::Vector3d ang_accel = state.segment<3>(StateIndex::AngularAcceleration::Begin());

  // Convert to message types using tf2_eigen
  tf2::toMsg(lin_accel, msg.accel.accel.linear);
  tf2::toMsg(ang_accel, msg.accel.accel.angular);

  // Extract acceleration covariance
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // Linear acceleration covariance (top-left 3x3 block)
      int li = StateIndex::LinearAcceleration::X + i;
      int lj = StateIndex::LinearAcceleration::X + j;
      msg.accel.covariance[i*6 + j] = covariance(li, lj);

      // Angular acceleration covariance (bottom-right 3x3 block)
      int ai = StateIndex::AngularAcceleration::X + i;
      int aj = StateIndex::AngularAcceleration::X + j;
      msg.accel.covariance[(i+3)*6 + (j+3)] = covariance(ai, aj);
    }
  }

  return msg;
}

void FilterWrapper::predictTo(const rclcpp::Time& time) {
  double timestamp = rosTimeToSeconds(time);
  filter_->PredictNewReference(timestamp);
}

// Utility conversion methods
double FilterWrapper::rosTimeToSeconds(const rclcpp::Time& time) {
  // Convert to nanoseconds, then to seconds with double precision
  double seconds_with_offset = time.nanoseconds() * 1e-9;
  return seconds_with_offset - TIME_OFFSET;
}

/**
 * @brief Set the transform from sensor to body frame for a sensor
 *
 * @param sensor_id The sensor identifier
 * @param transform The transform from sensor to body frame (ROS2 format)
 * @return true if successful, false otherwise
 */
bool FilterWrapper::setSensorTransform(const std::string& sensor_id,
                                      const geometry_msgs::msg::TransformStamped& transform) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end()) {
    return false;
  }

  const SensorInfo& info = it->second;

  // Convert ROS2 transform to Eigen
  Eigen::Isometry3d eigen_transform = tf2::transformToEigen(transform.transform);

  // Use the filter's helper method to set the transform
  if (!filter_->SetSensorPoseInBodyFrameByIndex(info.index, eigen_transform)) {
    std::cerr << "Failed to set transform for sensor '" << sensor_id << "'" << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief Get the transform from sensor to body frame for a sensor
 *
 * @param sensor_id The sensor identifier
 * @param[out] transform The transform from sensor to body frame (ROS2 format)
 * @return true if successful, false otherwise
 */
bool FilterWrapper::getSensorTransform(const std::string& sensor_id, const std::string& body_frame_id,
                                      geometry_msgs::msg::TransformStamped& transform) const {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end()) {
    std::cerr << "Invalid sensor ID: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;

  // Get Eigen transform from the filter
  Eigen::Isometry3d eigen_transform;
  if (!filter_->GetSensorPoseInBodyFrameByIndex(info.index, eigen_transform)) {
    std::cerr << "Failed to get transform for sensor '" << sensor_id << "'" << std::endl;
    return false;
  }

  // Convert Eigen transform to ROS2
  transform = tf2::eigenToTransform(eigen_transform);

  // Set header with filter's time if available
  if (filter_->IsInitialized()) {
    transform.header.stamp = doubleTimeToRosTime(filter_->GetCurrentTime());
  } else {
    transform.header.stamp = rclcpp::Clock().now();
  }
  transform.header.frame_id = body_frame_id;
  transform.child_frame_id = sensor_id;

  return true;
}

/**
 * @brief Convert double time to ROS time
 *
 * @param time Time in seconds
 * @return ROS time
 */
rclcpp::Time FilterWrapper::doubleTimeToRosTime(double time) const {
  // Add offset to ensure positive time
  double time_with_offset = time + TIME_OFFSET;

  // Convert to nanoseconds and use the nanoseconds constructor
  int64_t nanoseconds = static_cast<int64_t>(time_with_offset * 1e9);

  // Create time from nanoseconds
  return rclcpp::Time(nanoseconds);
}

/**
 * @brief Get expected measurement for a sensor using a specific trajectory
 */
bool FilterWrapper::getExpectedMeasurementByID(const std::string& sensor_id,
                                             kinematic_arbiter::core::MeasurementModelInterface::DynamicVector& measurement,
                                             const StateVector& state_at_sensor_time) const {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end()) {
    std::cerr << "Invalid sensor ID: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  return filter_->GetExpectedMeasurementByIndex(info.index, measurement, state_at_sensor_time);
}

/**
 * @brief Get expected measurement for a sensor using current reference state
 */
bool FilterWrapper::getExpectedMeasurementByID(const std::string& sensor_id,
                                             kinematic_arbiter::core::MeasurementModelInterface::DynamicVector& measurement) const {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end()) {
    std::cerr << "Invalid sensor ID: " << sensor_id << std::endl;
    return false;
  }

  const SensorInfo& info = it->second;
  return filter_->GetExpectedMeasurementByIndex(info.index, measurement);
}

} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter
