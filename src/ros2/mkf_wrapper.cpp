#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace wrapper {

// Constructor
FilterWrapper::FilterWrapper(const kinematic_arbiter::models::RigidBodyStateModel::Params& model_params) {
  // Create state model
  auto state_model = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>(model_params);

  // Create the filter
  filter_ = std::make_shared<kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>>(state_model);
}

std::string FilterWrapper::registerPositionSensor(const std::string& name) {
  auto sensor = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();
  size_t idx = filter_->AddSensor(sensor);
  sensors_[name] = {idx, "position"};
  return name;
}

std::string FilterWrapper::registerPoseSensor(const std::string& name) {
  auto sensor = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();
  size_t idx = filter_->AddSensor(sensor);
  sensors_[name] = {idx, "pose"};
  return name;
}

std::string FilterWrapper::registerBodyVelocitySensor(const std::string& name) {
  auto sensor = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();
  size_t idx = filter_->AddSensor(sensor);
  sensors_[name] = {idx, "body_velocity"};
  return name;
}

std::string FilterWrapper::registerImuSensor(const std::string& name) {
  auto sensor = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();
  size_t idx = filter_->AddSensor(sensor);
  sensors_[name] = {idx, "imu"};
  return name;
}

void FilterWrapper::setMaxDelayWindow(double seconds) {
  filter_->SetMaxDelayWindow(seconds);
}

bool FilterWrapper::processPosition(const std::string& sensor_id, const geometry_msgs::msg::PointStamped& msg) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != "position") {
    return false;
  }

  // Convert to measurement type
  double timestamp = rosTimeToSeconds(msg.header.stamp);
  Eigen::Vector3d position;
  tf2::fromMsg(msg.point, position);

  // Store the header for this sensor (for expected measurement generation)
  last_headers_[sensor_id] = msg.header;

  // Process with filter
  return filter_->ProcessMeasurementByIndex(it->second.index, timestamp, position);
}

geometry_msgs::msg::PoseWithCovarianceStamped FilterWrapper::getExpectedPosition(const std::string& sensor_id) {
  auto it = sensors_.find(sensor_id);
  if (it == sensors_.end() || it->second.type != "position") {
    geometry_msgs::msg::PoseWithCovarianceStamped empty;
    return empty;
  }

  // Get sensor model using proper template and index
  auto sensor = filter_->GetSensorByIndex<kinematic_arbiter::sensors::PositionSensorModel>(it->second.index);
  if (!sensor) {
    geometry_msgs::msg::PoseWithCovarianceStamped empty;
    return empty;
  }

  // Get expected measurement and measurement covariance
  Eigen::Vector3d expected_measurement = sensor->PredictMeasurement(filter_->GetStateEstimate());
  Eigen::Matrix<double, 3, 3> measurement_covariance = sensor->GetMeasurementCovariance();

  // Create ROS message
  geometry_msgs::msg::PoseWithCovarianceStamped msg;

  // Use the original header if available, or current filter time if not
  auto header_it = last_headers_.find(sensor_id);
  if (header_it != last_headers_.end()) {
    msg.header = header_it->second;
  } else {
    msg.header.stamp = rclcpp::Time(filter_->GetCurrentTime() * 1e9); // Convert to nanoseconds
    msg.header.frame_id = "base_link"; // Should be configurable
  }

  // Set position (only valid fields for a position sensor)
  msg.pose.pose.position = vectorToPointMsg(expected_measurement);

  // Identity quaternion as placeholder (not relevant for position sensor)
  msg.pose.pose.orientation.w = 1.0;
  msg.pose.pose.orientation.x = 0.0;
  msg.pose.pose.orientation.y = 0.0;
  msg.pose.pose.orientation.z = 0.0;

  // Set covariance (only position block is relevant)
  // ROS uses 6x6 covariance in order [x, y, z, roll, pitch, yaw]
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      msg.pose.covariance[i*6 + j] = measurement_covariance(i, j);
    }
  }

  return msg;
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
  return time.seconds();
}

Eigen::Vector3d FilterWrapper::pointMsgToVector(const geometry_msgs::msg::Point& point) {
  // Manual conversion for Point to Vector3d
  Eigen::Vector3d vec;
  vec(0) = point.x;
  vec(1) = point.y;
  vec(2) = point.z;
  return vec;
}

Eigen::Quaterniond FilterWrapper::quaternionMsgToEigen(const geometry_msgs::msg::Quaternion& quat) {
  // Manual conversion for Quaternion to Eigen::Quaterniond
  Eigen::Quaterniond q;
  q.x() = quat.x;
  q.y() = quat.y;
  q.z() = quat.z;
  q.w() = quat.w;
  return q;
}

Eigen::Vector3d FilterWrapper::vectorMsgToEigen(const geometry_msgs::msg::Vector3& vec) {
  // Manual conversion for Vector3 to Vector3d
  Eigen::Vector3d v;
  v(0) = vec.x;
  v(1) = vec.y;
  v(2) = vec.z;
  return v;
}

geometry_msgs::msg::Point FilterWrapper::vectorToPointMsg(const Eigen::Vector3d& vec) {
  // Manual conversion for Point
  geometry_msgs::msg::Point point;
  point.x = vec(0);
  point.y = vec(1);
  point.z = vec(2);
  return point;
}

geometry_msgs::msg::Quaternion FilterWrapper::quaternionToQuaternionMsg(const Eigen::Quaterniond& quat) {
  // Using proper tf2_eigen method for Isometry3d conversion
  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();

  // Convert to TransformStamped and extract Quaternion
  geometry_msgs::msg::TransformStamped transform_stamped = tf2::eigenToTransform(isometry);
  return transform_stamped.transform.rotation;
}

geometry_msgs::msg::Vector3 FilterWrapper::eigenToVectorMsg(const Eigen::Vector3d& vec) {
  // Manual conversion for Vector3
  geometry_msgs::msg::Vector3 vector;
  vector.x = vec(0);
  vector.y = vec(1);
  vector.z = vec(2);
  return vector;
}

} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter
