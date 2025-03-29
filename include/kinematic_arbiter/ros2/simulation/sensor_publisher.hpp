#ifndef KINEMATIC_ARBITER_ROS2_SIMULATION_SENSOR_PUBLISHER_HPP_
#define KINEMATIC_ARBITER_ROS2_SIMULATION_SENSOR_PUBLISHER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <random>
#include <string>
#include <Eigen/Dense>

#include "kinematic_arbiter/core/sensor_types.hpp"
#include "kinematic_arbiter/core/statistical_utils.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace simulation {

/**
 * @brief Templated sensor publisher that handles different sensor types
 *
 * Publishes four variants of each sensor:
 * - Raw (noisy measurements)
 * - Truth (ground truth)
 * - Upper bound (truth + 3*sigma)
 * - Lower bound (truth - 3*sigma)
 */
template <typename MsgType>
class SensorPublisher {
public:
  using MeasurementModelInterface = kinematic_arbiter::core::MeasurementModelInterface;

  /**
   * @brief Constructor for SensorPublisher
   *
   * @param node The ROS2 node to create publishers from
   * @param sensor_type The type of sensor (from core::SensorType)
   * @param name The name/prefix for the sensor topics
   * @param noise_sigma Standard deviation of noise to apply
   */
  SensorPublisher(
      rclcpp::Node* node,
      kinematic_arbiter::core::SensorType sensor_type,
      const std::string& name,
      double noise_sigma)
    : node_(node),
      sensor_type_(sensor_type),
      name_(name),
      noise_sigma_(noise_sigma) {

    // Create all publishers with consistent naming
    raw_pub_ = node_->create_publisher<MsgType>(
        "sensors/" + name_, 10);

    truth_pub_ = node_->create_publisher<MsgType>(
        "sensors/" + name_ + "/truth", 10);

    upper_bound_pub_ = node_->create_publisher<MsgType>(
        "sensors/" + name_ + "/upper_bound", 10);

    lower_bound_pub_ = node_->create_publisher<MsgType>(
        "sensors/" + name_ + "/lower_bound", 10);

    // Create noise distribution
    noise_dist_ = std::normal_distribution<>(0.0, noise_sigma_);

    RCLCPP_INFO(node_->get_logger(), "Created %s sensor publisher for '%s'",
                kinematic_arbiter::core::SensorTypeToString(sensor_type_).c_str(), name_.c_str());
  }

  /**
   * @brief Publish all variants of the sensor message
   *
   * @param measurement The true measurement vector
   * @param timestamp ROS timestamp for the message
   * @param frame_id The frame ID for the message
   * @param generator Random number generator for noise
   */
  void publish(
      const MeasurementModelInterface::DynamicVector& measurement,
      const rclcpp::Time& timestamp,
      const std::string& frame_id,
      std::mt19937& generator) {

    // Verify the dimension of the incoming measurement
    int expected_dim = kinematic_arbiter::core::GetMeasurementDimension(sensor_type_);
    if (measurement.size() != expected_dim) {
      RCLCPP_ERROR(node_->get_logger(), "Measurement dimension mismatch for %s: expected %d, got %d",
                  name_.c_str(), expected_dim, (int)measurement.size());
      return;
    }

    // Create the true message
    auto true_msg = eigenToRos(measurement, frame_id, timestamp);

    // Create noise vector with proper dimension
    Eigen::MatrixXd noise_covariance = Eigen::MatrixXd::Identity(expected_dim, expected_dim) * (noise_sigma_ * noise_sigma_);
    Eigen::VectorXd noise = kinematic_arbiter::utils::generateMultivariateNoise(noise_covariance, generator);

    // Create the noisy message
    auto raw_msg = eigenToRos(measurement + noise, frame_id, timestamp);

    // Create bound messages
    Eigen::VectorXd upper_bound = measurement;
    Eigen::VectorXd lower_bound = measurement;

    // Add/subtract 3*sigma to each component
    for (int i = 0; i < expected_dim; ++i) {
      upper_bound(i) += 3.0 * noise_sigma_;
      lower_bound(i) -= 3.0 * noise_sigma_;
    }

    auto upper_msg = eigenToRos(upper_bound, frame_id, timestamp);
    auto lower_msg = eigenToRos(lower_bound, frame_id, timestamp);

    // Publish all variants
    truth_pub_->publish(true_msg);
    raw_pub_->publish(raw_msg);
    upper_bound_pub_->publish(upper_msg);
    lower_bound_pub_->publish(lower_msg);
  }

  /**
   * @brief Convert an Eigen vector to a ROS message
   *
   * @param measurement The Eigen vector to convert
   * @param frame_id The frame ID for the message
   * @param timestamp ROS timestamp for the message
   * @return The ROS message
   */
  virtual MsgType eigenToRos(
      const Eigen::VectorXd& measurement,
      const std::string& frame_id,
      const rclcpp::Time& timestamp) = 0;

protected:
  rclcpp::Node* node_;
  kinematic_arbiter::core::SensorType sensor_type_;
  std::string name_;
  double noise_sigma_;

  // Publishers for the different variants
  typename rclcpp::Publisher<MsgType>::SharedPtr raw_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr truth_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr upper_bound_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr lower_bound_pub_;

  // Random distribution for noise
  std::normal_distribution<> noise_dist_;
};

/**
 * @brief Position sensor publisher implementation
 */
class PositionPublisher : public SensorPublisher<geometry_msgs::msg::PointStamped> {
public:
  using Base = SensorPublisher<geometry_msgs::msg::PointStamped>;
  using MIdx = kinematic_arbiter::sensors::PositionSensorModel::MeasurementIndex;

  PositionPublisher(
      rclcpp::Node* node,
      const std::string& name,
      double noise_sigma)
    : Base(node, kinematic_arbiter::core::SensorType::Position, name, noise_sigma) {}

  geometry_msgs::msg::PointStamped eigenToRos(
      const Eigen::VectorXd& measurement,
      const std::string& frame_id,
      const rclcpp::Time& timestamp) override {

    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = timestamp;
    msg.header.frame_id = frame_id;

    msg.point.x = measurement[MIdx::X];
    msg.point.y = measurement[MIdx::Y];
    msg.point.z = measurement[MIdx::Z];

    return msg;
  }
};

/**
 * @brief Pose sensor publisher implementation
 */
class PosePublisher : public SensorPublisher<geometry_msgs::msg::PoseStamped> {
public:
  using Base = SensorPublisher<geometry_msgs::msg::PoseStamped>;
  using MIdx = kinematic_arbiter::sensors::PoseSensorModel::MeasurementIndex;

  PosePublisher(
      rclcpp::Node* node,
      const std::string& name,
      double noise_sigma)
    : Base(node, kinematic_arbiter::core::SensorType::Pose, name, noise_sigma) {}

  geometry_msgs::msg::PoseStamped eigenToRos(
      const Eigen::VectorXd& measurement,
      const std::string& frame_id,
      const rclcpp::Time& timestamp) override {

    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = timestamp;
    msg.header.frame_id = frame_id;

    // Position
    msg.pose.position.x = measurement[MIdx::X];
    msg.pose.position.y = measurement[MIdx::Y];
    msg.pose.position.z = measurement[MIdx::Z];

    // Quaternion (normalize to ensure it's valid)
    double qw = measurement[MIdx::QW];
    double qx = measurement[MIdx::QX];
    double qy = measurement[MIdx::QY];
    double qz = measurement[MIdx::QZ];

    // Normalize
    double norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (norm > 1e-10) {
      msg.pose.orientation.w = qw / norm;
      msg.pose.orientation.x = qx / norm;
      msg.pose.orientation.y = qy / norm;
      msg.pose.orientation.z = qz / norm;
    } else {
      // Default to identity if quaternion is too small
      msg.pose.orientation.w = 1.0;
      msg.pose.orientation.x = 0.0;
      msg.pose.orientation.y = 0.0;
      msg.pose.orientation.z = 0.0;
    }

    return msg;
  }
};

/**
 * @brief Velocity sensor publisher implementation
 */
class VelocityPublisher : public SensorPublisher<geometry_msgs::msg::TwistStamped> {
public:
  using Base = SensorPublisher<geometry_msgs::msg::TwistStamped>;
  using MIdx = kinematic_arbiter::sensors::BodyVelocitySensorModel::MeasurementIndex;

  VelocityPublisher(
      rclcpp::Node* node,
      const std::string& name,
      double noise_sigma)
    : Base(node, kinematic_arbiter::core::SensorType::BodyVelocity, name, noise_sigma) {}

  geometry_msgs::msg::TwistStamped eigenToRos(
      const Eigen::VectorXd& measurement,
      const std::string& frame_id,
      const rclcpp::Time& timestamp) override {

    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = timestamp;
    msg.header.frame_id = frame_id;

    // Linear velocity
    msg.twist.linear.x = measurement[MIdx::VX];
    msg.twist.linear.y = measurement[MIdx::VY];
    msg.twist.linear.z = measurement[MIdx::VZ];

    // Angular velocity
    msg.twist.angular.x = measurement[MIdx::WX];
    msg.twist.angular.y = measurement[MIdx::WY];
    msg.twist.angular.z = measurement[MIdx::WZ];

    return msg;
  }
};

/**
 * @brief IMU sensor publisher implementation
 */
class ImuPublisher : public SensorPublisher<sensor_msgs::msg::Imu> {
public:
  using Base = SensorPublisher<sensor_msgs::msg::Imu>;
  using MIdx = kinematic_arbiter::sensors::ImuSensorModel::MeasurementIndex;

  ImuPublisher(
      rclcpp::Node* node,
      const std::string& name,
      double noise_sigma)
    : Base(node, kinematic_arbiter::core::SensorType::Imu, name, noise_sigma) {}

  sensor_msgs::msg::Imu eigenToRos(
      const Eigen::VectorXd& measurement,
      const std::string& frame_id,
      const rclcpp::Time& timestamp) override {

    sensor_msgs::msg::Imu msg;
    msg.header.stamp = timestamp;
    msg.header.frame_id = frame_id;

    // Linear acceleration
    msg.linear_acceleration.x = measurement[MIdx::AX];
    msg.linear_acceleration.y = measurement[MIdx::AY];
    msg.linear_acceleration.z = measurement[MIdx::AZ];

    // Angular velocity
    msg.angular_velocity.x = measurement[MIdx::GX];
    msg.angular_velocity.y = measurement[MIdx::GY];
    msg.angular_velocity.z = measurement[MIdx::GZ];

    return msg;
  }
};

} // namespace simulation
} // namespace ros2
} // namespace kinematic_arbiter

#endif // KINEMATIC_ARBITER_ROS2_SIMULATION_SENSOR_PUBLISHER_HPP_
