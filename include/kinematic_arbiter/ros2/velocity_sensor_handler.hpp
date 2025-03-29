#pragma once

#include <geometry_msgs/msg/twist_stamped.hpp>
#include "kinematic_arbiter/ros2/sensor_handler.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"

namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Velocity sensor handler
 */
class VelocitySensorHandler : public SensorHandler<geometry_msgs::msg::TwistStamped> {
public:
  using Base = SensorHandler<geometry_msgs::msg::TwistStamped>;
  using ModelType = kinematic_arbiter::sensors::BodyVelocitySensorModel;
  using MeasurementIndex = ModelType::MeasurementIndex;

  VelocitySensorHandler(
      rclcpp::Node* node,
      std::shared_ptr<Filter> filter,
      std::shared_ptr<tf2_ros::Buffer> tf_buffer,
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id,
      const std::string& reference_frame_id,
      const std::string& body_frame_id)
    : Base(node, filter, tf_buffer, sensor_name, topic,
          sensor_frame_id, reference_frame_id, body_frame_id,
          std::make_shared<ModelType>(),
          SensorType::BodyVelocity) {}

protected:

  bool msgToVector(const geometry_msgs::msg::TwistStamped& msg, DynamicVector& vector) override {
    vector.resize(ModelType::kMeasurementDimension);

    // Linear velocity
    vector(MeasurementIndex::VX) = msg.twist.linear.x;
    vector(MeasurementIndex::VY) = msg.twist.linear.y;
    vector(MeasurementIndex::VZ) = msg.twist.linear.z;

    // Angular velocity
    vector(MeasurementIndex::WX) = msg.twist.angular.x;
    vector(MeasurementIndex::WY) = msg.twist.angular.y;
    vector(MeasurementIndex::WZ) = msg.twist.angular.z;

    return true;
  }

  geometry_msgs::msg::TwistStamped vectorToMsg(
      const DynamicVector& vector,
      const std_msgs::msg::Header& header) override {

    geometry_msgs::msg::TwistStamped msg;
    msg.header = header;

    // Linear velocity
    msg.twist.linear.x = vector(MeasurementIndex::VX);
    msg.twist.linear.y = vector(MeasurementIndex::VY);
    msg.twist.linear.z = vector(MeasurementIndex::VZ);

    // Angular velocity
    msg.twist.angular.x = vector(MeasurementIndex::WX);
    msg.twist.angular.y = vector(MeasurementIndex::WY);
    msg.twist.angular.z = vector(MeasurementIndex::WZ);

    return msg;
  }

  geometry_msgs::msg::TwistStamped applyBound(
      const geometry_msgs::msg::TwistStamped& base_msg,
      const DynamicCovariance& covariance,
      bool positive) override {

    auto bounded_msg = base_msg;
    double sign = positive ? 1.0 : -1.0;

    // Apply bounds to linear velocity components using measurement indices
    double vx_std_dev = std::sqrt(covariance(MeasurementIndex::VX, MeasurementIndex::VX));
    double vy_std_dev = std::sqrt(covariance(MeasurementIndex::VY, MeasurementIndex::VY));
    double vz_std_dev = std::sqrt(covariance(MeasurementIndex::VZ, MeasurementIndex::VZ));

    bounded_msg.twist.linear.x += sign * vx_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.twist.linear.y += sign * vy_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.twist.linear.z += sign * vz_std_dev * SIGMA_BOUND_FACTOR;

    // Apply bounds to angular velocity components using measurement indices
    double wx_std_dev = std::sqrt(covariance(MeasurementIndex::WX, MeasurementIndex::WX));
    double wy_std_dev = std::sqrt(covariance(MeasurementIndex::WY, MeasurementIndex::WY));
    double wz_std_dev = std::sqrt(covariance(MeasurementIndex::WZ, MeasurementIndex::WZ));

    bounded_msg.twist.angular.x += sign * wx_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.twist.angular.y += sign * wy_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.twist.angular.z += sign * wz_std_dev * SIGMA_BOUND_FACTOR;

    return bounded_msg;
  }
};

} // namespace ros2
} // namespace kinematic_arbiter
