#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include "kinematic_arbiter/ros2/sensor_handler.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"
namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Pose sensor handler
 */
class PoseSensorHandler : public SensorHandler<geometry_msgs::msg::PoseStamped> {
public:
  using Base = SensorHandler<geometry_msgs::msg::PoseStamped>;
  using ModelType = kinematic_arbiter::sensors::PoseSensorModel;
  using MeasurementIndex = ModelType::MeasurementIndex;

  PoseSensorHandler(
      rclcpp::Node* node,
      std::shared_ptr<Filter> filter,
      std::shared_ptr<tf2_ros::Buffer> tf_buffer,
      std::shared_ptr<utils::TimeManager> time_manager,
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id,
      const std::string& reference_frame_id,
      const std::string& body_frame_id)
    : Base(node, filter, tf_buffer, time_manager, sensor_name, topic,
          sensor_frame_id, reference_frame_id, body_frame_id,
          std::make_shared<ModelType>(),
          SensorType::Pose) {}

protected:

  bool msgToVector(const geometry_msgs::msg::PoseStamped& msg, DynamicVector& vector) override {
    vector.resize(ModelType::kMeasurementDimension);

    // Position
    vector(MeasurementIndex::X) = msg.pose.position.x;
    vector(MeasurementIndex::Y) = msg.pose.position.y;
    vector(MeasurementIndex::Z) = msg.pose.position.z;

    // Orientation (quaternion)
    vector(MeasurementIndex::QW) = msg.pose.orientation.w;
    vector(MeasurementIndex::QX) = msg.pose.orientation.x;
    vector(MeasurementIndex::QY) = msg.pose.orientation.y;
    vector(MeasurementIndex::QZ) = msg.pose.orientation.z;

    return true;
  }

  geometry_msgs::msg::PoseStamped vectorToMsg(
      const DynamicVector& vector,
      const std_msgs::msg::Header& header) override {

    geometry_msgs::msg::PoseStamped msg;
    msg.header = header;

    // Position
    msg.pose.position.x = vector(MeasurementIndex::X);
    msg.pose.position.y = vector(MeasurementIndex::Y);
    msg.pose.position.z = vector(MeasurementIndex::Z);

    // Orientation
    msg.pose.orientation.w = vector(MeasurementIndex::QW);
    msg.pose.orientation.x = vector(MeasurementIndex::QX);
    msg.pose.orientation.y = vector(MeasurementIndex::QY);
    msg.pose.orientation.z = vector(MeasurementIndex::QZ);

    return msg;
  }

  geometry_msgs::msg::PoseStamped applyBound(
      const geometry_msgs::msg::PoseStamped& base_msg,
      const DynamicCovariance& covariance,
      bool positive) override {

    auto bounded_msg = base_msg;
    double sign = positive ? 1.0 : -1.0;

    // Apply bounds to position components using measurement indices
    double x_std_dev = std::sqrt(covariance(MeasurementIndex::X, MeasurementIndex::X));
    double y_std_dev = std::sqrt(covariance(MeasurementIndex::Y, MeasurementIndex::Y));
    double z_std_dev = std::sqrt(covariance(MeasurementIndex::Z, MeasurementIndex::Z));

    bounded_msg.pose.position.x += sign * x_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.pose.position.y += sign * y_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.pose.position.z += sign * z_std_dev * SIGMA_BOUND_FACTOR;

    // For quaternion components, we simply copy the original
    // Applying uncertainties to quaternions is more complex and would require
    // using axis-angle and rotation matrix conversions to maintain valid quaternions
    bounded_msg.pose.orientation = base_msg.pose.orientation;

    return bounded_msg;
  }
};

} // namespace ros2
} // namespace kinematic_arbiter
