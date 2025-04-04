#pragma once

#include <geometry_msgs/msg/point_stamped.hpp>
#include "kinematic_arbiter/ros2/sensor_handler.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"
namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Position sensor handler
 */
class PositionSensorHandler : public SensorHandler<geometry_msgs::msg::PointStamped> {
public:
  using Base = SensorHandler<geometry_msgs::msg::PointStamped>;
  using ModelType = kinematic_arbiter::sensors::PositionSensorModel;
  using MeasurementIndex = ModelType::MeasurementIndex;

  PositionSensorHandler(
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
          SensorType::Position) {}

protected:

  bool msgToVector(const geometry_msgs::msg::PointStamped& msg, DynamicVector& vector) override {
    vector.resize(ModelType::kMeasurementDimension);
    vector(MeasurementIndex::X) = msg.point.x;
    vector(MeasurementIndex::Y) = msg.point.y;
    vector(MeasurementIndex::Z) = msg.point.z;
    return true;
  }

  geometry_msgs::msg::PointStamped vectorToMsg(
      const DynamicVector& vector,
      const std_msgs::msg::Header& header) override {

    geometry_msgs::msg::PointStamped msg;
    msg.header = header;

    msg.point.x = vector(MeasurementIndex::X);
    msg.point.y = vector(MeasurementIndex::Y);
    msg.point.z = vector(MeasurementIndex::Z);

    return msg;
  }

  geometry_msgs::msg::PointStamped applyBound(
      const geometry_msgs::msg::PointStamped& base_msg,
      const DynamicCovariance& covariance,
      bool positive) override {

    auto bounded_msg = base_msg;
    double sign = positive ? 1.0 : -1.0;

    // Apply bounds using measurement indices
    double x_std_dev = std::sqrt(covariance(MeasurementIndex::X, MeasurementIndex::X));
    double y_std_dev = std::sqrt(covariance(MeasurementIndex::Y, MeasurementIndex::Y));
    double z_std_dev = std::sqrt(covariance(MeasurementIndex::Z, MeasurementIndex::Z));

    bounded_msg.point.x += sign * x_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.point.y += sign * y_std_dev * SIGMA_BOUND_FACTOR;
    bounded_msg.point.z += sign * z_std_dev * SIGMA_BOUND_FACTOR;

    return bounded_msg;
  }
};

} // namespace ros2
} // namespace kinematic_arbiter
