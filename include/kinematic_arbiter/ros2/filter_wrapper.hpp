#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"

#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/ros2/position_sensor_handler.hpp"
#include "kinematic_arbiter/ros2/pose_sensor_handler.hpp"
#include "kinematic_arbiter/ros2/velocity_sensor_handler.hpp"
#include "kinematic_arbiter/ros2/imu_sensor_handler.hpp"

namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Wrapper for the MediatedKalmanFilter that handles ROS2 integrations
 */
class FilterWrapper {
  using StateIndex = ::kinematic_arbiter::core::StateIndex;
  using StateModelInterface = ::kinematic_arbiter::core::StateModelInterface;
  using MediatedKalmanFilter = ::kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, StateModelInterface>;

public:
  /**
   * @brief Constructor
   *
   * @param node ROS2 node for creating publishers and subscribers
   * @param tf_buffer TF buffer for transform lookups
   * @param model_params Parameters for the rigid body model
   * @param body_frame_id Frame ID for the body (default: "base_link")
   * @param world_frame_id Frame ID for the world (default: "map")
   */
  FilterWrapper(
      rclcpp::Node* node,
      std::shared_ptr<tf2_ros::Buffer> tf_buffer,
      const ::kinematic_arbiter::models::RigidBodyStateModel::Params& model_params,
      const std::string& body_frame_id = "base_link",
      const std::string& world_frame_id = "map");

  /**
   * @brief Set maximum delay window for handling out-of-sequence measurements
   *
   * @param seconds Maximum time in seconds to keep trajectory history
   */
  void setMaxDelayWindow(double seconds);

  /**
   * @brief Add a position sensor
   *
   * @param sensor_name Human-readable sensor name
   * @param topic Topic to subscribe to
   * @param sensor_frame_id Frame ID of the sensor
   * @return bool True if sensor was successfully added
   */
  bool addPositionSensor(
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id);

  /**
   * @brief Add a pose sensor
   *
   * @param sensor_name Human-readable sensor name
   * @param topic Topic to subscribe to
   * @param sensor_frame_id Frame ID of the sensor
   * @return bool True if sensor was successfully added
   */
  bool addPoseSensor(
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id);

  /**
   * @brief Add a velocity sensor
   *
   * @param sensor_name Human-readable sensor name
   * @param topic Topic to subscribe to
   * @param sensor_frame_id Frame ID of the sensor
   * @return bool True if sensor was successfully added
   */
  bool addVelocitySensor(
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id);

  /**
   * @brief Add an IMU sensor
   *
   * @param sensor_name Human-readable sensor name
   * @param topic Topic to subscribe to
   * @param sensor_frame_id Frame ID of the sensor
   * @return bool True if sensor was successfully added
   */
  bool addImuSensor(
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id);

  /**
   * @brief Get the estimated pose with covariance
   *
   * @param time Time to estimate at (uses current filter time if omitted)
   * @return Pose with covariance stamped message
   */
  geometry_msgs::msg::PoseWithCovarianceStamped getPoseEstimate(
      const rclcpp::Time& time = rclcpp::Time(0));

  /**
   * @brief Get the estimated velocity with covariance
   *
   * @param time Time to estimate at (uses current filter time if omitted)
   * @return Twist with covariance stamped message
   */
  geometry_msgs::msg::TwistWithCovarianceStamped getVelocityEstimate(
      const rclcpp::Time& time = rclcpp::Time(0));

  /**
   * @brief Get the estimated acceleration with covariance
   *
   * @param time Time to estimate at (uses current filter time if omitted)
   * @return Accel with covariance stamped message
   */
  geometry_msgs::msg::AccelWithCovarianceStamped getAccelerationEstimate(
      const rclcpp::Time& time = rclcpp::Time(0));

  /**
   * @brief Dead reckon to specified time
   *
   * @param time Time to predict to
   */
  void predictTo(const rclcpp::Time& time);

  /**
   * @brief Check if the filter is initialized
   *
   * @return bool True if filter is initialized
   */
  bool isInitialized() const;

private:
  // ROS node and TF buffer
  rclcpp::Node* node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  // Frame IDs
  std::string body_frame_id_;
  std::string world_frame_id_;

  // The filter
  std::shared_ptr<MediatedKalmanFilter> filter_;

  // Sensor handlers
  std::vector<std::shared_ptr<PositionSensorHandler>> position_handlers_;
  std::vector<std::shared_ptr<PoseSensorHandler>> pose_handlers_;
  std::vector<std::shared_ptr<VelocitySensorHandler>> velocity_handlers_;
  std::vector<std::shared_ptr<ImuSensorHandler>> imu_handlers_;
};

} // namespace ros2
} // namespace kinematic_arbiter
