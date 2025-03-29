// include/kinematic_arbiter/ros2/kinematic_arbiter_node.hpp
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include "kinematic_arbiter/ros2/filter_wrapper.hpp"

namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief ROS2 node for managing kinematic state estimation
 */
class KinematicArbiterNode : public rclcpp::Node {
public:
  /**
   * @brief Constructor - initializes node, parameters and filter
   */
  KinematicArbiterNode();

  /**
   * @brief Destructor
   */
  virtual ~KinematicArbiterNode() = default;

private:
  // Parameters
  double publish_rate_;
  std::string world_frame_id_;
  std::string body_frame_id_;

  // TF components
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Filter wrapper
  std::unique_ptr<FilterWrapper> filter_wrapper_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // Callbacks
  void publishEstimates();

  // Sensor initialization
  void configureSensors();
  void configurePositionSensors();
  void configurePoseSensors();
  void configureVelocitySensors();
  void configureImuSensors();
};

} // namespace ros2
} // namespace kinematic_arbiter
