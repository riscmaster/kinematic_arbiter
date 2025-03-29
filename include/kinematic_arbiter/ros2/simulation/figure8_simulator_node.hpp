#ifndef KINEMATIC_ARBITER_ROS2_SIMULATION_FIGURE8_SIMULATOR_NODE_HPP_
#define KINEMATIC_ARBITER_ROS2_SIMULATION_FIGURE8_SIMULATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <random>
#include <chrono>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/trajectory_utils.hpp"
#include "kinematic_arbiter/ros2/simulation/sensor_publisher.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"


namespace kinematic_arbiter {
namespace ros2 {
namespace simulation {

/**
 * @brief Simplified simulator node that generates a Figure-8 trajectory and publishes sensor data
 */
class Figure8SimulatorNode : public rclcpp::Node {
public:
  using SIdx = kinematic_arbiter::core::StateIndex;
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;
  using MeasurementModelInterface = kinematic_arbiter::core::MeasurementModelInterface;

  /**
   * @brief Constructor for Figure8SimulatorNode
   *
   * Initializes the node with all necessary publishers, parameters, and timers.
   */
  Figure8SimulatorNode();

private:
  /**
   * @brief Main update function called by the timer
   *
   * Generates trajectory state and publishes sensor data at appropriate rates
   */
  void update();

  /**
   * @brief Publish ground truth and TF transforms
   * @param state The current trajectory state
   */
  void publishGroundTruth(const StateVector& state);

  /**
   * @brief Publish position sensor data
   * @param state The current trajectory state
   * @param elapsed_seconds Current simulation time in seconds
   */
  void publishPosition(const StateVector& state, double elapsed_seconds);

  /**
   * @brief Publish pose sensor data
   * @param state The current trajectory state
   * @param elapsed_seconds Current simulation time in seconds
   */
  void publishPose(const StateVector& state, double elapsed_seconds);

  /**
   * @brief Publish velocity sensor data
   * @param state The current trajectory state
   * @param elapsed_seconds Current simulation time in seconds
   */
  void publishVelocity(const StateVector& state, double elapsed_seconds);

  /**
   * @brief Publish IMU sensor data
   * @param state The current trajectory state
   * @param elapsed_seconds Current simulation time in seconds
   */
  void publishImu(const StateVector& state, double elapsed_seconds);

  /**
   * @brief Set up sensor transforms
   *
   * Creates transforms for all sensors based on parameters
   */
  void publishSensorTransforms();

  // MKF Wrapper (for sensor registration and measurement prediction)
  std::unique_ptr<sensors::PositionSensorModel> position_sensor_model_;
  std::unique_ptr<sensors::PoseSensorModel> pose_sensor_model_;
  std::unique_ptr<sensors::BodyVelocitySensorModel> velocity_sensor_model_;
  std::unique_ptr<sensors::ImuSensorModel> imu_sensor_model_;

  // Trajectory parameters
  kinematic_arbiter::utils::Figure8Config trajectory_config_{};

  // Sensor publishers
  std::unique_ptr<PositionPublisher> position_publisher_;
  std::unique_ptr<PosePublisher> pose_publisher_;
  std::unique_ptr<VelocityPublisher> velocity_publisher_;
  std::unique_ptr<ImuPublisher> imu_publisher_;

  // Ground truth publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr truth_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr truth_velocity_pub_;

  // TF broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  // Timer
  rclcpp::TimerBase::SharedPtr update_timer_;

  // Frame IDs
  std::string world_frame_id_;
  std::string body_frame_id_;
  std::string position_sensor_id_;
  std::string pose_sensor_id_;
  std::string velocity_sensor_id_;
  std::string imu_sensor_id_;

  // Add these new frame ID variables
  std::string position_frame_id_;
  std::string pose_frame_id_;
  std::string velocity_frame_id_;
  std::string imu_frame_id_;

  // Timing control
  double main_update_rate_;
  double position_rate_;
  double pose_rate_;
  double velocity_rate_;
  double imu_rate_;

  int position_divider_ = 1;
  int pose_divider_ = 1;
  int velocity_divider_ = 1;
  int imu_divider_ = 1;
  int iteration_ = 0;

  // Noise parameters
  double noise_sigma_;
  double time_jitter_;

  // Random generators
  std::mt19937 generator_;
  std::normal_distribution<> noise_dist_;
  std::uniform_real_distribution<> jitter_dist_;

  // Start time tracking
  rclcpp::Time start_time_;

  // Sensor transforms (from sensor frame to body frame)
  geometry_msgs::msg::TransformStamped position_transform_;
  geometry_msgs::msg::TransformStamped pose_transform_;
  geometry_msgs::msg::TransformStamped velocity_transform_;
  geometry_msgs::msg::TransformStamped imu_transform_;
};

} // namespace simulation
} // namespace ros2
} // namespace kinematic_arbiter

#endif  // KINEMATIC_ARBITER_ROS2_SIMULATION_FIGURE8_SIMULATOR_NODE_HPP_
