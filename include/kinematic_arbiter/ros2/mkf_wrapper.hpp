#ifndef KINEMATIC_ARBITER_ROS2_MKF_WRAPPER_HPP_
#define KINEMATIC_ARBITER_ROS2_MKF_WRAPPER_HPP_

#include <map>
#include <memory>
#include <string>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "rclcpp/time.hpp"

#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace wrapper {

/**
 * @brief Wrapper for the MediatedKalmanFilter that handles ROS2 message conversions
 */
class FilterWrapper {
public:
  using StateIndex = kinematic_arbiter::core::StateIndex;
  using StateVector = Eigen::Matrix<double, StateIndex::kFullStateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateIndex::kFullStateSize, StateIndex::kFullStateSize>;

  /**
   * @brief Constructor with RigidBodyStateModel parameters
   * @param model_params Parameters for the rigid body model
   */
  explicit FilterWrapper(const kinematic_arbiter::models::RigidBodyStateModel::Params& model_params = kinematic_arbiter::models::RigidBodyStateModel::Params());

  /**
   * @brief Register sensors of different types
   * @return The sensor ID for later reference
   */
  std::string registerPositionSensor(const std::string& name);
  std::string registerPoseSensor(const std::string& name);
  std::string registerBodyVelocitySensor(const std::string& name);
  std::string registerImuSensor(const std::string& name);

  /**
   * @brief Set maximum delay window for stale measurements
   */
  void setMaxDelayWindow(double seconds);

  /**
   * @brief Process different measurement types
   */
  bool processPosition(const std::string& sensor_id, const geometry_msgs::msg::PointStamped& msg);
//   bool processPose(const std::string& sensor_id, const geometry_msgs::msg::PoseStamped& msg);
//   bool processBodyVelocity(const std::string& sensor_id, const geometry_msgs::msg::TwistStamped& msg);
//   bool processImu(const std::string& sensor_id, const sensor_msgs::msg::Imu& msg);

  /**
   * @brief Get expected measuremen different measurement for each type
   */
  geometry_msgs::msg::PoseWithCovarianceStamped getExpectedPosition(const std::string& sensor_id);
//   geometry_msgs::msg::PoseWithCovarianceStamped getExpectedPose(const std::string& sensor_id);
//   geometry_msgs::msg::TwistWithCovarianceStamped getExpectedBodyVelocity(const std::string& sensor_id);
//   sensor_msgs::msg::Imu getExpectedImu(const std::string& sensor_id);

  /**
   * @brief Get state estimates as ROS2 messages
   */
  geometry_msgs::msg::PoseWithCovarianceStamped getPoseEstimate(const rclcpp::Time& time);
  geometry_msgs::msg::TwistWithCovarianceStamped getVelocityEstimate(const rclcpp::Time& time);
  geometry_msgs::msg::AccelWithCovarianceStamped getAccelerationEstimate(const rclcpp::Time& time);

  /**
   * @brief Dead reckon to current time (convenient method for publishing)
   */
  void predictTo(const rclcpp::Time& time);

  /**
   * @brief Convert ROS Time to seconds
   */
  double rosTimeToSeconds(const rclcpp::Time& time);

  /**
   * @brief Convert between ROS and Eigen types
   */
  Eigen::Vector3d pointMsgToVector(const geometry_msgs::msg::Point& point);
  Eigen::Quaterniond quaternionMsgToEigen(const geometry_msgs::msg::Quaternion& quat);
  Eigen::Vector3d vectorMsgToEigen(const geometry_msgs::msg::Vector3& vec);

  geometry_msgs::msg::Point vectorToPointMsg(const Eigen::Vector3d& vec);
  geometry_msgs::msg::Quaternion quaternionToQuaternionMsg(const Eigen::Quaterniond& quat);
  geometry_msgs::msg::Vector3 eigenToVectorMsg(const Eigen::Vector3d& vec);

private:
  // The core filter
  std::shared_ptr<kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>> filter_;

  // Sensor registries
  struct SensorInfo {
    size_t index;
    std::string type;
  };

  std::map<std::string, SensorInfo> sensors_;

  // Store message headers for reuse in expected measurements
  std::map<std::string, std_msgs::msg::Header> last_headers_;
};

} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter

#endif // KINEMATIC_ARBITER_ROS2_MKF_WRAPPER_HPP_
