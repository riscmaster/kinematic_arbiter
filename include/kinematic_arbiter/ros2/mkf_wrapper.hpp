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
#include "geometry_msgs/msg/transform_stamped.hpp"
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
#include "kinematic_arbiter/core/sensor_types.hpp"

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
   * @brief Register a position sensor with a given name
   * @param name The name of the sensor
   * @return The sensor ID for later reference
   */
  std::string registerPositionSensor(const std::string& name) {
    return registerSensor(name, kinematic_arbiter::core::SensorType::Position);
  }

  /**
   * @brief Register a pose sensor with a given name
   * @param name The name of the sensor
   * @return The sensor ID for later reference
   */
  std::string registerPoseSensor(const std::string& name) {
    return registerSensor(name, kinematic_arbiter::core::SensorType::Pose);
  }

  /**
   * @brief Register a body velocity sensor with a given name
   * @param name The name of the sensor
   * @return The sensor ID for later reference
   */
  std::string registerBodyVelocitySensor(const std::string& name) {
    return registerSensor(name, kinematic_arbiter::core::SensorType::BodyVelocity);
    }

  /**
   * @brief Register an IMU sensor with a given name
   * @param name The name of the sensor
   * @return The sensor ID for later reference
   */
  std::string registerImuSensor(const std::string& name) {
    return registerSensor(name, kinematic_arbiter::core::SensorType::Imu);
  }

   /* @brief Set maximum delay window for stale measurements
   */
  void setMaxDelayWindow(double seconds);

  /**
   * @brief Process Position measurement
   */
  bool processPosition(const std::string& sensor_id, const geometry_msgs::msg::PointStamped& msg);

  /**
   * @brief Process Pose measurement
   */
  bool processPose(const std::string& sensor_id, const geometry_msgs::msg::PoseStamped& msg);

  /**
   * @brief Process Body Velocity measurement
   */
  bool processBodyVelocity(const std::string& sensor_id, const geometry_msgs::msg::TwistStamped& msg);

  /**
   * @brief Process IMU measurement
   */
  bool processImu(const std::string& sensor_id, const sensor_msgs::msg::Imu& msg);

  /**
   * @brief Get expected measurement for position sensor
   * @param sensor_id The sensor identifier
   * @return Expected position measurement as PoseWithCovarianceStamped
   */
  geometry_msgs::msg::PoseWithCovarianceStamped getExpectedPosition(const std::string& sensor_id);

  /**
   * @brief Get expected measurement for a sensor using a specific trajectory
   * @param sensor_id The sensor identifier
   * @param[out] measurement The expected measurement vector
   * @param[in] state_at_sensor_time The state at the sensor time
   * @return True if successful, false otherwise
   */
  bool getExpectedMeasurementByID(const std::string& sensor_id,
                                 kinematic_arbiter::core::MeasurementModelInterface::DynamicVector& measurement,
                                 const StateVector& state_at_sensor_time) const;

  /**
   * @brief Get expected measurement for a sensor using current reference state
   * @param sensor_id The sensor identifier
   * @param[out] measurement The expected measurement vector
   * @return True if successful, false otherwise
   */
  bool getExpectedMeasurementByID(const std::string& sensor_id,
                                 kinematic_arbiter::core::MeasurementModelInterface::DynamicVector& measurement) const;


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
   * @brief Set the transform from sensor to body frame for a sensor
   *
   * @param sensor_id The sensor identifier
   * @param transform The transform from sensor to body frame (ROS2 format)
   * @return true if successful, false otherwise
   */
  bool setSensorTransform(const std::string& sensor_id,
                         const geometry_msgs::msg::TransformStamped& transform);

  /**
   * @brief Get the transform from sensor to body frame for a sensor
   *
   * @param sensor_id The sensor identifier
   * @param[out] transform The transform from sensor to body frame (ROS2 format)
   * @return true if successful, false otherwise
   */
  bool getSensorTransform(const std::string& sensor_id, const std::string& body_frame_id,
                         geometry_msgs::msg::TransformStamped& transform) const;

                         /**
   * @brief Convert ROS Time to seconds
   */
  double rosTimeToSeconds(const rclcpp::Time& time);

  // Helper method to convert double time to ROS time
  rclcpp::Time doubleTimeToRosTime(double time) const;

  /**
   * @brief Get the current time
   */
  double GetCurrentTime() const { return filter_->GetCurrentTime(); }


private:

  /**
   * @brief Register sensors of different types
   * @return The sensor ID for later reference
   */
  std::string registerSensor(const std::string& name, kinematic_arbiter::core::SensorType type);



  // Improved sensor info with the enum
  struct SensorInfo {
    size_t index;
    kinematic_arbiter::core::SensorType type;
  };

  std::map<std::string, SensorInfo> sensors_;

  // Filter instance
  std::shared_ptr<kinematic_arbiter::core::MediatedKalmanFilter<
      StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>> filter_;
};

} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter

#endif // KINEMATIC_ARBITER_ROS2_MKF_WRAPPER_HPP_
