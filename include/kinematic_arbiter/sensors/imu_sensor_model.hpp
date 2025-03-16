#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {
class ImuStationaryTest;
}  // namespace test

// IMU-specific measurement dimensions
constexpr int kImuAngularVelocityDof = 3;  // 3 DOF for gyroscope
constexpr int kImuLinearAccelerationDof = 3;  // 3 DOF for accelerometer
constexpr int kImuMeasurementDof = kImuAngularVelocityDof + kImuLinearAccelerationDof;

// Move Config struct outside the class
struct ImuSensorConfig {
  uint32_t bias_estimation_window_size = 1000;
  bool calibration_enabled = false;
  double stationary_confidence_threshold = 0.01;
};

/**
 * @brief IMU measurement model (gyroscope + accelerometer)
 *
 * Models an IMU sensor that measures angular velocity and linear acceleration.
 * Measurement vector is [gx, gy, gz, ax, ay, az]' where
 * [gx, gy, gz] represents angular velocity and [ax, ay, az] represents linear acceleration.
 */
class ImuSensorModel : public core::MeasurementModelInterface<Eigen::Matrix<double, kImuMeasurementDof, 1>> {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface<Eigen::Matrix<double, kImuMeasurementDof, 1>>;
  using StateVector = typename Base::StateVector;
  using StateCovariance = typename Base::StateCovariance;
  using MeasurementVector = typename Base::MeasurementVector;
  using MeasurementJacobian = typename Base::MeasurementJacobian;
  using MeasurementCovariance = typename Base::MeasurementCovariance;
  using StateFlags = typename Base::StateFlags;

  /**
   * @brief Indices for accessing IMU measurement components
   */
  struct MeasurementIndex {
    // Gyroscope indices
    static constexpr int GX = 0;
    static constexpr int GY = 1;
    static constexpr int GZ = 2;

    // Accelerometer indices
    static constexpr int AX = 3;
    static constexpr int AY = 4;
    static constexpr int AZ = 5;
  };

  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param config IMU sensor configuration
   * @param params Validation parameters
   */
  explicit ImuSensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ImuSensorConfig& config = ImuSensorConfig(),
      const ValidationParams& params = ValidationParams())
    : Base(sensor_pose_in_body_frame, params),
      bias_estimator_(config.bias_estimation_window_size),
      config_(config) {
        this->can_predict_input_accelerations_ = true;
      }

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [gx, gy, gz, ax, ay, az]'
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override;

  /**
   * @brief Update bias estimates
   *
   * Extended version that can update bias estimates if provided with raw measurements
   * and the vehicle is determined to be stationary.
   *
   * @param state Current state estimate
   * @param state_covariance State covariance
   * @param raw_measurement Raw IMU measurement
   * @return If bias estimates were updated (will be false if vehicle is not stationary)
   */
  bool UpdateBiasEstimates(
      const StateVector& state,
      const Eigen::MatrixXd& state_covariance,
      const MeasurementVector& raw_measurement);

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate
   * @return Jacobian of measurement with respect to state
   */
  MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const override;

  /**
   * @brief Get the inputs to the prediction model
   *
   * @param state_before_prediction Current state estimate
   * @param measurement_after_prediction Actual measurement y_k after prediction of dt
   * @param dt Time step in seconds
   * @return Linear and angular acceleration as inputs to the prediction model
   */
  Eigen::Matrix<double, 6, 1> GetPredictionModelInputs(const StateVector& state_before_prediction, const StateCovariance& state_covariance_before_prediction, const MeasurementVector& measurement_after_prediction, double dt) const override;

  /**
   * @brief Enable or disable bias calibration
   */
  void EnableCalibration(bool enable) {
    config_.calibration_enabled = enable;
  }

  /**
   * @brief Set the configuration for the IMU sensor model
   *
   * @param config Configuration to set
   */
  void SetConfig(const ImuSensorConfig& config) {
    config_ = config;
  }

  /**
   * @brief Get the gravity constant
   *
   * @return Gravity constant
   */
  double GetGravity() const {
    return kGravity;
  }

  /**
   * @brief Get states that this sensor can directly initialize
   *
   * IMU provides measurements of angular velocity and linear acceleration.
   * When stationary, it can also help initialize roll and pitch components
   * of orientation from gravity direction.
   *
   * @return Flags indicating initializable states
   */
  StateFlags GetInitializableStates() const override;

  /**
   * @brief Initialize state from IMU measurement
   *
   * Initializes angular velocity and linear acceleration directly.
   * When stationary, initializes orientation (roll/pitch) from gravity
   * and sets angular acceleration to zero.
   *
   * @param measurement IMU measurement [gx, gy, gz, ax, ay, az]
   * @param valid_states Flags indicating which states are valid
   * @param state State vector to update
   * @param covariance State covariance to update
   * @return Flags indicating which states were initialized
   */
  StateFlags InitializeState(
      const MeasurementVector& measurement,
      const StateFlags&,
      StateVector& state,
      StateCovariance& covariance) const override;

  friend class kinematic_arbiter::sensors::test::ImuStationaryTest;

private:
  ImuBiasEstimator bias_estimator_;
  ImuSensorConfig config_;

  // Gravity constants moved here from global scope
  static constexpr double kGravity = 9.80665;
  static constexpr double kGravityVariance = 0.012 * 0.012;  // Variance in gravity estimate

  /**
   * @brief Determine if the vehicle is stationary based on state and measurements
   *
   * @param state Current state estimate
   * @param state_covariance Current state covariance
   * @param measurement IMU measurement vector
   * @return Boolean indicating if vehicle is stationary
   */
  bool IsStationary(
      const StateVector& state,
      const StateCovariance& state_covariance,
      const MeasurementVector& measurement) const;
};

} // namespace sensors
} // namespace kinematic_arbiter
