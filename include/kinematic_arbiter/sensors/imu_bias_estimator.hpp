#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief IMU-specific bias estimator using recursive averaging
 *
 * Provides bias estimation for both gyroscope and accelerometer
 * measurements using recursive averaging. Should be used only when robot is stationary.
 */
class ImuBiasEstimator {
public:

  explicit ImuBiasEstimator(uint32_t window_size = 100)
      : window_size_(window_size),
        gyro_bias_(Eigen::Vector3d::Zero()),
        accel_bias_(Eigen::Vector3d::Zero()) {}

  /**
   * @brief Estimate biases if vehicle is stationary
   *
   * @param measured_gyro Raw gyroscope measurement
   * @param measured_accel Raw accelerometer measurement
   * @param predicted_gyro Expected gyroscope measurement for current state
   * @param predicted_accel Expected accelerometer measurement for current state
   */
  void EstimateBiases(
      const Eigen::Vector3d& measured_gyro,
      const Eigen::Vector3d& measured_accel,
      const Eigen::Vector3d& predicted_gyro,
      const Eigen::Vector3d& predicted_accel);

  /**
   * @brief Reset all bias calibration parameters to default values
   */
  void ResetCalibration() {
    gyro_bias_.setZero();
    accel_bias_.setZero();
  }

  /**
   * @brief Get current gyroscope bias estimate
   */
  const Eigen::Vector3d& GetGyroBias() const { return gyro_bias_; }

  /**
   * @brief Get current accelerometer bias estimate
   */
  const Eigen::Vector3d& GetAccelBias() const { return accel_bias_; }


private:
  uint32_t window_size_;
  Eigen::Vector3d gyro_bias_;
  Eigen::Vector3d accel_bias_;
};

} // namespace sensors
} // namespace kinematic_arbiter
