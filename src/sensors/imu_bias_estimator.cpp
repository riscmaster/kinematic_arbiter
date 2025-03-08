#include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"

namespace kinematic_arbiter {
namespace sensors {

void ImuBiasEstimator::EstimateBiases(
    const Eigen::Vector3d& measured_gyro,
    const Eigen::Vector3d& measured_accel,
    const Eigen::Vector3d& predicted_gyro,
    const Eigen::Vector3d& predicted_accel) {


  // Calculate instantaneous bias estimate (measured - predicted)
  Eigen::Vector3d current_gyro_bias = measured_gyro - predicted_gyro;
  Eigen::Vector3d current_accel_bias = measured_accel - predicted_accel;

  // Update biases using recursive averaging formula
  gyro_bias_ = gyro_bias_ + (current_gyro_bias - gyro_bias_) / window_size_;
  accel_bias_ = accel_bias_ + (current_accel_bias - accel_bias_) / window_size_;
}

} // namespace sensors
} // namespace kinematic_arbiter
