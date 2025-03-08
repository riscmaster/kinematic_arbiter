#include <gtest/gtest.h>
#include <random>
#include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

class ImuBiasEstimatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use the new constructor signature with direct window_size parameter
    const uint32_t window_size = 20;
    estimator_ = std::make_unique<ImuBiasEstimator>(window_size);
  }

  std::unique_ptr<ImuBiasEstimator> estimator_;
};

// Test that the bias estimator correctly converges to a constant bias
TEST_F(ImuBiasEstimatorTest, ConstantBiasEstimation) {
  // Define constant biases to simulate
  Eigen::Vector3d true_gyro_bias(0.01, -0.02, 0.03);
  Eigen::Vector3d true_accel_bias(-0.05, 0.1, -0.15);

  // Zero expected values (biases will be the difference)
  Eigen::Vector3d predicted_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_accel = Eigen::Vector3d::Zero();

  // Run the estimator through multiple iterations
  const int iterations = 200;
  for (int i = 0; i < iterations; ++i) {
    // Measured values include the true bias
    Eigen::Vector3d measured_gyro = predicted_gyro + true_gyro_bias;
    Eigen::Vector3d measured_accel = predicted_accel + true_accel_bias;

    // Update bias estimate
    estimator_->EstimateBiases(
        measured_gyro, measured_accel,
        predicted_gyro, predicted_accel);
  }

  // After many iterations, the estimated bias should be close to the true bias
  Eigen::Vector3d estimated_gyro_bias = estimator_->GetGyroBias();
  Eigen::Vector3d estimated_accel_bias = estimator_->GetAccelBias();

  // Compare with a small tolerance
  const double tolerance = 1e-3;
  EXPECT_NEAR(true_gyro_bias.x(), estimated_gyro_bias.x(), tolerance);
  EXPECT_NEAR(true_gyro_bias.y(), estimated_gyro_bias.y(), tolerance);
  EXPECT_NEAR(true_gyro_bias.z(), estimated_gyro_bias.z(), tolerance);

  EXPECT_NEAR(true_accel_bias.x(), estimated_accel_bias.x(), tolerance);
  EXPECT_NEAR(true_accel_bias.y(), estimated_accel_bias.y(), tolerance);
  EXPECT_NEAR(true_accel_bias.z(), estimated_accel_bias.z(), tolerance);
}

// Test bias estimation with noisy measurements
TEST_F(ImuBiasEstimatorTest, NoisyBiasEstimation) {
  // Define true biases
  Eigen::Vector3d true_gyro_bias(0.03, -0.04, 0.05);
  Eigen::Vector3d true_accel_bias(-0.1, 0.2, -0.3);

  // Zero predictions
  Eigen::Vector3d predicted_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d predicted_accel = Eigen::Vector3d::Zero();

  // Random number generator for adding noise
  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::normal_distribution<double> gyro_noise(0.0, 0.005);
  std::normal_distribution<double> accel_noise(0.0, 0.02);

  // Run the estimator with noisy measurements
  const int iterations = 500; // More iterations to overcome noise
  for (int i = 0; i < iterations; ++i) {
    // Add noise to measurements
    Eigen::Vector3d noise_gyro(gyro_noise(rng), gyro_noise(rng), gyro_noise(rng));
    Eigen::Vector3d noise_accel(accel_noise(rng), accel_noise(rng), accel_noise(rng));

    // Measured values include true bias and noise
    Eigen::Vector3d measured_gyro = predicted_gyro + true_gyro_bias + noise_gyro;
    Eigen::Vector3d measured_accel = predicted_accel + true_accel_bias + noise_accel;

    // Update bias estimate
    estimator_->EstimateBiases(
        measured_gyro, measured_accel,
        predicted_gyro, predicted_accel);
  }

  // With noise, we need a larger tolerance
  Eigen::Vector3d estimated_gyro_bias = estimator_->GetGyroBias();
  Eigen::Vector3d estimated_accel_bias = estimator_->GetAccelBias();

  const double gyro_tolerance = 0.01;
  const double accel_tolerance = 0.03;

  EXPECT_NEAR(true_gyro_bias.x(), estimated_gyro_bias.x(), gyro_tolerance);
  EXPECT_NEAR(true_gyro_bias.y(), estimated_gyro_bias.y(), gyro_tolerance);
  EXPECT_NEAR(true_gyro_bias.z(), estimated_gyro_bias.z(), gyro_tolerance);

  EXPECT_NEAR(true_accel_bias.x(), estimated_accel_bias.x(), accel_tolerance);
  EXPECT_NEAR(true_accel_bias.y(), estimated_accel_bias.y(), accel_tolerance);
  EXPECT_NEAR(true_accel_bias.z(), estimated_accel_bias.z(), accel_tolerance);
}

// Test bias reset functionality
TEST_F(ImuBiasEstimatorTest, ResetCalibration) {
  // First establish a non-zero bias
  Eigen::Vector3d true_bias(0.05, -0.05, 0.05);

  for (int i = 0; i < 100; ++i) {
    estimator_->EstimateBiases(
        true_bias, true_bias,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  }

  // Verify bias was established
  EXPECT_GT(estimator_->GetGyroBias().norm(), 0.01);
  EXPECT_GT(estimator_->GetAccelBias().norm(), 0.01);

  // Reset calibration
  estimator_->ResetCalibration();

  // Verify bias is now zero
  EXPECT_NEAR(estimator_->GetGyroBias().norm(), 0.0, 1e-10);
  EXPECT_NEAR(estimator_->GetAccelBias().norm(), 0.0, 1e-10);
}

// Test that window size affects convergence rate
TEST_F(ImuBiasEstimatorTest, WindowSizeEffect) {
  // Create two estimators with different window sizes
  // Use the new constructor signature directly
  ImuBiasEstimator fast_estimator(10); // Small window = fast convergence
  ImuBiasEstimator slow_estimator(100); // Large window = slow convergence

  // Define bias to estimate
  Eigen::Vector3d true_bias(0.1, 0.1, 0.1);

  // Run both estimators for a few iterations
  const int iterations = 20;
  for (int i = 0; i < iterations; ++i) {
    Eigen::Vector3d measured = true_bias;
    Eigen::Vector3d predicted = Eigen::Vector3d::Zero();

    fast_estimator.EstimateBiases(measured, measured, predicted, predicted);
    slow_estimator.EstimateBiases(measured, measured, predicted, predicted);
  }

  // The fast estimator should be closer to the true bias
  double fast_gyro_error = (fast_estimator.GetGyroBias() - true_bias).norm();
  double slow_gyro_error = (slow_estimator.GetGyroBias() - true_bias).norm();

  EXPECT_LT(fast_gyro_error, slow_gyro_error);
}

} // namespace test
} // namespace sensors
} // namespace kinematic_arbiter

// Main function to run the tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
