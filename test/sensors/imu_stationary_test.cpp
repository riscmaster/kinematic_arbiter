#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <memory>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using Eigen::Vector3d;
using core::StateIndex;

// Forward declaration needed for friend class
class ImuStationaryTest;

class ImuStationaryTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create sensor model with identity transform and calibration enabled
    ImuSensorConfig config;
    config.calibration_enabled = true;
    config.stationary_confidence_threshold = 0.01;
    imu_model_ = std::make_unique<ImuSensorModel>(Eigen::Isometry3d::Identity(), config);

    // Initialize state vector and covariance
    state_ = Eigen::VectorXd::Zero(StateIndex::kFullStateSize);
    state_covariance_ = Eigen::MatrixXd::Identity(StateIndex::kFullStateSize, StateIndex::kFullStateSize);

    // Set default measurement (stationary with gravity in -Z direction)
    measurement_ = Eigen::VectorXd::Zero(6);
    measurement_.segment<3>(3) = Vector3d(0, 0, -imu_model_->GetGravity());

    // Set reasonable covariance values
    SetVelocityCovariance(0.01);
    SetAccelerationCovariance(0.01);
  }

  // Helper methods to modify state
  void SetLinearVelocity(const Vector3d& velocity) {
    state_.segment<3>(StateIndex::LinearVelocity::Begin()) = velocity;
  }

  void SetLinearAcceleration(const Vector3d& acceleration) {
    state_.segment<3>(StateIndex::LinearAcceleration::Begin()) = acceleration;
  }

  void SetVelocityCovariance(double variance) {
    state_covariance_.block<3, 3>(
        StateIndex::LinearVelocity::Begin(),
        StateIndex::LinearVelocity::Begin()) =
        Eigen::Matrix3d::Identity() * variance;
  }

  void SetAccelerationCovariance(double variance) {
    state_covariance_.block<3, 3>(
        StateIndex::LinearAcceleration::Begin(),
        StateIndex::LinearAcceleration::Begin()) =
        Eigen::Matrix3d::Identity() * variance;
  }

  // Helper to set IMU measurements
  void SetImuAcceleration(const Vector3d& accel) {
    measurement_.segment<3>(3) = accel;
  }

  void SetImuAngularVelocity(const Vector3d& angular_vel) {
    measurement_.segment<3>(0) = angular_vel;
  }

  // Direct access to IsStationary using friend relationship
  bool TestIsStationary(const ImuSensorModel* model) {
    return model->IsStationary(state_, state_covariance_, measurement_);
  }

  bool TestIsStationary() {
    return imu_model_->IsStationary(state_, state_covariance_, measurement_);
  }

  // Direct access to UpdateBiasEstimates
  bool TestUpdateBiasEstimates() {
    return imu_model_->UpdateBiasEstimates(state_, state_covariance_, measurement_);
  }

  std::unique_ptr<ImuSensorModel> imu_model_;
  Eigen::VectorXd state_;
  Eigen::MatrixXd state_covariance_;
  Eigen::VectorXd measurement_;
};

// Test stationary detection with zero velocity and acceleration
TEST_F(ImuStationaryTest, DetectStationaryWithZeroMotion) {
  // Default setup should be stationary
  EXPECT_TRUE(TestIsStationary());

  // Should update bias when stationary
  EXPECT_TRUE(TestUpdateBiasEstimates());
}

// Test non-stationary detection with non-zero velocity
TEST_F(ImuStationaryTest, DetectMovingWithNonZeroVelocity) {
  // Set small non-zero velocity
  SetLinearVelocity(Vector3d(0.1, 0, 0));
  EXPECT_FALSE(TestIsStationary());

  // Should not update bias when moving
  EXPECT_FALSE(TestUpdateBiasEstimates());

  // Set larger velocity
  SetLinearVelocity(Vector3d(1.0, 1.0, 1.0));
  EXPECT_FALSE(TestIsStationary());
}

// Test non-stationary detection with non-zero acceleration
TEST_F(ImuStationaryTest, DetectMovingWithNonZeroAcceleration) {
  // Set small non-zero acceleration
  SetLinearAcceleration(Vector3d(0.1, 0, 0));
  EXPECT_FALSE(TestIsStationary());

  // Set larger acceleration
  SetLinearAcceleration(Vector3d(1.0, 1.0, 1.0));
  EXPECT_FALSE(TestIsStationary());
}

// Test non-stationary detection with incorrect gravity magnitude
TEST_F(ImuStationaryTest, DetectMovingWithIncorrectGravity) {
  // Set acceleration magnitude too small
  SetImuAcceleration(Vector3d(0, 0, -8.0));
  EXPECT_FALSE(TestIsStationary());
}

// Test bias update functionality
TEST_F(ImuStationaryTest, BiasUpdateWhenStationary) {
  // Default setup should be stationary
  EXPECT_TRUE(TestIsStationary());

  // Should update bias when stationary and calibration enabled
  EXPECT_TRUE(TestUpdateBiasEstimates());

  // Create a model with calibration disabled
  ImuSensorConfig disabled_config;
  disabled_config.calibration_enabled = false;
  auto disabled_model = std::make_unique<ImuSensorModel>(Eigen::Isometry3d::Identity(), disabled_config);

  // Should not update bias when calibration disabled
  EXPECT_FALSE(disabled_model->UpdateBiasEstimates(state_, state_covariance_, measurement_));
}

// Test stationary detection with high uncertainty
TEST_F(ImuStationaryTest, StationaryWithHighUncertainty) {
  // Set small non-zero velocity but high uncertainty
  SetLinearVelocity(Vector3d(0.05, 0.05, 0.05));
  SetVelocityCovariance(1.0);  // High uncertainty

  // Should still detect as stationary due to high uncertainty
  EXPECT_TRUE(TestIsStationary());
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
