#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <Eigen/Geometry>

#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using core::StateIndex;
using Eigen::Vector3d;
using models::RigidBodyStateModel;

// Add this constant to avoid direct access to the private kGravity in the class
constexpr double kTestGravity = 9.80665;

// Helper class to access protected members for testing
class TestableImuSensorModel : public ImuSensorModel {
public:
  explicit TestableImuSensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      bool calibration_enabled = false)
    : ImuSensorModel(sensor_pose_in_body_frame) {

    // Use the public setter method instead of direct access to private member
    ImuSensorConfig config;
    config.calibration_enabled = calibration_enabled;
    SetConfig(config);
  }
};

class ImuPredictionInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a state model for prediction
    state_model_ = std::make_unique<RigidBodyStateModel>();

    // Create a sensor model with identity pose for simplicity
    sensor_model_ = std::make_unique<TestableImuSensorModel>(Eigen::Isometry3d::Identity(), false);

    // Create a default state vector
    state_before_ = Eigen::VectorXd::Zero(StateIndex::kFullStateSize);

    // Set quaternion to identity
    state_before_(StateIndex::Quaternion::W) = 1.0;

    // Default time step
    dt_ = 0.1;
  }

  // Helper to manually create an IMU measurement
  Eigen::Matrix<double, 6, 1> CreateManualIMUMeasurement(
      const Vector3d& angular_velocity,
      const Vector3d& linear_acceleration) {

    Eigen::Matrix<double, 6, 1> measurement;

    // For identity orientation and sensor mounting:
    // Gyro measures angular velocity directly
    measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX) = angular_velocity;

    // Accelerometer measures linear acceleration plus gravity
    Vector3d gravity(0, 0, kTestGravity);
    measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX) = linear_acceleration + gravity;

    return measurement;
  }

  // Helper to generate detailed diagnostic information
  std::string GetDiagnosticInfo(
      const Vector3d& linear_accel_input,
      const Vector3d& angular_accel_input,
      const Vector3d& linear_accel_recovered,
      const Vector3d& angular_accel_recovered,
      const Eigen::VectorXd& state_before,
      const Eigen::VectorXd& state_with_accel,
      const Eigen::VectorXd& state_after,
      const Eigen::Matrix<double, 6, 1>& measurement_after) {

    std::stringstream ss;

    // State information
    ss << "=== State Before ===" << std::endl;
    ss << "Position: "
       << state_before.segment<3>(StateIndex::Position::Begin()).transpose() << std::endl;
    ss << "Quaternion: ["
       << state_before(StateIndex::Quaternion::W) << ", "
       << state_before(StateIndex::Quaternion::X) << ", "
       << state_before(StateIndex::Quaternion::Y) << ", "
       << state_before(StateIndex::Quaternion::Z) << "]" << std::endl;
    ss << "Linear Velocity: "
       << state_before.segment<3>(StateIndex::LinearVelocity::Begin()).transpose() << std::endl;
    ss << "Angular Velocity: "
       << state_before.segment<3>(StateIndex::AngularVelocity::Begin()).transpose() << std::endl;
    ss << "Linear Acceleration: "
       << state_before.segment<3>(StateIndex::LinearAcceleration::Begin()).transpose() << std::endl;
    ss << "Angular Acceleration: "
       << state_before.segment<3>(StateIndex::AngularAcceleration::Begin()).transpose() << std::endl;

    ss << "=== State With Accelerations ===" << std::endl;
    ss << "Position: "
       << state_with_accel.segment<3>(StateIndex::Position::Begin()).transpose() << std::endl;
    ss << "Quaternion: ["
       << state_with_accel(StateIndex::Quaternion::W) << ", "
       << state_with_accel(StateIndex::Quaternion::X) << ", "
       << state_with_accel(StateIndex::Quaternion::Y) << ", "
       << state_with_accel(StateIndex::Quaternion::Z) << "]" << std::endl;
    ss << "Linear Velocity: "
       << state_with_accel.segment<3>(StateIndex::LinearVelocity::Begin()).transpose() << std::endl;
    ss << "Angular Velocity: "
       << state_with_accel.segment<3>(StateIndex::AngularVelocity::Begin()).transpose() << std::endl;
    ss << "Linear Acceleration: "
       << state_with_accel.segment<3>(StateIndex::LinearAcceleration::Begin()).transpose() << std::endl;
    ss << "Angular Acceleration: "
       << state_with_accel.segment<3>(StateIndex::AngularAcceleration::Begin()).transpose() << std::endl;

    ss << "=== State After ===" << std::endl;
    ss << "Position: "
       << state_after.segment<3>(StateIndex::Position::Begin()).transpose() << std::endl;
    ss << "Quaternion: ["
       << state_after(StateIndex::Quaternion::W) << ", "
       << state_after(StateIndex::Quaternion::X) << ", "
       << state_after(StateIndex::Quaternion::Y) << ", "
       << state_after(StateIndex::Quaternion::Z) << "]" << std::endl;
    ss << "Linear Velocity: "
       << state_after.segment<3>(StateIndex::LinearVelocity::Begin()).transpose() << std::endl;
    ss << "Angular Velocity: "
       << state_after.segment<3>(StateIndex::AngularVelocity::Begin()).transpose() << std::endl;
    ss << "Linear Acceleration: "
       << state_after.segment<3>(StateIndex::LinearAcceleration::Begin()).transpose() << std::endl;
    ss << "Angular Acceleration: "
       << state_after.segment<3>(StateIndex::AngularAcceleration::Begin()).transpose() << std::endl;

    // Measurement information
    ss << "Manual IMU Measurement:" << std::endl;
    ss << "Gyro: " << measurement_after.segment<3>(ImuSensorModel::MeasurementIndex::GX).transpose() << std::endl;
    ss << "Accel: " << measurement_after.segment<3>(ImuSensorModel::MeasurementIndex::AX).transpose() << std::endl;

    // Acceleration comparison
    ss << "Input linear accel: " << linear_accel_input.transpose() << std::endl;
    ss << "Recovered linear accel: " << linear_accel_recovered.transpose() << std::endl;
    ss << "Difference: " << (linear_accel_recovered - linear_accel_input).transpose() << std::endl;

    Vector3d rel_error;
    for (int i = 0; i < 3; ++i) {
      rel_error[i] = std::abs(linear_accel_recovered[i] - linear_accel_input[i]) /
                     std::abs(linear_accel_input[i]) * 100.0;
    }
    ss << "Relative error: " << rel_error.transpose() << "%" << std::endl;

    ss << "Input angular accel: " << angular_accel_input.transpose() << std::endl;
    ss << "Recovered angular accel: " << angular_accel_recovered.transpose() << std::endl;
    ss << "Difference: " << (angular_accel_recovered - angular_accel_input).transpose() << std::endl;

    return ss.str();
  }

  std::unique_ptr<RigidBodyStateModel> state_model_;
  std::unique_ptr<TestableImuSensorModel> sensor_model_;
  Eigen::VectorXd state_before_;
  double dt_;
};

// Test with direct measurement creation
TEST_F(ImuPredictionInputsTest, DirectMeasurementTest) {
  // Define input accelerations
  Vector3d linear_accel_input(1.0, 2.0, 3.0);
  Vector3d angular_accel_input(0.5, -0.3, 0.1);

  // Set accelerations in the state
  Eigen::VectorXd state_with_accel = state_before_;
  state_with_accel.segment<3>(StateIndex::LinearAcceleration::Begin()) = linear_accel_input;
  state_with_accel.segment<3>(StateIndex::AngularAcceleration::Begin()) = angular_accel_input;

  // Predict state forward using the state model
  Eigen::VectorXd state_after = state_model_->PredictState(state_with_accel, dt_);

  // Extract angular velocity from the state after prediction
  Vector3d angular_velocity_after = state_after.segment<3>(StateIndex::AngularVelocity::Begin());

  // Manually create an IMU measurement
  Eigen::Matrix<double, 6, 1> measurement_after = CreateManualIMUMeasurement(
      angular_velocity_after, linear_accel_input);

  // Now use GetPredictionModelInputs to recover the accelerations
  auto recovered_inputs = sensor_model_->GetPredictionModelInputs(
      state_before_, measurement_after, dt_);

  // Extract the recovered accelerations
  Vector3d linear_accel_recovered = recovered_inputs.segment<3>(0);
  Vector3d angular_accel_recovered = recovered_inputs.segment<3>(3);

  // Use a reasonable tolerance for numerical precision
  // For linear acceleration, we'll use a relative tolerance of 2%
  const double kRelativeTolerance = 0.02;  // 2%

  // For angular acceleration, we'll use an absolute tolerance
  const double kAngularTolerance = 1e-10;

  // Generate diagnostic info only if needed for failure messages
  std::string diagnostic_info = GetDiagnosticInfo(
      linear_accel_input, angular_accel_input,
      linear_accel_recovered, angular_accel_recovered,
      state_before_, state_with_accel, state_after, measurement_after);

  // Check linear accelerations with relative tolerance
  for (int i = 0; i < 3; ++i) {
    double abs_error = std::abs(linear_accel_recovered[i] - linear_accel_input[i]);
    double rel_error = abs_error / std::abs(linear_accel_input[i]);
    EXPECT_LT(rel_error, kRelativeTolerance)
        << "Linear acceleration component " << i
        << " exceeds relative tolerance of " << kRelativeTolerance * 100 << "%" << std::endl
        << diagnostic_info;
  }

  // Check angular accelerations with absolute tolerance
  EXPECT_NEAR(angular_accel_recovered[0], angular_accel_input[0], kAngularTolerance)
      << "Angular acceleration X component differs by "
      << (angular_accel_recovered[0] - angular_accel_input[0]) << std::endl
      << diagnostic_info;
  EXPECT_NEAR(angular_accel_recovered[1], angular_accel_input[1], kAngularTolerance)
      << "Angular acceleration Y component differs by "
      << (angular_accel_recovered[1] - angular_accel_input[1]) << std::endl
      << diagnostic_info;
  EXPECT_NEAR(angular_accel_recovered[2], angular_accel_input[2], kAngularTolerance)
      << "Angular acceleration Z component differs by "
      << (angular_accel_recovered[2] - angular_accel_input[2]) << std::endl
      << diagnostic_info;
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
