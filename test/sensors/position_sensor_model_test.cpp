#include "gtest/gtest.h"

#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Geometry>

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using core::StateIndex;

class PositionSensorModelTest : public ::testing::Test {
protected:
  // Type definitions for clarity
  using StateVector = PositionSensorModel::StateVector;
  using MeasurementVector = PositionSensorModel::MeasurementVector;
  using MeasurementJacobian = PositionSensorModel::MeasurementJacobian;

  void SetUp() override {
    // Default initialization
    default_state_ = StateVector::Zero(core::StateIndex::kFullStateSize);

    // Set position at [1, 2, 3]
    default_state_(core::StateIndex::Position::X) = 1.0;
    default_state_(core::StateIndex::Position::Y) = 2.0;
    default_state_(core::StateIndex::Position::Z) = 3.0;

    // Set orientation to identity quaternion [1, 0, 0, 0]
    default_state_(core::StateIndex::Quaternion::W) = 1.0;
    default_state_(core::StateIndex::Quaternion::X) = 0.0;
    default_state_(core::StateIndex::Quaternion::Y) = 0.0;
    default_state_(core::StateIndex::Quaternion::Z) = 0.0;

    // Create a non-identity sensor transform (2m offset in x, 45-degree rotation around z)
    Eigen::AngleAxisd rotation(M_PI / 4.0, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d translation(2.0, 0.0, 0.0);
    offset_transform_.translation() = translation;
    offset_transform_.linear() = rotation.toRotationMatrix();

    // Create default sensor model
    sensor_model_ = std::make_unique<PositionSensorModel>();
  }

  // Helper to set orientation in the state vector
  void SetQuaternion(const Quaterniond& q) {
    default_state_(core::StateIndex::Quaternion::W) = q.w();
    default_state_(core::StateIndex::Quaternion::X) = q.x();
    default_state_(core::StateIndex::Quaternion::Y) = q.y();
    default_state_(core::StateIndex::Quaternion::Z) = q.z();
  }

  // Helper to set position in the state vector
  void SetPosition(const Vector3d& position) {
    default_state_.segment<3>(core::StateIndex::Position::X) = position;
  }

  // Set sensor-to-body transform
  void SetSensorToBodyTransform(const Eigen::Isometry3d& T_BS) {
    // Recreate the model with the new transform
    sensor_model_ = std::make_unique<PositionSensorModel>(T_BS);
  }

  // Test if the Jacobian correctly predicts measurement changes
  void TestJacobianLinearization(const std::string& description) {
    // Get the baseline measurement and Jacobian
    MeasurementVector y0 = sensor_model_->PredictMeasurement(default_state_);
    MeasurementJacobian C = sensor_model_->GetMeasurementJacobian(default_state_);

    // Test position perturbations
    bool pos_success = TestPositionPerturbations(y0, C);

    // Test quaternion perturbations (only relevant if sensor has offset)
    bool quat_success = TestQuaternionPerturbations(y0, C);

    // Only print detailed diagnostics if any test failed
    if (!pos_success || !quat_success) {
      SCOPED_TRACE("==== Jacobian linearization failed for " + description + " ====");

      // Print the current state
      SCOPED_TRACE("Current state:");
      SCOPED_TRACE("  Position: " +
                  ::testing::PrintToString(default_state_.segment<3>(StateIndex::Position::X).transpose()));
      SCOPED_TRACE("  Quaternion: w=" + std::to_string(default_state_(StateIndex::Quaternion::W)) +
                  ", x=" + std::to_string(default_state_(StateIndex::Quaternion::X)) +
                  ", y=" + std::to_string(default_state_(StateIndex::Quaternion::Y)) +
                  ", z=" + std::to_string(default_state_(StateIndex::Quaternion::Z)));

      // Print the Jacobian
      std::ostringstream jac_stream;
      jac_stream << "Jacobian:\n" << C;
      SCOPED_TRACE(jac_stream.str());

      // Print the baseline measurement
      SCOPED_TRACE("Baseline measurement: " +
                  ::testing::PrintToString(y0.transpose()));
    }

    EXPECT_TRUE(pos_success && quat_success) << "Jacobian linearization test failed";
  }

  // Test position perturbations
  bool TestPositionPerturbations(const MeasurementVector& expected,
                                const MeasurementJacobian& jacobian,
                                const std::string& /* description */ = "") {
    bool all_passed = true;

    // Test perturbation in each component
    for (int i = 0; i < 3; i++) {
      // Create a small perturbation
      StateVector perturbed_state = default_state_;
      double delta = 0.1;  // 0.1 meters
      perturbed_state(StateIndex::Position::X + i) += delta;

      // Get the actual measurement at the perturbed state
      MeasurementVector y1 = sensor_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      MeasurementVector actual_delta_y = y1 - expected;

      // Calculate the predicted change using the Jacobian
      // Extract only the relevant columns from the Jacobian for the position components
      Eigen::Vector3d state_delta = Eigen::Vector3d::Zero();
      state_delta(i) = delta;

      // Use only the position part of the Jacobian
      Eigen::Matrix3d position_jacobian = jacobian.block<3, 3>(0, StateIndex::Position::X);
      Eigen::Vector3d predicted_delta_y = position_jacobian * state_delta;

      // Calculate the absolute error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Only print on failure
      bool passed = abs_error < 0.01;  // Use absolute error threshold
      if (!passed) {
        all_passed = false;
        SCOPED_TRACE("Position component " + std::to_string(i) + ": " +
                    std::to_string(abs_error) + " absolute error");
        SCOPED_TRACE("Actual change: " +
                    ::testing::PrintToString(actual_delta_y.transpose()));
        SCOPED_TRACE("Predicted change: " +
                    ::testing::PrintToString(predicted_delta_y.transpose()));
      }

      EXPECT_LT(abs_error, 0.01) << "Position Jacobian linearization error too large";
    }

    return all_passed;
  }

  // Test quaternion perturbations using proper rotation perturbations
  bool TestQuaternionPerturbations(const MeasurementVector& expected,
                                   const MeasurementJacobian& jacobian,
                                   const std::string& /* description */ = "") {
    bool all_passed = true;

    // Get current quaternion
    Quaterniond q_current(
        default_state_(StateIndex::Quaternion::W),
        default_state_(StateIndex::Quaternion::X),
        default_state_(StateIndex::Quaternion::Y),
        default_state_(StateIndex::Quaternion::Z));

    // Test small rotations around each axis
    Vector3d axes[3] = {
        Vector3d(1, 0, 0),  // X-axis
        Vector3d(0, 1, 0),  // Y-axis
        Vector3d(0, 0, 1)   // Z-axis
    };

    // Use a larger angle for better numerical stability
    const double angle = 0.01;  // 0.01 rad ≈ 0.57°

    for (int i = 0; i < 3; i++) {
      // Create a small rotation perturbation
      Quaterniond q_delta(Eigen::AngleAxisd(angle, axes[i]));

      // Apply the perturbation (right multiplication for local frame rotation)
      Quaterniond q_perturbed = q_current * q_delta;
      q_perturbed.normalize();  // Ensure unit quaternion

      // Create perturbed state
      StateVector perturbed_state = default_state_;
      perturbed_state(StateIndex::Quaternion::W) = q_perturbed.w();
      perturbed_state(StateIndex::Quaternion::X) = q_perturbed.x();
      perturbed_state(StateIndex::Quaternion::Y) = q_perturbed.y();
      perturbed_state(StateIndex::Quaternion::Z) = q_perturbed.z();

      // Get the actual measurement at the perturbed state
      MeasurementVector y1 = sensor_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      MeasurementVector actual_delta_y = y1 - expected;

      // Calculate the quaternion delta for Jacobian multiplication
      Eigen::Vector4d quat_delta;
      quat_delta << q_perturbed.w() - q_current.w(),
                  q_perturbed.x() - q_current.x(),
                  q_perturbed.y() - q_current.y(),
                  q_perturbed.z() - q_current.z();

      // Use only the quaternion part of the Jacobian
      Eigen::Matrix<double, 3, 4> quat_jacobian = jacobian.block<3, 4>(0, StateIndex::Quaternion::W);
      Eigen::Vector3d predicted_delta_y = quat_jacobian * quat_delta;

      // Calculate the absolute error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Check if the error is acceptable
      bool passed = abs_error < 0.01;  // Use absolute error threshold
      if (!passed) {
        all_passed = false;
        SCOPED_TRACE("Quaternion rotation around " +
                    std::string(i == 0 ? "X" : i == 1 ? "Y" : "Z") +
                    "-axis: " + std::to_string(abs_error) + " absolute error");
        SCOPED_TRACE("Actual change: " + ::testing::PrintToString(actual_delta_y.transpose()));
        SCOPED_TRACE("Predicted change: " + ::testing::PrintToString(predicted_delta_y.transpose()));
      }

      if (abs_error >= 0.01) {
        std::cout << "Quaternion component " << (i == 0 ? "X" : i == 1 ? "Y" : "Z")
                  << " perturbation details:\n";
        std::cout << "  Perturbed quaternion: w=" << q_perturbed.w()
                  << ", x=" << q_perturbed.x()
                  << ", y=" << q_perturbed.y()
                  << ", z=" << q_perturbed.z() << "\n";
        std::cout << "  Jacobian quaternion block:\n" << quat_jacobian << "\n";
      }

      EXPECT_LT(abs_error, 0.01) << "Quaternion Jacobian linearization error too large";
    }

    return all_passed;
  }

  // Default state vector with position and quaternion
  StateVector default_state_;

  // Non-identity transform with offset and rotation
  Eigen::Isometry3d offset_transform_ = Eigen::Isometry3d::Identity();

  // Sensor model
  std::unique_ptr<PositionSensorModel> sensor_model_;
};

// Test default construction and initialization
TEST_F(PositionSensorModelTest, DefaultInitialization) {
  // Model should be initialized with identity transform by default
  EXPECT_TRUE(true); // Placeholder to confirm test runs
}

// Test measurement prediction with identity transform
TEST_F(PositionSensorModelTest, PredictMeasurementIdentity) {
  auto measurement = sensor_model_->PredictMeasurement(default_state_);

  // With identity transform, sensor position = body position
  EXPECT_DOUBLE_EQ(measurement(PositionSensorModel::MeasurementIndex::X), 1.0);
  EXPECT_DOUBLE_EQ(measurement(PositionSensorModel::MeasurementIndex::Y), 2.0);
  EXPECT_DOUBLE_EQ(measurement(PositionSensorModel::MeasurementIndex::Z), 3.0);
}

// Test measurement prediction with offset transform
TEST_F(PositionSensorModelTest, PredictMeasurementWithTransform) {
  SetSensorToBodyTransform(offset_transform_);

  auto measurement = sensor_model_->PredictMeasurement(default_state_);

  // With 2m x-offset, sensor position should be [3, 2, 3]
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::X), 3.0, 1e-10);
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::Y), 2.0, 1e-10);
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::Z), 3.0, 1e-10);
}

// Test measurement prediction with rotated body
TEST_F(PositionSensorModelTest, PredictMeasurementWithBodyRotation) {
  // Create sensor with offset but no rotation
  Eigen::Isometry3d sensor_transform = Eigen::Isometry3d::Identity();
  sensor_transform.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  SetSensorToBodyTransform(sensor_transform);

  // Set body orientation to 90° rotation around Z
  Quaterniond rotation = Quaterniond(
      Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  SetQuaternion(rotation);

  auto measurement = sensor_model_->PredictMeasurement(default_state_);

  // With 90° Z rotation and 1m x-offset, sensor position should be [1, 3, 3]
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::X), 1.0, 1e-10);
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::Y), 3.0, 1e-10);
  EXPECT_NEAR(measurement(PositionSensorModel::MeasurementIndex::Z), 3.0, 1e-10);
}

// Test the structure of the measurement Jacobian
TEST_F(PositionSensorModelTest, MeasurementJacobianStructure) {
  auto jacobian = sensor_model_->GetMeasurementJacobian(default_state_);

  // Position-position block should be identity
  Eigen::Matrix3d position_block = jacobian.block<3, 3>(
      0, core::StateIndex::Position::X);
  Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

  EXPECT_TRUE(position_block.isApprox(identity));

  // Position-quaternion block should be zero for identity transform
  Eigen::Matrix<double, 3, 4> position_quat_block = jacobian.block<3, 4>(
      0, core::StateIndex::Quaternion::W);
  Eigen::Matrix<double, 3, 4> zeros = Eigen::Matrix<double, 3, 4>::Zero();

  EXPECT_TRUE(position_quat_block.isApprox(zeros));
}

// Test Jacobian with non-identity sensor transform
TEST_F(PositionSensorModelTest, MeasurementJacobianWithTransform) {
  SetSensorToBodyTransform(offset_transform_);

  auto jacobian = sensor_model_->GetMeasurementJacobian(default_state_);

  // Position-position block should still be identity
  Eigen::Matrix3d position_block = jacobian.block<3, 3>(
      0, core::StateIndex::Position::X);

  EXPECT_TRUE(position_block.isApprox(Eigen::Matrix3d::Identity()));

  // Position-quaternion block should be non-zero with offset transform
  Eigen::Matrix<double, 3, 4> position_quat_block = jacobian.block<3, 4>(
      0, core::StateIndex::Quaternion::W);

  // At least one element should be non-zero
  EXPECT_FALSE(position_quat_block.isApprox(Eigen::Matrix<double, 3, 4>::Zero()));
}

// Test Jacobian linearization with identity transform
TEST_F(PositionSensorModelTest, JacobianLinearizationIdentity) {
  SCOPED_TRACE("Testing with identity transform");
  TestJacobianLinearization("identity transform");
}

// Test Jacobian linearization with offset transform
TEST_F(PositionSensorModelTest, JacobianLinearizationWithTransform) {
  SetSensorToBodyTransform(offset_transform_);
  SCOPED_TRACE("Testing with offset transform");
  TestJacobianLinearization("offset transform");
}

// Test Jacobian linearization with rotated body
TEST_F(PositionSensorModelTest, JacobianLinearizationWithBodyRotation) {
  // Create sensor with offset but no rotation
  Eigen::Isometry3d sensor_transform = Eigen::Isometry3d::Identity();
  sensor_transform.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  SetSensorToBodyTransform(sensor_transform);

  // Set body orientation to 45° rotation around Y
  Quaterniond rotation = Quaterniond(
      Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitY()));
  SetQuaternion(rotation);

  SCOPED_TRACE("Testing with rotated body");
  TestJacobianLinearization("rotated body");
}

// Test initializable states
TEST_F(PositionSensorModelTest, InitializableStates) {
  PositionSensorModel model;

  // Get initializable states
  PositionSensorModel::StateFlags init_flags = model.GetInitializableStates();

  // Verify position states are initializable
  EXPECT_TRUE(init_flags[StateIndex::Position::X]);
  EXPECT_TRUE(init_flags[StateIndex::Position::Y]);
  EXPECT_TRUE(init_flags[StateIndex::Position::Z]);

  // Verify other states are not initializable
  EXPECT_FALSE(init_flags[StateIndex::Quaternion::W]);
  EXPECT_FALSE(init_flags[StateIndex::LinearVelocity::X]);
  EXPECT_FALSE(init_flags[StateIndex::AngularVelocity::X]);
}

// Test initialization with identity transform (no lever arm)
TEST_F(PositionSensorModelTest, InitializeWithIdentityTransform) {
  // Create model with identity transform
  PositionSensorModel model;

  // Create state, covariance and valid_states
  StateVector state = StateVector::Zero();
  PositionSensorModel::StateCovariance covariance = PositionSensorModel::StateCovariance::Zero();
  PositionSensorModel::StateFlags valid_states = PositionSensorModel::StateFlags::Zero();

  // Create position measurement [5, 6, 7]
  MeasurementVector measurement;
  measurement << 5.0, 6.0, 7.0;

  // Initialize state from measurement
  PositionSensorModel::StateFlags initialized_states = model.InitializeState(
      measurement, valid_states, state, covariance);

  // Verify position states were initialized
  EXPECT_TRUE(initialized_states[StateIndex::Position::X]);
  EXPECT_TRUE(initialized_states[StateIndex::Position::Y]);
  EXPECT_TRUE(initialized_states[StateIndex::Position::Z]);

  // Verify position values (should directly match measurement)
  EXPECT_NEAR(state(StateIndex::Position::X), 5.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Y), 6.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Z), 7.0, 1e-6);

  // Verify covariance was set (should be non-zero)
  EXPECT_GT(covariance(StateIndex::Position::X, StateIndex::Position::X), 0.0);
  EXPECT_GT(covariance(StateIndex::Position::Y, StateIndex::Position::Y), 0.0);
  EXPECT_GT(covariance(StateIndex::Position::Z, StateIndex::Position::Z), 0.0);
}

// Test initialization with lever arm and valid quaternion
TEST_F(PositionSensorModelTest, InitializeWithLeverArmAndValidQuaternion) {
  // Create transform with 2m offset in X direction
  Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
  sensor_pose.translation() = Eigen::Vector3d(2.0, 0.0, 0.0);

  // Create model with offset transform
  PositionSensorModel model(sensor_pose);

  // Create state, covariance and valid_states
  StateVector state = StateVector::Zero();
  PositionSensorModel::StateCovariance covariance = PositionSensorModel::StateCovariance::Zero();
  PositionSensorModel::StateFlags valid_states = PositionSensorModel::StateFlags::Zero();

  // Set a 90-degree rotation around Z-axis in the state
  Quaterniond rotation(0.7071, 0.0, 0.0, 0.7071);  // 90° rotation around Z
  state(StateIndex::Quaternion::W) = rotation.w();
  state(StateIndex::Quaternion::X) = rotation.x();
  state(StateIndex::Quaternion::Y) = rotation.y();
  state(StateIndex::Quaternion::Z) = rotation.z();

  // Mark quaternion as valid in valid_states
  valid_states[StateIndex::Quaternion::W] = true;
  valid_states[StateIndex::Quaternion::X] = true;
  valid_states[StateIndex::Quaternion::Y] = true;
  valid_states[StateIndex::Quaternion::Z] = true;

  // Create position measurement [10, 5, 3]
  MeasurementVector measurement;
  measurement << 10.0, 5.0, 3.0;

  // Initialize state from measurement
  model.InitializeState(measurement, valid_states, state, covariance);

  // Expected result: With 90° rotation around Z, the 2m X-offset becomes a 2m Y-offset
  // So body position = [10, 5, 3] - R * [2, 0, 0] = [10, 5, 3] - [0, 2, 0] = [10, 3, 3]
  // Use a more relaxed tolerance to account for numerical precision
  EXPECT_NEAR(state(StateIndex::Position::X), 10.0, 1e-4);
  EXPECT_NEAR(state(StateIndex::Position::Y), 3.0, 1e-4);  // 5.0 - 2.0
  EXPECT_NEAR(state(StateIndex::Position::Z), 3.0, 1e-4);
}

// Test initialization with lever arm but invalid quaternion
TEST_F(PositionSensorModelTest, InitializeWithLeverArmButInvalidQuaternion) {
  // Create transform with 2m offset in X direction
  Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
  sensor_pose.translation() = Eigen::Vector3d(2.0, 0.0, 0.0);

  // Create model with offset transform
  PositionSensorModel model(sensor_pose);

  // Create state, covariance and valid_states
  StateVector state = StateVector::Zero();
  PositionSensorModel::StateCovariance covariance = PositionSensorModel::StateCovariance::Identity();
  PositionSensorModel::StateFlags valid_states = PositionSensorModel::StateFlags::Zero();

  // Set quaternion in state, but don't mark it as valid in valid_states
  Quaterniond rotation(0.7071, 0.0, 0.0, 0.7071);
  state(StateIndex::Quaternion::W) = rotation.w();
  state(StateIndex::Quaternion::X) = rotation.x();
  state(StateIndex::Quaternion::Y) = rotation.y();
  state(StateIndex::Quaternion::Z) = rotation.z();

  // Store initial covariance values for comparison
  double initial_variance = covariance(StateIndex::Position::X, StateIndex::Position::X);

  // Create position measurement [10, 5, 3]
  MeasurementVector measurement;
  measurement << 10.0, 5.0, 3.0;

  // Initialize state from measurement
  model.InitializeState(measurement, valid_states, state, covariance);

  // Without valid quaternion, position should be raw sensor position without lever arm compensation
  EXPECT_NEAR(state(StateIndex::Position::X), 10.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Y), 5.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Z), 3.0, 1e-6);

  // Covariance should be increased due to lever arm uncertainty
  // Additional variance is lever_arm_length² = 2.0² = 4.0
  double expected_variance = initial_variance + 4.0;
  EXPECT_NEAR(covariance(StateIndex::Position::X, StateIndex::Position::X), expected_variance, 1e-6);
  EXPECT_NEAR(covariance(StateIndex::Position::Y, StateIndex::Position::Y), expected_variance, 1e-6);
  EXPECT_NEAR(covariance(StateIndex::Position::Z, StateIndex::Position::Z), expected_variance, 1e-6);
}

// Test initialization with small lever arm (should be ignored)
TEST_F(PositionSensorModelTest, InitializeWithSmallLeverArm) {
  // Create transform with very small offset (should be ignored)
  Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
  sensor_pose.translation() = Eigen::Vector3d(1e-7, 1e-7, 1e-7);  // Below 1e-6 threshold

  // Create model with tiny offset transform
  PositionSensorModel model(sensor_pose);

  // Create state, covariance and valid_states
  StateVector state = StateVector::Zero();
  PositionSensorModel::StateCovariance covariance = PositionSensorModel::StateCovariance::Identity();
  PositionSensorModel::StateFlags valid_states = PositionSensorModel::StateFlags::Zero();

  // Create position measurement
  MeasurementVector measurement;
  measurement << 10.0, 5.0, 3.0;

  // Initialize state from measurement
  model.InitializeState(measurement, valid_states, state, covariance);

  // Position should match measurement exactly since lever arm is negligible
  EXPECT_NEAR(state(StateIndex::Position::X), 10.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Y), 5.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Z), 3.0, 1e-6);

  // No additional uncertainty should be added to covariance
  EXPECT_NEAR(covariance(StateIndex::Position::X, StateIndex::Position::X), 1.0, 1e-6);
  EXPECT_NEAR(covariance(StateIndex::Position::Y, StateIndex::Position::Y), 1.0, 1e-6);
  EXPECT_NEAR(covariance(StateIndex::Position::Z, StateIndex::Position::Z), 1.0, 1e-6);
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
