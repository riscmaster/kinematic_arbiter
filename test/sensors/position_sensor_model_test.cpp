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
    bool pos_success = TestPositionPerturbations(y0, C, description);

    // Test quaternion perturbations (only relevant if sensor has offset)
    bool quat_success = TestQuaternionPerturbations(y0, C, description);

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
  bool TestPositionPerturbations(const MeasurementVector& y0,
                               const MeasurementJacobian& C,
                               const std::string& description) {
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
      MeasurementVector actual_delta_y = y1 - y0;

      // Calculate the predicted change using the Jacobian
      // Extract only the relevant columns from the Jacobian for the position components
      Eigen::Vector3d state_delta = Eigen::Vector3d::Zero();
      state_delta(i) = delta;

      // Use only the position part of the Jacobian
      Eigen::Matrix3d position_jacobian = C.block<3, 3>(0, StateIndex::Position::X);
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
  bool TestQuaternionPerturbations(const MeasurementVector& y0,
                                 const MeasurementJacobian& C,
                                 const std::string& description) {
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
      MeasurementVector actual_delta_y = y1 - y0;

      // Calculate the quaternion delta for Jacobian multiplication
      Eigen::Vector4d quat_delta;
      quat_delta << q_perturbed.w() - q_current.w(),
                  q_perturbed.x() - q_current.x(),
                  q_perturbed.y() - q_current.y(),
                  q_perturbed.z() - q_current.z();

      // Use only the quaternion part of the Jacobian
      Eigen::Matrix<double, 3, 4> quat_jacobian = C.block<3, 4>(0, StateIndex::Quaternion::W);
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

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
