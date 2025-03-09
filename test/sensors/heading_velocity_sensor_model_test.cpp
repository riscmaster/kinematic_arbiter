#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/sensors/heading_velocity_sensor_model.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

class HeadingVelocitySensorModelTest : public ::testing::Test {
protected:
  // Define types for clarity
  using StateVector = HeadingVelocitySensorModel::StateVector;
  using MeasurementVector = HeadingVelocitySensorModel::MeasurementVector;
  using MeasurementJacobian = HeadingVelocitySensorModel::MeasurementJacobian;

  // Constants for state indexing
  static constexpr int STATE_QUAT_W = core::StateIndex::Quaternion::W;
  static constexpr int STATE_QUAT_X = core::StateIndex::Quaternion::X;
  static constexpr int STATE_QUAT_Y = core::StateIndex::Quaternion::Y;
  static constexpr int STATE_QUAT_Z = core::StateIndex::Quaternion::Z;
  static constexpr int STATE_LIN_VEL = core::StateIndex::LinearVelocity::Begin();

  void SetUp() override {
    // Initialize test state
    test_state_ = StateVector::Zero(core::StateIndex::kFullStateSize);

    // Set default quaternion to identity (no rotation)
    test_state_(STATE_QUAT_W) = 1.0;
    test_state_(STATE_QUAT_X) = 0.0;
    test_state_(STATE_QUAT_Y) = 0.0;
    test_state_(STATE_QUAT_Z) = 0.0;

    // Set default velocity
    velocity_ << 1.0, 0.5, -0.3;
    test_state_.segment<3>(STATE_LIN_VEL) = velocity_;

    // Create sensor model (always aligned with body frame)
    sensor_model_ = std::make_unique<HeadingVelocitySensorModel>();
  }

  // Helper to set orientation in the state vector
  void setOrientation(const Eigen::Quaterniond& orientation) {
    test_state_(STATE_QUAT_W) = orientation.w();
    test_state_(STATE_QUAT_X) = orientation.x();
    test_state_(STATE_QUAT_Y) = orientation.y();
    test_state_(STATE_QUAT_Z) = orientation.z();
  }

  // Helper to set velocity in the state vector
  void setVelocity(const Eigen::Vector3d& velocity) {
    test_state_.segment<3>(STATE_LIN_VEL) = velocity;
    velocity_ = velocity;
  }

  // Helper to compute heading vector from quaternion
  Eigen::Vector3d computeHeadingVector(const Eigen::Quaterniond& orientation) const {
    // Heading vector is the x-axis of the body frame rotated by the orientation
    return orientation * Eigen::Vector3d::UnitX();
  }

  // Helper to compute expected measurement
  double computeExpectedMeasurement(const Eigen::Vector3d& velocity,
                                   const Eigen::Quaterniond& orientation) const {
    Eigen::Vector3d heading = computeHeadingVector(orientation);
    return velocity.dot(heading);
  }

  // Test quaternion perturbations using proper rotation perturbations
  bool TestQuaternionPerturbations(const MeasurementVector& baseline_measurement,
                                  const MeasurementJacobian& jacobian,
                                  const std::string& description) {
    bool all_passed = true;

    // Get current quaternion
    Eigen::Quaterniond q_current(
        test_state_(STATE_QUAT_W),
        test_state_(STATE_QUAT_X),
        test_state_(STATE_QUAT_Y),
        test_state_(STATE_QUAT_Z));

    // Test small rotations around each axis
    Eigen::Vector3d axes[3] = {
        Eigen::Vector3d(1, 0, 0),  // X-axis
        Eigen::Vector3d(0, 1, 0),  // Y-axis
        Eigen::Vector3d(0, 0, 1)   // Z-axis
    };

    // Use a small angle for perturbation
    const double angle = 0.01;  // 0.01 rad ≈ 0.57°

    for (int i = 0; i < 3; i++) {
      // Create a small rotation perturbation
      Eigen::Quaterniond q_delta(Eigen::AngleAxisd(angle, axes[i]));

      // Apply the perturbation (right multiplication for local frame rotation)
      Eigen::Quaterniond q_perturbed = q_current * q_delta;
      q_perturbed.normalize();  // Ensure unit quaternion

      // Create perturbed state
      StateVector perturbed_state = test_state_;
      perturbed_state(STATE_QUAT_W) = q_perturbed.w();
      perturbed_state(STATE_QUAT_X) = q_perturbed.x();
      perturbed_state(STATE_QUAT_Y) = q_perturbed.y();
      perturbed_state(STATE_QUAT_Z) = q_perturbed.z();

      // Get the actual measurement at the perturbed state
      MeasurementVector perturbed_measurement = sensor_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      MeasurementVector actual_delta = perturbed_measurement - baseline_measurement;

      // Calculate the state delta
      StateVector state_delta = perturbed_state - test_state_;

      // Calculate the predicted change using the Jacobian
      MeasurementVector predicted_delta = jacobian * state_delta;

      // Calculate the absolute error
      double abs_error = std::abs(predicted_delta(0) - actual_delta(0));

      // Only print on failure
      bool passed = abs_error < 0.01;  // Use absolute error threshold
      if (!passed) {
        all_passed = false;
        SCOPED_TRACE("Quaternion rotation around " +
                    std::string(i == 0 ? "X" : i == 1 ? "Y" : "Z") +
                    "-axis: " + std::to_string(abs_error) + " absolute error");
        SCOPED_TRACE("Actual change: " + std::to_string(actual_delta(0)));
        SCOPED_TRACE("Predicted change: " + std::to_string(predicted_delta(0)));
      }

      EXPECT_LT(abs_error, 0.01) << "Quaternion Jacobian linearization error too large for "
                                << (i == 0 ? "X" : i == 1 ? "Y" : "Z") << "-axis rotation in "
                                << description;
    }

    return all_passed;
  }

  // Test velocity perturbations
  bool TestVelocityPerturbations(const MeasurementVector& baseline_measurement,
                               const MeasurementJacobian& jacobian,
                               const std::string& description) {
    bool all_passed = true;

    // Test perturbation in each component
    for (int i = 0; i < 3; i++) {
      // Create a small perturbation
      StateVector perturbed_state = test_state_;
      double delta = 0.1;  // 0.1 m/s
      perturbed_state(STATE_LIN_VEL + i) += delta;

      // Get the actual measurement at the perturbed state
      MeasurementVector perturbed_measurement = sensor_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      MeasurementVector actual_delta = perturbed_measurement - baseline_measurement;

      // Calculate the state delta
      StateVector state_delta = perturbed_state - test_state_;

      // Calculate the predicted change using the Jacobian
      MeasurementVector predicted_delta = jacobian * state_delta;

      // Calculate the absolute error
      double abs_error = std::abs(predicted_delta(0) - actual_delta(0));

      // Only print on failure
      bool passed = abs_error < 0.01;  // Use absolute error threshold
      if (!passed) {
        all_passed = false;
        SCOPED_TRACE("Velocity component " + std::to_string(i) +
                    " perturbation: " + std::to_string(abs_error) + " absolute error");
        SCOPED_TRACE("Actual change: " + std::to_string(actual_delta(0)));
        SCOPED_TRACE("Predicted change: " + std::to_string(predicted_delta(0)));
      }

      EXPECT_LT(abs_error, 0.01) << "Velocity Jacobian linearization error too large for "
                                << "component " << i << " in " << description;
    }

    return all_passed;
  }

  // Test data
  StateVector test_state_;
  Eigen::Vector3d velocity_;
  std::unique_ptr<HeadingVelocitySensorModel> sensor_model_;
};

// Test with identity orientation (heading along x-axis)
TEST_F(HeadingVelocitySensorModelTest, IdentityOrientation) {
  // With identity orientation, heading is [1,0,0]
  // So measurement should be just the x component of velocity
  auto measurement = sensor_model_->PredictMeasurement(test_state_);

  EXPECT_NEAR(measurement(0), velocity_.x(), 1e-6);
}

// Test with 90-degree rotation around Z (heading along y-axis)
TEST_F(HeadingVelocitySensorModelTest, RotationAroundZ) {
  // Create 90-degree rotation around Z
  Eigen::Quaterniond rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  setOrientation(rotation);

  // With 90-degree Z rotation, heading is [0,1,0]
  // So measurement should be just the y component of velocity
  auto measurement = sensor_model_->PredictMeasurement(test_state_);

  EXPECT_NEAR(measurement(0), velocity_.y(), 1e-6);
}

// Test with arbitrary orientation
TEST_F(HeadingVelocitySensorModelTest, ArbitraryOrientation) {
  // Create arbitrary rotation
  Eigen::Quaterniond rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d::UnitZ()));
  setOrientation(rotation);

  // Compute expected measurement
  double expected = computeExpectedMeasurement(velocity_, rotation);

  // Get actual measurement
  auto measurement = sensor_model_->PredictMeasurement(test_state_);

  EXPECT_NEAR(measurement(0), expected, 1e-6);
}

// Test with zero velocity
TEST_F(HeadingVelocitySensorModelTest, ZeroVelocity) {
  // Set velocity to zero
  setVelocity(Eigen::Vector3d::Zero());

  // Measurement should be zero regardless of orientation
  auto measurement = sensor_model_->PredictMeasurement(test_state_);

  EXPECT_NEAR(measurement(0), 0.0, 1e-6);
}

// Test Jacobian structure
TEST_F(HeadingVelocitySensorModelTest, JacobianStructure) {
  // Get Jacobian with identity orientation
  auto jacobian = sensor_model_->GetMeasurementJacobian(test_state_);

  // With identity orientation, heading is [1,0,0]
  // So Jacobian w.r.t. velocity should be [1,0,0]
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL), 1.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL + 1), 0.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL + 2), 0.0, 1e-6);

  // Jacobian w.r.t. quaternion should be specific values
  // For identity quaternion and velocity [1,0.5,-0.3]:
  // ∂z/∂qw = 2(vy*qz - vz*qy) = 0
  // ∂z/∂qx = 2(vy*qy + vz*qz) = 0
  // ∂z/∂qy = 2(-2*vx*qy + vy*qx - vz*qw) = -2*vz = 0.6
  // ∂z/∂qz = 2(-2*vx*qz + vy*qw + vz*qx) = 2*vy = 1.0
  EXPECT_NEAR(jacobian(0, STATE_QUAT_W), 0.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_QUAT_X), 0.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_QUAT_Y), 0.6, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_QUAT_Z), 1.0, 1e-6);
}

// Test Jacobian with different orientation
TEST_F(HeadingVelocitySensorModelTest, JacobianWithRotation) {
  // Create 90-degree rotation around Z
  Eigen::Quaterniond rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
  setOrientation(rotation);

  // Get Jacobian
  auto jacobian = sensor_model_->GetMeasurementJacobian(test_state_);

  // With 90-degree Z rotation, heading is [0,1,0]
  // So Jacobian w.r.t. velocity should be [0,1,0]
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL), 0.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL + 1), 1.0, 1e-6);
  EXPECT_NEAR(jacobian(0, STATE_LIN_VEL + 2), 0.0, 1e-6);
}

// Test numerical validation of the Jacobian
TEST_F(HeadingVelocitySensorModelTest, NumericalJacobianValidation) {
  // Create arbitrary rotation for a more general test
  Eigen::Quaterniond rotation = Eigen::Quaterniond(
      Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d::UnitZ()));
  setOrientation(rotation);

  // Get analytical Jacobian and baseline measurement
  const auto jacobian = sensor_model_->GetMeasurementJacobian(test_state_);
  const auto baseline_measurement = sensor_model_->PredictMeasurement(test_state_);

  // Test quaternion perturbations
  SCOPED_TRACE("Testing quaternion perturbations");
  bool quat_success = TestQuaternionPerturbations(baseline_measurement, jacobian, "quaternion test");

  // Test velocity perturbations
  SCOPED_TRACE("Testing velocity perturbations");
  bool vel_success = TestVelocityPerturbations(baseline_measurement, jacobian, "velocity test");

  // Only print detailed diagnostics if any test failed
  if (!quat_success || !vel_success) {
    SCOPED_TRACE("==== Jacobian linearization failed ====");

    // Print the current state
    SCOPED_TRACE("Current state:");
    SCOPED_TRACE("  Quaternion: w=" + std::to_string(test_state_(STATE_QUAT_W)) +
                ", x=" + std::to_string(test_state_(STATE_QUAT_X)) +
                ", y=" + std::to_string(test_state_(STATE_QUAT_Y)) +
                ", z=" + std::to_string(test_state_(STATE_QUAT_Z)));

    SCOPED_TRACE("  Velocity: " +
                ::testing::PrintToString(test_state_.segment<3>(STATE_LIN_VEL).transpose()));

    // Print the Jacobian
    std::ostringstream jac_stream;
    jac_stream << "Jacobian:\n" << jacobian;
    SCOPED_TRACE(jac_stream.str());

    // Print the baseline measurement
    SCOPED_TRACE("Baseline measurement: " + std::to_string(baseline_measurement(0)));
  }

  EXPECT_TRUE(quat_success && vel_success) << "Jacobian linearization test failed";
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
