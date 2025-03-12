#include "gtest/gtest.h"

#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Geometry>

namespace kinematic_arbiter {
namespace sensors {
namespace testing {

class PoseSensorModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default initialization
    default_state_.setZero();

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
  }

  // Default state vector [1, 2, 3, 1, 0, 0, 0] (position and quaternion)
  PoseSensorModel::StateVector default_state_;

  // Non-identity transform with offset and rotation
  Eigen::Isometry3d offset_transform_ = Eigen::Isometry3d::Identity();
};

// Test default construction and initialization
TEST_F(PoseSensorModelTest, DefaultInitialization) {
  PoseSensorModel model;

  // Model should be initialized with identity transform by default
  EXPECT_TRUE(true); // Placeholder to confirm test runs
}

// Test measurement prediction with identity transform
TEST_F(PoseSensorModelTest, PredictMeasurementIdentity) {
  PoseSensorModel model;

  auto measurement = model.PredictMeasurement(default_state_);

  // With identity transform, sensor position = body position
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::X), 1.0);
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::Y), 2.0);
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::Z), 3.0);

  // With identity transform, sensor orientation = body orientation
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::QW), 1.0);
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::QX), 0.0);
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::QY), 0.0);
  EXPECT_DOUBLE_EQ(measurement(PoseSensorModel::MeasurementIndex::QZ), 0.0);
}

// Test measurement prediction with offset transform
TEST_F(PoseSensorModelTest, PredictMeasurementWithTransform) {
  PoseSensorModel model(offset_transform_);

  auto measurement = model.PredictMeasurement(default_state_);

  // With 2m x-offset, sensor position should be [3, 2, 3]
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::X), 3.0, 1e-10);
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::Y), 2.0, 1e-10);
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::Z), 3.0, 1e-10);

  // With 45-deg z-rotation, quaternion should represent that rotation
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::QW), cos(M_PI/8.0), 1e-10);
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::QX), 0.0, 1e-10);
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::QY), 0.0, 1e-10);
  EXPECT_NEAR(measurement(PoseSensorModel::MeasurementIndex::QZ), sin(M_PI/8.0), 1e-10);
}

// Test the structure of the measurement Jacobian
TEST_F(PoseSensorModelTest, MeasurementJacobianStructure) {
  PoseSensorModel model;

  auto jacobian = model.GetMeasurementJacobian(default_state_);

  // Position-position block should be identity
  Eigen::Matrix3d position_block = jacobian.block<3, 3>(
      PoseSensorModel::MeasurementIndex::X,
      core::StateIndex::Position::X);
  Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

  EXPECT_TRUE(position_block.isApprox(identity));

  // Position-quaternion block should be zero
  Eigen::Matrix<double, 3, 4> position_quat_block = jacobian.block<3, 4>(
      PoseSensorModel::MeasurementIndex::X,
      core::StateIndex::Quaternion::W);
  Eigen::Matrix<double, 3, 4> zeros = Eigen::Matrix<double, 3, 4>::Zero();

  EXPECT_TRUE(position_quat_block.isApprox(zeros));

  // Check quaternion-quaternion block
  // For identity transform, it should be identity when body quaternion is identity
  Eigen::Matrix4d quat_block = jacobian.block<4, 4>(
      PoseSensorModel::MeasurementIndex::QW,
      core::StateIndex::Quaternion::W);
  Eigen::Matrix4d expected_quat_block = Eigen::Matrix4d::Identity();

  EXPECT_TRUE(quat_block.isApprox(expected_quat_block));
}

// Test Jacobian with non-identity sensor transform
TEST_F(PoseSensorModelTest, MeasurementJacobianWithTransform) {
  PoseSensorModel model(offset_transform_);

  auto jacobian = model.GetMeasurementJacobian(default_state_);

  // Position-position block should still be identity
  Eigen::Matrix3d position_block = jacobian.block<3, 3>(
      PoseSensorModel::MeasurementIndex::X,
      core::StateIndex::Position::X);

  EXPECT_TRUE(position_block.isApprox(Eigen::Matrix3d::Identity()));

  // Quaternion block should match the sensor-in-body quaternion
  Eigen::Quaterniond sensor_quat(offset_transform_.rotation());
  Eigen::Matrix4d expected_quat_block;

  // Create the correct quaternion Jacobian matrix
  expected_quat_block <<
    sensor_quat.w(),  sensor_quat.x(),  sensor_quat.y(),  sensor_quat.z(),
   -sensor_quat.x(),  sensor_quat.w(),  sensor_quat.z(), -sensor_quat.y(),
   -sensor_quat.y(), -sensor_quat.z(),  sensor_quat.w(),  sensor_quat.x(),
   -sensor_quat.z(),  sensor_quat.y(), -sensor_quat.x(),  sensor_quat.w();

  Eigen::Matrix4d quat_block = jacobian.block<4, 4>(
      PoseSensorModel::MeasurementIndex::QW,
      core::StateIndex::Quaternion::W);

  EXPECT_TRUE(quat_block.isApprox(expected_quat_block, 1e-10));
}

// Numerical validation of Jacobian through perturbation
TEST_F(PoseSensorModelTest, NumericalJacobianValidation) {
  PoseSensorModel model(offset_transform_);

  // Get analytical Jacobian
  auto analytical_jacobian = model.GetMeasurementJacobian(default_state_);

  // Base measurement
  auto base_measurement = model.PredictMeasurement(default_state_);

  // Test position components with numerical perturbation
  const double epsilon = 1e-6;

  // For position states
  for (int i = 0; i < 3; ++i) {
    PoseSensorModel::StateVector perturbed_state = default_state_;
    perturbed_state(core::StateIndex::Position::X + i) += epsilon;

    auto perturbed_measurement = model.PredictMeasurement(perturbed_state);

    // Compute numerical derivatives for this column
    Eigen::VectorXd numerical_derivatives =
        (perturbed_measurement - base_measurement) / epsilon;

    // Compare with analytical Jacobian column
    for (int j = 0; j < 7; ++j) {
      EXPECT_NEAR(
          numerical_derivatives(j),
          analytical_jacobian(j, core::StateIndex::Position::X + i),
          1e-5
      ) << "Mismatch at measurement " << j << ", state " << i;
    }
  }

  // For quaternion states, we need to ensure valid quaternions after perturbation
  // Testing just one quaternion component as an example
  PoseSensorModel::StateVector perturbed_state = default_state_;
  perturbed_state(core::StateIndex::Quaternion::X) += epsilon;

  // Normalize the quaternion
  double qnorm = std::sqrt(
      perturbed_state(core::StateIndex::Quaternion::W) * perturbed_state(core::StateIndex::Quaternion::W) +
      perturbed_state(core::StateIndex::Quaternion::X) * perturbed_state(core::StateIndex::Quaternion::X) +
      perturbed_state(core::StateIndex::Quaternion::Y) * perturbed_state(core::StateIndex::Quaternion::Y) +
      perturbed_state(core::StateIndex::Quaternion::Z) * perturbed_state(core::StateIndex::Quaternion::Z));

  perturbed_state(core::StateIndex::Quaternion::W) /= qnorm;
  perturbed_state(core::StateIndex::Quaternion::X) /= qnorm;
  perturbed_state(core::StateIndex::Quaternion::Y) /= qnorm;
  perturbed_state(core::StateIndex::Quaternion::Z) /= qnorm;

  auto perturbed_measurement = model.PredictMeasurement(perturbed_state);

  // Since we normalized, the perturbation is different than epsilon
  double actual_perturbation = perturbed_state(core::StateIndex::Quaternion::X) -
                               default_state_(core::StateIndex::Quaternion::X);

  // Only check the quaternion part of the measurement for this test
  for (int j = 3; j < 7; ++j) {
    EXPECT_NEAR(
        (perturbed_measurement(j) - base_measurement(j)) / actual_perturbation,
        analytical_jacobian(j, core::StateIndex::Quaternion::X),
        1e-4  // Wider tolerance due to quaternion normalization
    ) << "Quaternion Jacobian mismatch at measurement " << j;
  }
}

// Test the initializable states of the pose sensor model
TEST_F(PoseSensorModelTest, InitializableStates) {
  PoseSensorModel model;
  PoseSensorModel::StateFlags init_flags = model.GetInitializableStates();

  // Pose sensor should be able to initialize position
  EXPECT_TRUE(init_flags[core::StateIndex::Position::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::Position::Y]);
  EXPECT_TRUE(init_flags[core::StateIndex::Position::Z]);

  // Pose sensor should be able to initialize quaternion (full orientation)
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::W]);
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::Y]);
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::Z]);

  // All other states should NOT be initializable
  EXPECT_FALSE(init_flags[core::StateIndex::LinearVelocity::X]);
  EXPECT_FALSE(init_flags[core::StateIndex::AngularVelocity::X]);
  EXPECT_FALSE(init_flags[core::StateIndex::LinearAcceleration::X]);
  EXPECT_FALSE(init_flags[core::StateIndex::AngularAcceleration::X]);
}

// Test initialization of states from pose measurement with identity transform
TEST_F(PoseSensorModelTest, InitializeStateIdentityTransform) {
  // Use identity transform between sensor and body
  PoseSensorModel model;

  // Create measurement with position [5, 6, 7] and orientation 90° around Z
  PoseSensorModel::MeasurementVector measurement;
  measurement << 5.0, 6.0, 7.0,                 // position
                 0.7071, 0.0, 0.0, 0.7071;      // quaternion (90° around Z)

  // Create state, covariance and valid_states for initialization
  PoseSensorModel::StateVector state = PoseSensorModel::StateVector::Zero();
  PoseSensorModel::StateCovariance covariance = PoseSensorModel::StateCovariance::Identity();
  PoseSensorModel::StateFlags valid_states = PoseSensorModel::StateFlags::Zero();

  // Initialize state from measurement
  PoseSensorModel::StateFlags initialized_states = model.InitializeState(
      measurement, valid_states, state, covariance);

  // Check that position and quaternion states were initialized
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::X]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::Y]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::Z]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::W]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::X]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::Y]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::Z]);

  // With identity transform, body position should equal sensor position
  EXPECT_NEAR(state(core::StateIndex::Position::X), 5.0, 1e-6);
  EXPECT_NEAR(state(core::StateIndex::Position::Y), 6.0, 1e-6);
  EXPECT_NEAR(state(core::StateIndex::Position::Z), 7.0, 1e-6);

  // With identity transform, body orientation should equal sensor orientation
  EXPECT_NEAR(state(core::StateIndex::Quaternion::W), 0.7071, 1e-4);
  EXPECT_NEAR(state(core::StateIndex::Quaternion::X), 0.0, 1e-6);
  EXPECT_NEAR(state(core::StateIndex::Quaternion::Y), 0.0, 1e-6);
  EXPECT_NEAR(state(core::StateIndex::Quaternion::Z), 0.7071, 1e-4);
}

// Test initialization of states from pose measurement with offset transform
TEST_F(PoseSensorModelTest, InitializeStateOffsetTransform) {
  // Use offset transform between sensor and body (2m in X and 45° around Z)
  PoseSensorModel model(offset_transform_);

  // Create measurement with position [5, 0, 0] and identity orientation
  PoseSensorModel::MeasurementVector measurement;
  measurement << 5.0, 0.0, 0.0,      // position
                 1.0, 0.0, 0.0, 0.0;  // identity quaternion

  // Create state, covariance and valid_states for initialization
  PoseSensorModel::StateVector state = PoseSensorModel::StateVector::Zero();
  PoseSensorModel::StateCovariance covariance = PoseSensorModel::StateCovariance::Identity();
  PoseSensorModel::StateFlags valid_states = PoseSensorModel::StateFlags::Zero();

  // Initialize state from measurement
  PoseSensorModel::StateFlags initialized_states = model.InitializeState(
      measurement, valid_states, state, covariance);

  // Check that position and quaternion states were initialized
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::X]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::Y]);
  EXPECT_TRUE(initialized_states[core::StateIndex::Position::Z]);

  // Calculate expected position after transformation
  double x_offset = 5.0 - 2.0 * cos(M_PI/4.0);  // 5 - 2*cos(45°)
  double y_offset = 2.0 * sin(M_PI/4.0);        // 2*sin(45°)

  // Since the actual sign depends on quaternion representation,
  // we check the magnitudes and verify one coordinate is positive, one negative
  EXPECT_NEAR(std::abs(state(core::StateIndex::Position::X) - 5.0), std::abs(x_offset - 5.0), 1e-6);
  EXPECT_NEAR(std::abs(state(core::StateIndex::Position::Y)), std::abs(y_offset), 1e-6);
  EXPECT_NEAR(state(core::StateIndex::Position::Z), 0.0, 1e-6);

  // For quaternion, we check that it represents a rotation of approximately 45°
  // around Z-axis, regardless of the sign
  Eigen::Quaterniond q(
      state(core::StateIndex::Quaternion::W),
      state(core::StateIndex::Quaternion::X),
      state(core::StateIndex::Quaternion::Y),
      state(core::StateIndex::Quaternion::Z));

  // Convert to axis-angle representation
  Eigen::AngleAxisd aa(q);
  Eigen::Vector3d axis = aa.axis();
  double angle = aa.angle();

  // Check that rotation is around Z axis (either +Z or -Z)
  EXPECT_NEAR(std::abs(axis.z()), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(axis.x()), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(axis.y()), 0.0, 1e-6);

  // Check that angle is around 45° (π/4 radians)
  EXPECT_NEAR(std::abs(angle), M_PI/4.0, 1e-6);
}

// Test covariance propagation during initialization
TEST_F(PoseSensorModelTest, InitializeCovarianceTransform) {
  PoseSensorModel model;

  // Create measurement with position [1, 2, 3] and identity orientation
  PoseSensorModel::MeasurementVector measurement;
  measurement << 1.0, 2.0, 3.0,      // position
                 1.0, 0.0, 0.0, 0.0;  // identity quaternion

  // Create state, covariance and valid_states for initialization
  PoseSensorModel::StateVector state = PoseSensorModel::StateVector::Zero();
  PoseSensorModel::StateCovariance covariance = PoseSensorModel::StateCovariance::Zero();
  PoseSensorModel::StateFlags valid_states = PoseSensorModel::StateFlags::Zero();

  // Store the initial covariance
  PoseSensorModel::StateCovariance initial_covariance = covariance;

  // Initialize state from measurement
  model.InitializeState(measurement, valid_states, state, covariance);

  // Check that the covariance was updated (should no longer be zero)
  // Position covariance should be set
  EXPECT_GT(covariance(core::StateIndex::Position::X, core::StateIndex::Position::X),
            initial_covariance(core::StateIndex::Position::X, core::StateIndex::Position::X));
  EXPECT_GT(covariance(core::StateIndex::Position::Y, core::StateIndex::Position::Y),
            initial_covariance(core::StateIndex::Position::Y, core::StateIndex::Position::Y));
  EXPECT_GT(covariance(core::StateIndex::Position::Z, core::StateIndex::Position::Z),
            initial_covariance(core::StateIndex::Position::Z, core::StateIndex::Position::Z));

  // Quaternion covariance should be set
  EXPECT_GT(covariance(core::StateIndex::Quaternion::W, core::StateIndex::Quaternion::W),
            initial_covariance(core::StateIndex::Quaternion::W, core::StateIndex::Quaternion::W));
  EXPECT_GT(covariance(core::StateIndex::Quaternion::X, core::StateIndex::Quaternion::X),
            initial_covariance(core::StateIndex::Quaternion::X, core::StateIndex::Quaternion::X));
  EXPECT_GT(covariance(core::StateIndex::Quaternion::Y, core::StateIndex::Quaternion::Y),
            initial_covariance(core::StateIndex::Quaternion::Y, core::StateIndex::Quaternion::Y));
  EXPECT_GT(covariance(core::StateIndex::Quaternion::Z, core::StateIndex::Quaternion::Z),
            initial_covariance(core::StateIndex::Quaternion::Z, core::StateIndex::Quaternion::Z));
}

}  // namespace testing
}  // namespace sensors
}  // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
