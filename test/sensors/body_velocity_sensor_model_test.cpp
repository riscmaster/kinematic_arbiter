#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>

// Namespace qualifications
using kinematic_arbiter::core::StateIndex;
using kinematic_arbiter::sensors::BodyVelocitySensorModel;

class BodyVelocitySensorModelTest : public ::testing::Test {
protected:
  // Define types through the sensor model class
  using StateVector = BodyVelocitySensorModel::StateVector;
  using StateCovariance = BodyVelocitySensorModel::StateCovariance;
  using StateFlags = BodyVelocitySensorModel::StateFlags;
  using MeasurementVector = BodyVelocitySensorModel::MeasurementVector;
  using MeasurementJacobian = BodyVelocitySensorModel::MeasurementJacobian;
  using MeasurementCovariance = BodyVelocitySensorModel::MeasurementCovariance;

  // Constants for clearer indexing
  static constexpr int MEAS_LIN_VEL = 0; // First three elements are linear velocity
  static constexpr int MEAS_ANG_VEL = 3; // Last three elements are angular velocity
  static constexpr int STATE_LIN_VEL = StateIndex::LinearVelocity::X;
  static constexpr int STATE_ANG_VEL = StateIndex::AngularVelocity::X;

  void SetUp() override {
    // Define test velocity values
    body_lin_vel_ << 0.5, -0.3, 0.1;
    body_ang_vel_ << 0.1, 0.2, -0.3;

    // Initialize state vector
    test_state_.setZero();
    test_state_.segment<3>(STATE_LIN_VEL) = body_lin_vel_;
    test_state_.segment<3>(STATE_ANG_VEL) = body_ang_vel_;

    // Create an offset and rotated sensor transform
    const double sensor_angle = M_PI / 4.0;  // 45 degrees
    const Eigen::Vector3d sensor_offset(0.2, 0.3, -0.1);

    Eigen::AngleAxisd rotation(sensor_angle, Eigen::Vector3d::UnitZ());
    sensor_transform_ = Eigen::Isometry3d::Identity();
    sensor_transform_.linear() = rotation.toRotationMatrix();
    sensor_transform_.translation() = sensor_offset;
  }

  // Test state and velocities
  StateVector test_state_;
  Eigen::Vector3d body_lin_vel_;
  Eigen::Vector3d body_ang_vel_;
  Eigen::Isometry3d sensor_transform_;
};

// Test measurement prediction with identity transform
TEST_F(BodyVelocitySensorModelTest, PredictMeasurementIdentity) {
  const Eigen::Isometry3d identity_transform = Eigen::Isometry3d::Identity();
  BodyVelocitySensorModel model(identity_transform);

  const auto measurement = model.PredictMeasurement(test_state_);

  // With identity transform, predicted measurement should match state velocities
  EXPECT_TRUE(measurement.segment<3>(MEAS_LIN_VEL).isApprox(body_lin_vel_));
  EXPECT_TRUE(measurement.segment<3>(MEAS_ANG_VEL).isApprox(body_ang_vel_));
}

// Test initialization of states from body velocity
TEST_F(BodyVelocitySensorModelTest, StateInitialization) {
  // 1. Test with sensor aligned with body frame
  {
    // Create a sensor model with identity transform (aligned with body)
    BodyVelocitySensorModel aligned_model(Eigen::Isometry3d::Identity());

    // Create measurement vector [vx, vy, vz, wx, wy, wz]
    MeasurementVector measurement;
    measurement << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3;

    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Initialize state from measurement
    StateFlags initialized_states = aligned_model.InitializeState(
        measurement, valid_states, state, covariance);

    // Verify all velocity states were initialized
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Z]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Z]);

    // Verify state values (should match measurement for aligned sensor)
    EXPECT_NEAR(state(StateIndex::LinearVelocity::X), 1.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Y), 2.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Z), 3.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.1, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.2, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.3, 1e-6);
  }

  // 2. Test with offset sensor
  {
    // Create a sensor with translation offset
    Eigen::Isometry3d offset_transform = Eigen::Isometry3d::Identity();
    offset_transform.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);  // 1m offset in x
    BodyVelocitySensorModel offset_model(offset_transform);

    // Create measurement vector [vx, vy, vz, wx, wy, wz]
    MeasurementVector measurement;
    measurement << 1.0, 2.0, 3.0, 0.0, 0.0, 1.0;  // 1 rad/s around z

    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Note: Quaternion validity is irrelevant for body velocity initialization
    // Initialize state from measurement
    StateFlags initialized_states = offset_model.InitializeState(
        measurement, valid_states, state, covariance);

    // Verify all velocity states were initialized
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Z]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Z]);

    // Verify state values with lever arm effect
    // With 1m x-offset and 1 rad/s around z, we expect y-component of linear velocity
    // to be reduced by 1.0 m/s
    EXPECT_NEAR(state(StateIndex::LinearVelocity::X), 1.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Y), 1.0, 1e-6);  // 2.0 - 1.0 = 1.0
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Z), 3.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 1.0, 1e-6);
  }

  // 3. Test with rotated sensor (sensor frame different from body frame)
  {
    // Create a sensor with rotation
    Eigen::Isometry3d rotated_transform = Eigen::Isometry3d::Identity();
    // 45 degree rotation around Z axis
    rotated_transform.linear() = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    BodyVelocitySensorModel rotated_model(rotated_transform);

    // Create measurement vector [vx, vy, vz, wx, wy, wz] in sensor frame
    MeasurementVector measurement;
    measurement << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;  // Moving forward in sensor frame

    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Initialize state from measurement
    StateFlags initialized_states = rotated_model.InitializeState(
        measurement, valid_states, state, covariance);

    // Verify all velocity states were initialized
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Z]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Z]);

    // In body frame, this should be x=cos(pi/4), y=sin(pi/4) = 0.7071...
    const double sqrt2_2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::X), sqrt2_2, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Y), sqrt2_2, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Z), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 1.0, 1e-6);
  }

  // 4. Test with complex transform (both rotation and translation)
  {
    // Create a sensor with offset and rotation
    Eigen::Isometry3d complex_transform = Eigen::Isometry3d::Identity();
    complex_transform.translation() = Eigen::Vector3d(0.0, 1.0, 0.0);  // 1m offset in y
    complex_transform.linear() = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    BodyVelocitySensorModel complex_model(complex_transform);

    // Create measurement vector [vx, vy, vz, wx, wy, wz]
    MeasurementVector measurement;
    measurement << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;  // Forward motion and rotation around z

    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Initialize state from measurement
    StateFlags initialized_states = complex_model.InitializeState(
        measurement, valid_states, state, covariance);

    // Verify all velocity states were initialized regardless of quaternion validity
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::LinearVelocity::Z]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[StateIndex::AngularVelocity::Z]);

    // Calculate expected values
    const double sqrt2_2 = 1.0 / std::sqrt(2.0);

    // Linear velocity transforms to [sqrt2_2, sqrt2_2, 0] in body frame
    // Lever arm effect: with 1.0m in y and rotation 1.0 rad/s around z
    // This subtracts [−1.0, 0.0, 0.0] from velocity
    // (ω × r = [0,0,1] × [0,1,0] = [−1,0,0])
    // The implementation does: sensor_lin_vel_in_body - lever_arm_effect
    // So: [sqrt2_2, sqrt2_2, 0] - [-1, 0, 0] = [sqrt2_2 + 1, sqrt2_2, 0]
    EXPECT_NEAR(state(StateIndex::LinearVelocity::X), sqrt2_2 + 1.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Y), sqrt2_2, 1e-6);
    EXPECT_NEAR(state(StateIndex::LinearVelocity::Z), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.0, 1e-6);
    EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 1.0, 1e-6);
  }

  // 5. Test covariance propagation
  {
    BodyVelocitySensorModel model(Eigen::Isometry3d::Identity());

    // Create measurement vector
    MeasurementVector measurement;
    measurement << 1.0, 2.0, 3.0, 0.1, 0.2, 0.3;

    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Zero();
    StateFlags valid_states = StateFlags::Zero();

    // Initialize state from measurement
    model.InitializeState(measurement, valid_states, state, covariance);

    // Verify covariance was set
    EXPECT_GT(covariance(StateIndex::LinearVelocity::X, StateIndex::LinearVelocity::X), 0.0);
    EXPECT_GT(covariance(StateIndex::LinearVelocity::Y, StateIndex::LinearVelocity::Y), 0.0);
    EXPECT_GT(covariance(StateIndex::LinearVelocity::Z, StateIndex::LinearVelocity::Z), 0.0);
    EXPECT_GT(covariance(StateIndex::AngularVelocity::X, StateIndex::AngularVelocity::X), 0.0);
    EXPECT_GT(covariance(StateIndex::AngularVelocity::Y, StateIndex::AngularVelocity::Y), 0.0);
    EXPECT_GT(covariance(StateIndex::AngularVelocity::Z, StateIndex::AngularVelocity::Z), 0.0);
  }
}

// Test initializable states flags
TEST_F(BodyVelocitySensorModelTest, InitializableStates) {
  BodyVelocitySensorModel model(Eigen::Isometry3d::Identity());
  StateFlags init_flags = model.GetInitializableStates();

  // Check that only velocity states are marked as initializable
  EXPECT_TRUE(init_flags[StateIndex::LinearVelocity::X]);
  EXPECT_TRUE(init_flags[StateIndex::LinearVelocity::Y]);
  EXPECT_TRUE(init_flags[StateIndex::LinearVelocity::Z]);
  EXPECT_TRUE(init_flags[StateIndex::AngularVelocity::X]);
  EXPECT_TRUE(init_flags[StateIndex::AngularVelocity::Y]);
  EXPECT_TRUE(init_flags[StateIndex::AngularVelocity::Z]);

  // Check that other states are not initializable
  EXPECT_FALSE(init_flags[StateIndex::Position::X]);
  EXPECT_FALSE(init_flags[StateIndex::Quaternion::W]);
  EXPECT_FALSE(init_flags[StateIndex::LinearAcceleration::X]);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
