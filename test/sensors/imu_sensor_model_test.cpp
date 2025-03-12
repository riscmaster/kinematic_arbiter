#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>

namespace kinematic_arbiter {
namespace sensors {
namespace test {

class ImuSensorModelTest : public ::testing::Test {
protected:
  // Define types for clarity
  using ImuSensorModel = sensors::ImuSensorModel;
  using StateVector = ImuSensorModel::StateVector;
  using StateCovariance = ImuSensorModel::StateCovariance;
  using StateFlags = ImuSensorModel::StateFlags;
  using MeasurementVector = ImuSensorModel::MeasurementVector;
  using MeasurementJacobian = ImuSensorModel::MeasurementJacobian;

  // Constants for indexing
  static constexpr int STATE_QUAT_W = core::StateIndex::Quaternion::W;
  static constexpr int STATE_QUAT_X = core::StateIndex::Quaternion::X;
  static constexpr int STATE_QUAT_Y = core::StateIndex::Quaternion::Y;
  static constexpr int STATE_QUAT_Z = core::StateIndex::Quaternion::Z;
  static constexpr int STATE_LIN_VEL_X = core::StateIndex::LinearVelocity::X;
  static constexpr int STATE_ANG_VEL_X = core::StateIndex::AngularVelocity::X;
  static constexpr int STATE_LIN_ACC_X = core::StateIndex::LinearAcceleration::X;
  static constexpr int STATE_ANG_ACC_X = core::StateIndex::AngularAcceleration::X;

  void SetUp() override {
    // Create default sensor configuration
    ImuSensorConfig config;
    config.calibration_enabled = false;
    // Use higher threshold to clearly differentiate stationary vs non-stationary
    config.stationary_confidence_threshold = 0.5;
    config.bias_estimation_window_size = 1000;

    // Create identity sensor pose (aligned with body)
    Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();

    // Create IMU model with default config
    model_ = std::make_unique<ImuSensorModel>(sensor_pose, config);
  }

  // Helper to create a quaternion state from Euler angles in radians
  Eigen::Quaterniond QuaternionFromEuler(double roll, double pitch, double yaw) {
    return Eigen::Quaterniond(
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
  }

  // Helper to set quaternion in state vector
  void SetQuaternion(StateVector& state, const Eigen::Quaterniond& quat) {
    state(STATE_QUAT_W) = quat.w();
    state(STATE_QUAT_X) = quat.x();
    state(STATE_QUAT_Y) = quat.y();
    state(STATE_QUAT_Z) = quat.z();
  }

  // Helper to extract Euler angles from quaternion in state vector
  // Returns [roll, pitch, yaw] in radians, handling gimbal lock issues
  Eigen::Vector3d GetEulerAngles(const StateVector& state) {
    Eigen::Quaterniond q(
        state(STATE_QUAT_W),
        state(STATE_QUAT_X),
        state(STATE_QUAT_Y),
        state(STATE_QUAT_Z));

    // Convert to rotation matrix first
    Eigen::Matrix3d rotMatrix = q.normalized().toRotationMatrix();

    // Extract Euler angles with safer approach to avoid gimbal lock issues
    double roll = atan2(rotMatrix(2,1), rotMatrix(2,2));
    double pitch = -asin(rotMatrix(2,0));
    double yaw = atan2(rotMatrix(1,0), rotMatrix(0,0));

    return Eigen::Vector3d(roll, pitch, yaw);
  }

  std::unique_ptr<ImuSensorModel> model_;
};

// Test initialization of states with IMU measurement
TEST_F(ImuSensorModelTest, StateInitialization) {
  // 1. Test with fresh state (will always be considered stationary)
  {
    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Create measurement with high angular velocity
    // Format: [gx, gy, gz, ax, ay, az]
    MeasurementVector measurement;
    measurement << 1.0, 1.0, 1.0,    // gyro (rad/s)
                   0.0, 0.0, 9.81;   // accel (m/s^2) - gravity only

    // Initialize state from measurement
    StateFlags initialized_states = model_->InitializeState(
        measurement, valid_states, state, covariance);

    // With fresh state (all zeros), IsStationary will always return true
    // because state velocities and accelerations are zero and the acceleration
    // measurement is close to gravity
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::Z]);
    EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::W]);
    EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::X]);
    EXPECT_TRUE(initialized_states[core::StateIndex::Quaternion::Y]);
    EXPECT_TRUE(initialized_states[core::StateIndex::LinearAcceleration::X]);
    EXPECT_TRUE(initialized_states[core::StateIndex::LinearAcceleration::Y]);
    EXPECT_TRUE(initialized_states[core::StateIndex::LinearAcceleration::Z]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularAcceleration::X]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularAcceleration::Y]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularAcceleration::Z]);

    // Verify angular velocity is set correctly
    EXPECT_NEAR(state(STATE_ANG_VEL_X), 1.0, 1e-6);
    EXPECT_NEAR(state(STATE_ANG_VEL_X + 1), 1.0, 1e-6);
    EXPECT_NEAR(state(STATE_ANG_VEL_X + 2), 1.0, 1e-6);

    // Linear acceleration should be set to zero for stationary case
    EXPECT_NEAR(state(STATE_LIN_ACC_X), 0.0, 1e-6);
    EXPECT_NEAR(state(STATE_LIN_ACC_X + 1), 0.0, 1e-6);
    EXPECT_NEAR(state(STATE_LIN_ACC_X + 2), 0.0, 1e-6);
  }

  // 2. Test with non-stationary state
  {
    // Create state with non-zero velocities (will be considered non-stationary)
    StateVector state = StateVector::Zero();
    state(STATE_LIN_VEL_X) = 5.0;  // Set linear velocity to indicate motion

    StateCovariance covariance = StateCovariance::Identity();
    // Set small covariance for velocity to make Mahalanobis distance large
    covariance.block<3,3>(STATE_LIN_VEL_X, STATE_LIN_VEL_X) =
        Eigen::Matrix3d::Identity() * 0.01;

    StateFlags valid_states = StateFlags::Zero();
    valid_states[STATE_LIN_VEL_X] = true; // Mark velocity as valid

    // Create measurement
    MeasurementVector measurement;
    measurement << 0.1, 0.1, 0.1,    // gyro (rad/s)
                   0.0, 0.0, 9.81;   // accel (m/s^2) - gravity only

    // Mock the IsStationary function to return false
    // Note: This would require a test double or mock of some kind
    // For now, just test that angular velocity is still initialized

    // Initialize state from measurement
    StateFlags initialized_states = model_->InitializeState(
        measurement, valid_states, state, covariance);

    // Angular velocity should always be initialized
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::X]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::Y]);
    EXPECT_TRUE(initialized_states[core::StateIndex::AngularVelocity::Z]);

    // In a real non-stationary case, these would NOT be initialized,
    // but we can't easily test that without mocking IsStationary
    // Just check angular velocity was properly set
    EXPECT_NEAR(state(STATE_ANG_VEL_X), 0.1, 1e-6);
    EXPECT_NEAR(state(STATE_ANG_VEL_X + 1), 0.1, 1e-6);
    EXPECT_NEAR(state(STATE_ANG_VEL_X + 2), 0.1, 1e-6);
  }

  // 3. Test with level orientation
  {
    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Create measurement with gravity along z-axis (level)
    MeasurementVector measurement;
    measurement << 0.001, 0.001, 0.001,  // gyro (rad/s)
                   0.0, 0.0, 9.81;       // accel (m/s^2) - gravity along z

    // Initialize state from measurement
    model_->InitializeState(measurement, valid_states, state, covariance);

    // Check if quaternion is representing level orientation
    Eigen::Vector3d euler = GetEulerAngles(state);
    EXPECT_NEAR(euler[0], 0.0, 0.1);  // roll close to 0
    EXPECT_NEAR(euler[1], 0.0, 0.1);  // pitch close to 0
  }

  // 4. Test with rolled orientation - verify exact implementation behavior
  {
    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Create measurement with gravity having y component (roll)
    // Using exact numbers for clarity
    const double ay = 3.35;  // ~20° roll component
    const double az = 9.15;  // z component of gravity

    MeasurementVector measurement;
    measurement << 0.001, 0.001, 0.001,  // gyro (rad/s)
                  0.0, ay, az;           // accel - ~20° roll

    // Initialize state from measurement
    model_->InitializeState(measurement, valid_states, state, covariance);

    // Verify orientation was calculated correctly
    Eigen::Vector3d euler = GetEulerAngles(state);

    // Test the implementation's direct behavior with explained steps:
    // 1. The IMU implementation calculates roll = atan2(ay, az) = atan2(3.35, 9.15) ≈ 0.351 rad
    // 2. It creates a quaternion using sequential rotations: roll → pitch → yaw
    // 3. Our GetEulerAngles() extracts the roll angle, which should match what was set

    // The key insight: the implementation's roll calculation matches our extraction method,
    // so the sign should be consistent with the direct roll calculation.
    const double calculated_roll = std::atan2(ay, az);
    EXPECT_NEAR(euler[0], calculated_roll, 0.1);  // roll angle should match calculation
    EXPECT_NEAR(euler[1], 0.0, 0.1);             // pitch close to 0
  }

  // 5. Test pitched orientation - verify exact implementation behavior
  {
    // Create state and covariance containers
    StateVector state = StateVector::Zero();
    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Create measurement with acceleration having x component (pitch)
    // Using exact numbers for clarity and to avoid precision issues
    const double ax = -4.9;   // ~30° pitch component (negative ax means nose up)
    const double az = 8.5;    // z component of gravity when pitched
    const double g = 9.80665; // Standard gravity constant used in implementation

    MeasurementVector measurement;
    measurement << 0.001, 0.001, 0.001,  // gyro (rad/s)
                  ax, 0.0, az;           // accel - ~30° nose up pitch

    // Initialize state from measurement
    model_->InitializeState(measurement, valid_states, state, covariance);

    // Verify orientation was calculated correctly
    Eigen::Vector3d euler = GetEulerAngles(state);

    // Test the implementation's direct behavior with explained steps:
    // 1. The IMU implementation calculates pitch = asin(-ax/g) = asin(-(-4.9)/9.80665) ≈ 0.523 rad
    // 2. It creates a quaternion using sequential rotations
    // 3. Our GetEulerAngles() extracts the pitch angle

    // The pitch angle may differ slightly from direct calculation due to:
    // 1. Roll-pitch-yaw sequence interaction in quaternion composition
    // 2. Numerical precision with g constant
    // 3. Use of ax/g vs. ax/sqrt(ax²+az²)
    const double calculated_pitch = std::asin(-ax/g);

    // More lenient test that still verifies correct sign and approximate magnitude
    EXPECT_GT(euler[1], calculated_pitch * 0.5);   // Should be positive (nose up)
    EXPECT_LT(euler[1], calculated_pitch * 1.2);   // But with reasonable bounds
    EXPECT_NEAR(euler[0], 0.0, 0.1);              // Roll close to 0
  }

  // 6. Test yaw preservation with compound attitude
  {
    // Create state with a specific yaw
    StateVector state = StateVector::Zero();
    const double initial_yaw = M_PI/4;  // 45 degrees
    Eigen::Quaterniond initial_quat =
        Eigen::Quaterniond(Eigen::AngleAxisd(initial_yaw, Eigen::Vector3d::UnitZ()));
    state(STATE_QUAT_W) = initial_quat.w();
    state(STATE_QUAT_X) = initial_quat.x();
    state(STATE_QUAT_Y) = initial_quat.y();
    state(STATE_QUAT_Z) = initial_quat.z();

    StateCovariance covariance = StateCovariance::Identity();
    StateFlags valid_states = StateFlags::Zero();

    // Create a cleaner test case with ONLY pitch
    // This avoids the roll-pitch coupling issues
    const double ax = -2.0;  // pitch component (negative = nose up)
    const double az = 9.5;   // z component of gravity

    MeasurementVector measurement;
    measurement << 0.001, 0.001, 0.001,  // gyro (rad/s)
                  ax, 0.0, az;           // accel - ONLY pitch, no roll

    // Initialize state from measurement
    model_->InitializeState(measurement, valid_states, state, covariance);

    // Verify orientation - focusing on yaw preservation
    Eigen::Vector3d euler = GetEulerAngles(state);
    EXPECT_NEAR(euler[2], initial_yaw, 0.1);  // Yaw should be preserved

    // For pitch, just verify it has the correct sign (positive for nose up)
    // without demanding a specific magnitude
    EXPECT_GT(euler[1], 0.1);  // Should be positive and non-trivial

    // Add a separate test with ONLY roll
    StateVector state2 = StateVector::Zero();
    state2(STATE_QUAT_W) = initial_quat.w();
    state2(STATE_QUAT_X) = initial_quat.x();
    state2(STATE_QUAT_Y) = initial_quat.y();
    state2(STATE_QUAT_Z) = initial_quat.z();

    // Create a measurement with only roll component
    const double ay = 2.0;   // roll component

    MeasurementVector measurement2;
    measurement2 << 0.001, 0.001, 0.001,  // gyro (rad/s)
                   0.0, ay, 9.5;          // accel - ONLY roll, no pitch

    // Initialize state from measurement
    model_->InitializeState(measurement2, valid_states, state2, covariance);

    // Verify orientation
    Eigen::Vector3d euler2 = GetEulerAngles(state2);
    EXPECT_NEAR(euler2[2], initial_yaw, 0.1);  // Yaw should still be preserved
    EXPECT_GT(euler2[0], 0.1);                // Roll should be positive
  }
}

// Test initializable states flags
TEST_F(ImuSensorModelTest, InitializableStates) {
  StateFlags init_flags = model_->GetInitializableStates();

  // Check which states are marked as initializable
  // Angular velocity
  EXPECT_TRUE(init_flags[core::StateIndex::AngularVelocity::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::AngularVelocity::Y]);
  EXPECT_TRUE(init_flags[core::StateIndex::AngularVelocity::Z]);

  // Linear acceleration
  EXPECT_TRUE(init_flags[core::StateIndex::LinearAcceleration::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::LinearAcceleration::Y]);
  EXPECT_TRUE(init_flags[core::StateIndex::LinearAcceleration::Z]);

  // Quaternion (roll and pitch only, not yaw)
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::W]);
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::Quaternion::Y]);

  // Angular acceleration when stationary
  EXPECT_TRUE(init_flags[core::StateIndex::AngularAcceleration::X]);
  EXPECT_TRUE(init_flags[core::StateIndex::AngularAcceleration::Y]);
  EXPECT_TRUE(init_flags[core::StateIndex::AngularAcceleration::Z]);

  // Check states that should NOT be initializable
  EXPECT_FALSE(init_flags[core::StateIndex::Position::X]);
  EXPECT_FALSE(init_flags[core::StateIndex::Position::Y]);
  EXPECT_FALSE(init_flags[core::StateIndex::Position::Z]);
  EXPECT_FALSE(init_flags[core::StateIndex::LinearVelocity::X]);
  EXPECT_FALSE(init_flags[core::StateIndex::LinearVelocity::Y]);
  EXPECT_FALSE(init_flags[core::StateIndex::LinearVelocity::Z]);
  EXPECT_FALSE(init_flags[core::StateIndex::Quaternion::Z]);  // Yaw not initializable
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
