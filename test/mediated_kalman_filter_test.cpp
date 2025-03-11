#include <gtest/gtest.h>
#include <kinematic_arbiter/core/mediated_kalman_filter.hpp>
#include <kinematic_arbiter/core/state_index.hpp>
#include <kinematic_arbiter/models/rigid_body_state_model.hpp>
#include <kinematic_arbiter/sensors/position_sensor_model.hpp>
#include <kinematic_arbiter/sensors/pose_sensor_model.hpp>
#include <kinematic_arbiter/sensors/imu_sensor_model.hpp>
#include <kinematic_arbiter/sensors/body_velocity_sensor_model.hpp>
#include <kinematic_arbiter/sensors/heading_velocity_sensor_model.hpp>

using namespace kinematic_arbiter;
using namespace kinematic_arbiter::core;
using namespace kinematic_arbiter::models;
using namespace kinematic_arbiter::sensors;

// Test fixture
class MediatedKalmanFilterTest : public ::testing::Test {
protected:
  static constexpr int kStateSize = StateIndex::kFullStateSize;

  void SetUp() override {
    // Create process model
    process_model_ = std::make_shared<RigidBodyStateModel>();

    // Create position sensor with proper configuration
    position_sensor_ = std::make_shared<PositionSensorModel>();

    // Simplify the IMU sensor creation - use default constructor
    imu_sensor_ = std::make_shared<ImuSensorModel>();
    // Create filter with minimal setup
    filter_ = std::make_shared<MediatedKalmanFilter<kStateSize, RigidBodyStateModel,
                                                  PositionSensorModel,
                                                  ImuSensorModel>>(
        process_model_,
        position_sensor_,
        imu_sensor_
    );

    filter_->SetMaxDelayWindow(1.0);
  }

  std::shared_ptr<RigidBodyStateModel> process_model_;
  std::shared_ptr<PositionSensorModel> position_sensor_;
  std::shared_ptr<ImuSensorModel> imu_sensor_;
  std::shared_ptr<MediatedKalmanFilter<kStateSize, RigidBodyStateModel,
                                      PositionSensorModel,
                                      ImuSensorModel>> filter_;
};

// Test initialization
TEST_F(MediatedKalmanFilterTest, Initialization) {
  ASSERT_TRUE(filter_);

  // Initialize with position measurement
  Eigen::Vector3d pos_measurement(1.0, 2.0, 3.0);
  double time = 0.0;
  EXPECT_FALSE(filter_->IsInitialized());
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(time, pos_measurement));
  EXPECT_TRUE(filter_->IsInitialized());

  // Verify initialized position values
  auto state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::Position::X), 1.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Y), 2.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::Position::Z), 3.0, 1e-6);

  // Verify time was set
  EXPECT_DOUBLE_EQ(filter_->GetCurrentTime(), time);

  // Test initializing IMU states
  Eigen::Matrix<double, 6, 1> imu_measurement;
  imu_measurement << 0.1, 0.2, 0.3,  // angular velocity (rad/s)
                     0.0, 0.0, 9.81; // acceleration (m/s²)

  // Process IMU measurement
  EXPECT_TRUE(filter_->ProcessMeasurement<1>(0.1, imu_measurement));

  // Verify angular velocity was initialized
  state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.1, 1e-6);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.2, 1e-6);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.3, 1e-6);

  // Verify linear acceleration was initialized
  // Note: The exact values depend on the IMU sensor model's implementation
  // But they should be close to the input values accounting for gravity
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::X), 0.0, 0.5);
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::Y), 0.0, 0.5);
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::Z), 0.0, 0.5); // Stationary should be ~0 after gravity compensation

  // Check quaternion initialization (roll and pitch from gravity)
  // For a stationary IMU with gravity in Z direction, we expect roll and pitch near zero
  // and the quaternion to be close to identity [1,0,0,0]
  EXPECT_NEAR(state(StateIndex::Quaternion::W), 1.0, 0.1);
  EXPECT_NEAR(state(StateIndex::Quaternion::X), 0.0, 0.1);
  EXPECT_NEAR(state(StateIndex::Quaternion::Y), 0.0, 0.1);
  // We don't check Z (yaw) as it can't be determined from gravity alone
}

// Test IMU initialization
TEST_F(MediatedKalmanFilterTest, IMUInitialization) {
  // Create IMU measurement for initialization
  Eigen::Matrix<double, 6, 1> imu_init_measurement;
  imu_init_measurement << 0.0, 0.0, 0.0,  // angular velocity (rad/s)
                           0.0, 0.0, 9.81; // acceleration (m/s²) - stationary with gravity

  // Process IMU measurement for initialization

  EXPECT_FALSE(filter_->IsInitialized());
  EXPECT_TRUE(filter_->ProcessMeasurement<1>(0.0, imu_init_measurement));
  EXPECT_TRUE(filter_->IsInitialized());

  // Verify angular velocity was initialized
  auto state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::AngularVelocity::X), 0.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Y), 0.0, 1e-6);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.0, 1e-6);

  // Verify linear acceleration was initialized (compensated for gravity)
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::X), 0.0, 0.1);
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::Y), 0.0, 0.1);
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::Z), 0.0, 0.1);

  // Check quaternion initialization (roll and pitch from gravity)
  EXPECT_NEAR(state(StateIndex::Quaternion::W), 1.0, 0.1);
  EXPECT_NEAR(state(StateIndex::Quaternion::X), 0.0, 0.1);
  EXPECT_NEAR(state(StateIndex::Quaternion::Y), 0.0, 0.1);
}

// Test IMU measurements (for update, not initialization)
TEST_F(MediatedKalmanFilterTest, IMUMeasurements) {
  // First initialize the filter with position
  Eigen::Vector3d pos_measurement(0.0, 0.0, 0.0);
  filter_->ProcessMeasurement<0>(0.0, pos_measurement);

  // Initialize with IMU - zero motion
  Eigen::Matrix<double, 6, 1> imu_init;
  imu_init << 0.0, 0.0, 0.0,  // zero angular velocity
              0.0, 0.0, 9.81; // stationary with gravity
  filter_->ProcessMeasurement<1>(0.0, imu_init);

  // Set state covariance high for acceleration and angular velocity
  // to ensure measurements are trusted completely
  auto covariance = filter_->GetStateCovariance();
  covariance(StateIndex::LinearAcceleration::X, StateIndex::LinearAcceleration::X) = 100.0;
  covariance(StateIndex::LinearAcceleration::Y, StateIndex::LinearAcceleration::Y) = 100.0;
  covariance(StateIndex::LinearAcceleration::Z, StateIndex::LinearAcceleration::Z) = 100.0;
  covariance(StateIndex::AngularVelocity::X, StateIndex::AngularVelocity::X) = 100.0;
  covariance(StateIndex::AngularVelocity::Y, StateIndex::AngularVelocity::Y) = 100.0;
  covariance(StateIndex::AngularVelocity::Z, StateIndex::AngularVelocity::Z) = 100.0;
  filter_->SetStateCovariance(covariance);

  // Now test actual measurement update with high trust in measurements
  Eigen::Matrix<double, 6, 1> imu_measurement;
  imu_measurement << 0.0, 0.0, 0.1,  // angular velocity (rad/s)
                     1.0, 0.0, 0.0;  // acceleration (m/s²)

  // Process measurement update at t=0.5
  EXPECT_TRUE(filter_->ProcessMeasurement<1>(0.5, imu_measurement));

  // Check that state was updated almost exactly to measurement values
  // (higher precision due to high state covariance)
  auto state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::X), 1.0, 0.05);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.1, 0.01);

  // Now test with low state covariance (high confidence in current state)
  covariance = filter_->GetStateCovariance();
  covariance(StateIndex::LinearAcceleration::X, StateIndex::LinearAcceleration::X) = 0.001;
  covariance(StateIndex::AngularVelocity::Z, StateIndex::AngularVelocity::Z) = 0.001;
  filter_->SetStateCovariance(covariance);

  // New measurement with different values
  imu_measurement << 0.0, 0.0, 0.5,  // much higher angular velocity
                     2.0, 0.0, 0.0;  // higher acceleration

  EXPECT_TRUE(filter_->ProcessMeasurement<1>(1.0, imu_measurement));

  // Check that state barely changed due to low state covariance
  state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::X), 1.0, 0.2); // Should stay close to previous value
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.1, 0.1);    // Should stay close to previous value
}

// Test position update sequence
TEST_F(MediatedKalmanFilterTest, PositionSequence) {
  // Initialize
  Eigen::Vector3d pos1(0.0, 0.0, 0.0);
  filter_->ProcessMeasurement<0>(0.0, pos1);

  // Add more position measurements
  Eigen::Vector3d pos2(1.0, 0.0, 0.0);
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(1.0, pos2));

  Eigen::Vector3d pos3(2.0, 0.0, 0.0);
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(2.0, pos3));

  // Check that state estimates properly interpolate/extrapolate
  auto state_t1 = filter_->GetStateEstimate(1.0);
  EXPECT_NEAR(state_t1(StateIndex::Position::X), 1.0, 0.1);

  auto state_t1_5 = filter_->GetStateEstimate(1.5);
  EXPECT_NEAR(state_t1_5(StateIndex::Position::X), 1.5, 0.1);

  auto state_t3 = filter_->GetStateEstimate(3.0);
  EXPECT_NEAR(state_t3(StateIndex::Position::X), 3.0, 0.5);
}

// Test out-of-sequence measurements
TEST_F(MediatedKalmanFilterTest, OutOfSequenceMeasurements) {
  // Initialize
  Eigen::Vector3d pos1(0.0, 0.0, 0.0);
  filter_->ProcessMeasurement<0>(0.0, pos1);

  // Add second measurement
  Eigen::Vector3d pos2(1.0, 0.0, 0.0);
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(1.0, pos2));

  // Process out-of-sequence measurement
  Eigen::Vector3d pos_oosm(0.3, 0.1, 0.0);
  double oosm_time = 0.5;
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(oosm_time, pos_oosm));

  // Test too-old measurement
  Eigen::Vector3d old_measurement(0.1, 0.1, 0.1);
  double old_time = -0.5;  // Outside delay window
  EXPECT_FALSE(filter_->ProcessMeasurement<0>(old_time, old_measurement));
}

// Test combined sensor updates
TEST_F(MediatedKalmanFilterTest, CombinedSensorUpdates) {
  // Initialize with position
  Eigen::Vector3d pos_measurement(0.0, 0.0, 0.0);
  filter_->ProcessMeasurement<0>(0.0, pos_measurement);

  // Add IMU measurement
  Eigen::Matrix<double, 6, 1> imu_measurement;
  imu_measurement << 0.0, 0.0, 0.0,  // angular velocity (rad/s)
                     1.0, 0.0, 0.0;  // acceleration (m/s²)

  filter_->ProcessMeasurement<1>(0.1, imu_measurement);

  // Add another position update
  Eigen::Vector3d pos2(0.05, 0.0, 0.0); // Small movement consistent with acceleration
  filter_->ProcessMeasurement<0>(0.2, pos2);

  // Get state at future time
  auto future_state = filter_->GetStateEstimate(1.0);

  // With constant acceleration, should have moved further
  EXPECT_GT(future_state(StateIndex::Position::X), 0.05);
  EXPECT_NEAR(future_state(StateIndex::LinearVelocity::X), 1.0, 0.3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
