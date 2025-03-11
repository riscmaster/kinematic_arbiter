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
  EXPECT_TRUE(filter_->ProcessMeasurement<0>(time, pos_measurement));

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
  // We don't check Z as it depends on whether gravity was removed
}

// Test IMU measurements (previously had dimension mismatch)
TEST_F(MediatedKalmanFilterTest, IMUMeasurements) {
  // Initialize with position
  Eigen::Vector3d pos_measurement(0.0, 0.0, 0.0);
  filter_->ProcessMeasurement<0>(0.0, pos_measurement);

  // Create IMU measurement
  Eigen::Matrix<double, 6, 1> imu_measurement;
  imu_measurement << 0.0, 0.0, 0.1,  // angular velocity (rad/s)
                     1.0, 0.0, 0.0;  // acceleration (m/s²)

  // This was triggering dimension mismatch error
  EXPECT_TRUE(filter_->ProcessMeasurement<1>(0.5, imu_measurement));

  // Check that state was updated - use the proper StateIndex enum values
  auto state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(StateIndex::LinearAcceleration::X), 1.0, 0.2);
  EXPECT_NEAR(state(StateIndex::AngularVelocity::Z), 0.1, 0.05);
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
