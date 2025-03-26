#include <gtest/gtest.h>
#include <memory>
#include <eigen3/Eigen/Dense>

#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"

// Use the correct state index type
using SIdx = kinematic_arbiter::core::StateIndex;
using SensorType = kinematic_arbiter::core::SensorType;

namespace kinematic_arbiter {

// Test fixture for filter initialization
class FilterInitializationTest : public ::testing::Test {
protected:
  // FilterType definition
  using FilterType = kinematic_arbiter::core::MediatedKalmanFilter<19,
    kinematic_arbiter::models::RigidBodyStateModel>;

  // Type aliases
  using StateVector = Eigen::Matrix<double, 19, 1>;
  using StateMatrix = Eigen::Matrix<double, 19, 19>;
  using StateFlags = Eigen::Array<bool, 19, 1>;

  void SetUp() override {
    // Create state model (this is common to all tests)
    state_model_ = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();

    // Create a fresh filter for each test
    filter_ = std::make_shared<FilterType>(state_model_);
  }

  // Member variables
  std::shared_ptr<kinematic_arbiter::models::RigidBodyStateModel> state_model_;
  std::shared_ptr<FilterType> filter_;
};

// Test initialization with position sensor
TEST_F(FilterInitializationTest, PositionSensorInitialization) {
  // Create position sensor with explicit sensor type
  auto position_sensor = std::make_shared<sensors::PositionSensorModel>();

  // Add sensor to filter with explicit sensor type
  size_t sensor_idx = filter_->AddSensor<SensorType::Position>(position_sensor);

  // Create desired state with known position
  StateVector desired_state = StateVector::Zero();
  desired_state.segment<3>(SIdx::Position::X) = Eigen::Vector3d(1.0, 2.0, 3.0);

  // Generate measurement from desired state
  Eigen::Vector3d position_measurement = position_sensor->PredictMeasurement(desired_state);

  // Process the measurement with explicit sensor type
  bool result = filter_->ProcessMeasurementByIndex<SensorType::Position>(
      sensor_idx, position_measurement, 0.0);
  EXPECT_TRUE(result) << "Failed to process position measurement";

  // Check if filter is initialized
  EXPECT_TRUE(filter_->IsInitialized()) << "Filter not initialized after position measurement";

  // Check the actual state values
  StateVector state = filter_->GetStateEstimate();
  EXPECT_NEAR(state[SIdx::Position::X], desired_state[SIdx::Position::X], 1e-6) << "Position X not correctly initialized";
  EXPECT_NEAR(state[SIdx::Position::Y], desired_state[SIdx::Position::Y], 1e-6) << "Position Y not correctly initialized";
  EXPECT_NEAR(state[SIdx::Position::Z], desired_state[SIdx::Position::Z], 1e-6) << "Position Z not correctly initialized";
}

// Test initialization with pose sensor
TEST_F(FilterInitializationTest, PoseSensorInitialization) {
  // Create pose sensor
  auto pose_sensor = std::make_shared<sensors::PoseSensorModel>();

  // Add sensor to filter with explicit sensor type
  size_t sensor_idx = filter_->AddSensor<SensorType::Pose>(pose_sensor);

  // Create desired state with known position and orientation
  StateVector desired_state = StateVector::Zero();
  desired_state.segment<3>(SIdx::Position::X) = Eigen::Vector3d(1.0, 2.0, 3.0);

  // Use the exact value of sqrt(2)/2 for the quaternion
  const double sqrt2_2 = 0.70710678118654752440084436210485;
  desired_state.segment<4>(SIdx::Quaternion::W) = Eigen::Vector4d(sqrt2_2, 0.0, sqrt2_2, 0.0); // 90 deg rotation around Y

  // Generate measurement from desired state
  Eigen::Matrix<double, 7, 1> pose_measurement = pose_sensor->PredictMeasurement(desired_state);

  // Process the measurement with explicit sensor type
  bool result = filter_->ProcessMeasurementByIndex<SensorType::Pose>(
      sensor_idx, pose_measurement, 0.0);
  EXPECT_TRUE(result) << "Failed to process pose measurement";

  // Check if filter is initialized
  EXPECT_TRUE(filter_->IsInitialized()) << "Filter not initialized after pose measurement";

  // Check the actual state values
  StateVector state = filter_->GetStateEstimate();
  EXPECT_NEAR(state[SIdx::Position::X], desired_state[SIdx::Position::X], 1e-6) << "Position X not correctly initialized";
  EXPECT_NEAR(state[SIdx::Position::Y], desired_state[SIdx::Position::Y], 1e-6) << "Position Y not correctly initialized";
  EXPECT_NEAR(state[SIdx::Position::Z], desired_state[SIdx::Position::Z], 1e-6) << "Position Z not correctly initialized";
  EXPECT_NEAR(state[SIdx::Quaternion::W], desired_state[SIdx::Quaternion::W], 1e-6) << "Quaternion W not correctly initialized";
  EXPECT_NEAR(state[SIdx::Quaternion::X], desired_state[SIdx::Quaternion::X], 1e-6) << "Quaternion X not correctly initialized";
  EXPECT_NEAR(state[SIdx::Quaternion::Y], desired_state[SIdx::Quaternion::Y], 1e-6) << "Quaternion Y not correctly initialized";
  EXPECT_NEAR(state[SIdx::Quaternion::Z], desired_state[SIdx::Quaternion::Z], 1e-6) << "Quaternion Z not correctly initialized";
}

// Test initialization with body velocity sensor
TEST_F(FilterInitializationTest, BodyVelocitySensorInitialization) {
  // Create body velocity sensor
  auto velocity_sensor = std::make_shared<sensors::BodyVelocitySensorModel>();

  // Add sensor to filter with explicit sensor type
  size_t sensor_idx = filter_->AddSensor<SensorType::BodyVelocity>(velocity_sensor);

  // Create desired state with known velocities
  StateVector desired_state = StateVector::Zero();
  desired_state.segment<3>(SIdx::LinearVelocity::X) = Eigen::Vector3d(0.1, 0.2, 0.3);
  desired_state.segment<3>(SIdx::AngularVelocity::X) = Eigen::Vector3d(0.01, 0.02, 0.03);

  // Generate measurement from desired state
  Eigen::Matrix<double, 6, 1> velocity_measurement = velocity_sensor->PredictMeasurement(desired_state);

  // Process the measurement with explicit sensor type
  bool result = filter_->ProcessMeasurementByIndex<SensorType::BodyVelocity>(
      sensor_idx, velocity_measurement, 0.0);
  EXPECT_TRUE(result) << "Failed to process velocity measurement";

  // Check if filter is initialized
  EXPECT_TRUE(filter_->IsInitialized()) << "Filter not initialized after velocity measurement";

  // Check the actual state values
  StateVector state = filter_->GetStateEstimate();
  EXPECT_NEAR(state[SIdx::LinearVelocity::X], desired_state[SIdx::LinearVelocity::X], 1e-6) << "Linear Velocity X not correctly initialized";
  EXPECT_NEAR(state[SIdx::LinearVelocity::Y], desired_state[SIdx::LinearVelocity::Y], 1e-6) << "Linear Velocity Y not correctly initialized";
  EXPECT_NEAR(state[SIdx::LinearVelocity::Z], desired_state[SIdx::LinearVelocity::Z], 1e-6) << "Linear Velocity Z not correctly initialized";
  EXPECT_NEAR(state[SIdx::AngularVelocity::X], desired_state[SIdx::AngularVelocity::X], 1e-6) << "Angular Velocity X not correctly initialized";
  EXPECT_NEAR(state[SIdx::AngularVelocity::Y], desired_state[SIdx::AngularVelocity::Y], 1e-6) << "Angular Velocity Y not correctly initialized";
  EXPECT_NEAR(state[SIdx::AngularVelocity::Z], desired_state[SIdx::AngularVelocity::Z], 1e-6) << "Angular Velocity Z not correctly initialized";
}

// Test initialization with IMU sensor handled in IMU specific tests

// Test sequential initialization with multiple sensors
TEST_F(FilterInitializationTest, SequentialInitialization) {
  // Create sensors
  auto position_sensor = std::make_shared<sensors::PositionSensorModel>();
  auto velocity_sensor = std::make_shared<sensors::BodyVelocitySensorModel>();

  // Add sensors to filter with explicit sensor types
  size_t position_idx = filter_->AddSensor<SensorType::Position>(position_sensor);
  size_t velocity_idx = filter_->AddSensor<SensorType::BodyVelocity>(velocity_sensor);

  // Create desired state
  StateVector desired_state = StateVector::Zero();
  desired_state.segment<3>(SIdx::Position::X) = Eigen::Vector3d(1.0, 2.0, 3.0);
  desired_state.segment<3>(SIdx::LinearVelocity::X) = Eigen::Vector3d(0.1, 0.2, 0.3);
  desired_state.segment<3>(SIdx::AngularVelocity::X) = Eigen::Vector3d(0.01, 0.02, 0.03);

  // Generate measurements from desired state
  Eigen::Vector3d position_measurement = position_sensor->PredictMeasurement(desired_state);
  Eigen::Matrix<double, 6, 1> velocity_measurement = velocity_sensor->PredictMeasurement(desired_state);

  // Process position measurement first with explicit sensor type
  bool result1 = filter_->ProcessMeasurementByIndex<SensorType::Position>(
      position_idx, position_measurement, 0.0);
  EXPECT_TRUE(result1) << "Failed to process position measurement";

  // Check if filter is initialized after position measurement
  EXPECT_TRUE(filter_->IsInitialized()) << "Filter not initialized after position measurement";

  // Check position values
  StateVector state_after_position = filter_->GetStateEstimate();
  EXPECT_NEAR(state_after_position[SIdx::Position::X], desired_state[SIdx::Position::X], 1e-6);
  EXPECT_NEAR(state_after_position[SIdx::Position::Y], desired_state[SIdx::Position::Y], 1e-6);
  EXPECT_NEAR(state_after_position[SIdx::Position::Z], desired_state[SIdx::Position::Z], 1e-6);

  // Process velocity measurement next with explicit sensor type
  bool result2 = filter_->ProcessMeasurementByIndex<SensorType::BodyVelocity>(
      velocity_idx, velocity_measurement, 0.0);
  EXPECT_TRUE(result2) << "Failed to process velocity measurement";

  // Check all values after both measurements
  StateVector state_after_both = filter_->GetStateEstimate();

  // Position should still be initialized
  EXPECT_NEAR(state_after_both[SIdx::Position::X], desired_state[SIdx::Position::X], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::Position::Y], desired_state[SIdx::Position::Y], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::Position::Z], desired_state[SIdx::Position::Z], 1e-6);

  // Velocity should now be initialized
  EXPECT_NEAR(state_after_both[SIdx::LinearVelocity::X], desired_state[SIdx::LinearVelocity::X], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::LinearVelocity::Y], desired_state[SIdx::LinearVelocity::Y], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::LinearVelocity::Z], desired_state[SIdx::LinearVelocity::Z], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::AngularVelocity::X], desired_state[SIdx::AngularVelocity::X], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::AngularVelocity::Y], desired_state[SIdx::AngularVelocity::Y], 1e-6);
  EXPECT_NEAR(state_after_both[SIdx::AngularVelocity::Z], desired_state[SIdx::AngularVelocity::Z], 1e-6);
}

} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
