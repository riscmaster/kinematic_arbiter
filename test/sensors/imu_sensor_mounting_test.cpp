#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include "kinematic_arbiter/sensors/imu_sensor_model.h"
#include "kinematic_arbiter/core/state_index.h"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using core::StateIndex;

// Add this constant to avoid direct access to the private kGravity in the class
constexpr double kTestGravity = 9.80665;

// Helper class to access protected members for testing
class TestableImuSensorModel : public ImuSensorModel {
public:
  explicit TestableImuSensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      bool calibration_enabled = false,
      double stationary_threshold = 0.01)
    : ImuSensorModel(sensor_pose_in_body_frame) {

    // Use the public setter method instead of direct access to private member
    ImuSensorConfig config;
    config.calibration_enabled = calibration_enabled;
    config.stationary_confidence_threshold = stationary_threshold;
    SetConfig(config);
  }

  // Expose protected measurement_covariance_
  void SetTestCovariance(const MeasurementCovariance& covariance) {
    this->measurement_covariance_ = covariance;
  }
};

class ImuSensorMountingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a default sensor model for testing
    sensor_offset_ = Eigen::Vector3d(0.1, 0.2, 0.3);  // Non-zero offset for better testing
    Eigen::Isometry3d sensor_pose = Eigen::Isometry3d::Identity();
    sensor_pose.translation() = sensor_offset_;

    // Create a small rotation for the sensor mounting
    Eigen::AngleAxisd rotation(0.1, Eigen::Vector3d::UnitZ());
    sensor_pose.rotate(rotation);

    // Configure the sensor model with bias calibration enabled
    sensor_model_ = std::make_unique<TestableImuSensorModel>(
        sensor_pose, true, 0.01);

    // Set a reasonable measurement covariance using our accessor
    Eigen::Matrix<double, 6, 6> meas_cov = Eigen::Matrix<double, 6, 6>::Identity();
    meas_cov.block<3, 3>(0, 0) *= 0.0001;  // Gyro noise (rad/s)²
    meas_cov.block<3, 3>(3, 3) *= 0.01;    // Accel noise (m/s²)²

    // Use our test helper method to set the covariance
    sensor_model_->SetTestCovariance(meas_cov);

    // Create a default state vector with the correct size from StateIndex
    state_ = Eigen::VectorXd::Zero(StateIndex::kFullStateSize);

    // Set quaternion to identity
    state_(StateIndex::Quaternion::W) = 1.0;

    // Default state covariance - low uncertainty
    state_cov_ = Eigen::MatrixXd::Identity(StateIndex::kFullStateSize, StateIndex::kFullStateSize) * 0.01;
  }

  // Helper to create a quaternion for a simple rotation around an axis
  Eigen::Quaterniond QuaternionFromAxisAngle(const Eigen::Vector3d& axis, double angle) {
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis.normalized()));
  }

  // Helper to create a state with specific values
  void SetStateValues(
      const Eigen::Quaterniond& orientation,
      const Eigen::Vector3d& angular_velocity,
      const Eigen::Vector3d& linear_acceleration,
      const Eigen::Vector3d& angular_acceleration = Eigen::Vector3d::Zero()) {

    // Set orientation quaternion
    state_(StateIndex::Quaternion::W) = orientation.w();
    state_(StateIndex::Quaternion::X) = orientation.x();
    state_(StateIndex::Quaternion::Y) = orientation.y();
    state_(StateIndex::Quaternion::Z) = orientation.z();

    // Set angular velocities
    state_.segment<3>(StateIndex::AngularVelocity::X) = angular_velocity;

    // Set linear accelerations
    state_.segment<3>(StateIndex::LinearAcceleration::X) = linear_acceleration;

    // Set angular accelerations
    state_.segment<3>(StateIndex::AngularAcceleration::X) = angular_acceleration;
  }

  std::unique_ptr<TestableImuSensorModel> sensor_model_;
  Eigen::VectorXd state_;
  Eigen::MatrixXd state_cov_;
  Eigen::Vector3d sensor_offset_;
};

/// Test with perfect sensor mounting (aligned with body frame)
TEST_F(ImuSensorMountingTest, PerfectMounting) {
  // Create a sensor model with identity pose (perfect mounting)
  Eigen::Isometry3d identity_pose = Eigen::Isometry3d::Identity();
  auto perfect_sensor = std::make_unique<TestableImuSensorModel>(identity_pose);

  // Set same reasonable measurement covariance
  Eigen::Matrix<double, 6, 6> meas_cov = Eigen::Matrix<double, 6, 6>::Identity();
  meas_cov.block<3, 3>(0, 0) *= 0.0001;  // Gyro noise (rad/s)²
  meas_cov.block<3, 3>(3, 3) *= 0.01;    // Accel noise (m/s²)²
  perfect_sensor->SetTestCovariance(meas_cov);

  // For identity orientation, zero velocity and zero acceleration
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

  SetStateValues(orientation, angular_velocity, linear_acceleration);

  std::cout << "\n==== Perfect Mounting Test ====\n";
  std::cout << "Identity body orientation, identity mounting\n";

  // Predict measurement
  auto measurement = perfect_sensor->PredictMeasurement(state_);
  Eigen::Vector3d gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
  Eigen::Vector3d accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

  std::cout << "Gyro: " << gyro.transpose() << std::endl;
  std::cout << "Accel: " << accel.transpose() << std::endl;
  std::cout << "Accel Magnitude: " << accel.norm() << std::endl;

  // Gyro should be zero
  EXPECT_NEAR(gyro.norm(), 0.0, 1e-6);
  // Accel should be exactly gravity in the z-direction
  EXPECT_NEAR(accel[0], 0.0, 1e-6);
  EXPECT_NEAR(accel[1], 0.0, 1e-6);
  EXPECT_NEAR(accel[2], kTestGravity, 1e-6);

  // Test with 90° rotation around X axis
  std::cout << "\nRotation around X by 90°, identity mounting\n";
  Eigen::Quaterniond rotation_x = QuaternionFromAxisAngle(Eigen::Vector3d::UnitX(), M_PI/2);
  SetStateValues(rotation_x, angular_velocity, linear_acceleration);

  measurement = perfect_sensor->PredictMeasurement(state_);
  gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
  accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

  std::cout << "Gyro: " << gyro.transpose() << std::endl;
  std::cout << "Accel: " << accel.transpose() << std::endl;
  std::cout << "Accel Magnitude: " << accel.norm() << std::endl;

  // With 90° X rotation, gravity should be along Y axis
  EXPECT_NEAR(accel[0], 0.0, 1e-6);
  EXPECT_NEAR(accel[1], kTestGravity, 1e-6);
  EXPECT_NEAR(accel[2], 0.0, 1e-6);

  // Test with 90° rotation around Y axis
  std::cout << "\nRotation around Y by 90°, identity mounting\n";
  Eigen::Quaterniond rotation_y = QuaternionFromAxisAngle(Eigen::Vector3d::UnitY(), M_PI/2);
  SetStateValues(rotation_y, angular_velocity, linear_acceleration);

  measurement = perfect_sensor->PredictMeasurement(state_);
  gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
  accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

  std::cout << "Gyro: " << gyro.transpose() << std::endl;
  std::cout << "Accel: " << accel.transpose() << std::endl;
  std::cout << "Accel Magnitude: " << accel.norm() << std::endl;

  // With 90° Y rotation, gravity should be along negative X axis
  EXPECT_NEAR(accel[0], -kTestGravity, 1e-6);
  EXPECT_NEAR(accel[1], 0.0, 1e-6);
  EXPECT_NEAR(accel[2], 0.0, 1e-6);
}

// Test with different sensor mounting orientations
TEST_F(ImuSensorMountingTest, MountingOrientations) {
  std::cout << "\n==== Different Mounting Orientations ====\n";

  // Create an array of different mounting orientations to test
  std::vector<std::pair<Eigen::Isometry3d, std::string>> mounting_poses;

  // 1. Offset only
  Eigen::Isometry3d offset_only = Eigen::Isometry3d::Identity();
  offset_only.translation() = Eigen::Vector3d(0.1, 0.2, 0.3);
  mounting_poses.push_back({offset_only, "Translation offset only"});

  // 2. 45° rotation around Z
  Eigen::Isometry3d rot_z_45 = Eigen::Isometry3d::Identity();
  rot_z_45.rotate(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()));
  mounting_poses.push_back({rot_z_45, "45° rotation around Z"});

  // 3. 45° rotation around X with offset
  Eigen::Isometry3d rot_x_45_offset = Eigen::Isometry3d::Identity();
  rot_x_45_offset.translation() = Eigen::Vector3d(0.1, 0.2, 0.3);
  rot_x_45_offset.rotate(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitX()));
  mounting_poses.push_back({rot_x_45_offset, "45° rotation around X with offset"});

  // Test each mounting orientation
  for (const auto& [mounting_pose, description] : mounting_poses) {
    std::cout << "\n--- Testing: " << description << " ---\n";

    // Create sensor with this mounting
    auto test_sensor = std::make_unique<TestableImuSensorModel>(mounting_pose);

    // Set same covariance
    Eigen::Matrix<double, 6, 6> meas_cov = Eigen::Matrix<double, 6, 6>::Identity();
    meas_cov.block<3, 3>(0, 0) *= 0.0001;
    meas_cov.block<3, 3>(3, 3) *= 0.01;
    test_sensor->SetTestCovariance(meas_cov);

    // Print the sensor pose for reference
    Eigen::Isometry3d sensor_pose = test_sensor->GetSensorPoseInBodyFrame();
    std::cout << "Sensor Position: " << sensor_pose.translation().transpose() << std::endl;
    Eigen::Quaterniond sensor_quat(sensor_pose.rotation());
    std::cout << "Sensor Orientation (quat): " << sensor_quat.w() << ", "
              << sensor_quat.x() << ", " << sensor_quat.y() << ", "
              << sensor_quat.z() << std::endl;

    // Test with identity body orientation
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

    SetStateValues(orientation, angular_velocity, linear_acceleration);

    std::cout << "Identity body orientation:\n";
    auto measurement = test_sensor->PredictMeasurement(state_);
    Eigen::Vector3d gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
    Eigen::Vector3d accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

    std::cout << "Gyro: " << gyro.transpose() << std::endl;
    std::cout << "Accel: " << accel.transpose() << std::endl;
    std::cout << "Accel Magnitude: " << accel.norm() << std::endl;

    // Test with 90° rotation around X
    Eigen::Quaterniond rotation_x = QuaternionFromAxisAngle(Eigen::Vector3d::UnitX(), M_PI/2);
    SetStateValues(rotation_x, angular_velocity, linear_acceleration);

    std::cout << "\nBody rotated 90° around X:\n";
    measurement = test_sensor->PredictMeasurement(state_);
    gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
    accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

    std::cout << "Gyro: " << gyro.transpose() << std::endl;
    std::cout << "Accel: " << accel.transpose() << std::endl;
    std::cout << "Accel Magnitude: " << accel.norm() << std::endl;

    // Verify that acceleration magnitude is preserved
    EXPECT_NEAR(accel.norm(), kTestGravity, 1e-6);
  }
}

// Test basic measurement prediction with identity orientation
TEST_F(ImuSensorMountingTest, BasicMeasurementPrediction) {
  // Create sensor with identity mounting
  ImuSensorModel sensor;

  // Set angular velocity
  SetStateValues(Eigen::Quaterniond::Identity(),
                Eigen::Vector3d(1.0, 2.0, 3.0),
                Eigen::Vector3d::Zero());

  // Predict measurement
  Eigen::VectorXd measurement = sensor.PredictMeasurement(state_);

  // Extract components
  Eigen::Vector3d gyro = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
  Eigen::Vector3d accel = measurement.segment<3>(ImuSensorModel::MeasurementIndex::AX);

  // Expected: gyro = angular velocity, accel = gravity in body frame (0,0,g)
  Eigen::Vector3d expected_gyro(1.0, 2.0, 3.0);
  Eigen::Vector3d expected_accel(0.0, 0.0, kTestGravity);

  // Check results with relative error
  double gyro_error = (gyro - expected_gyro).norm() / expected_gyro.norm();
  double accel_error = (accel - expected_accel).norm() / expected_accel.norm();

  EXPECT_LT(gyro_error, kRelativeErrorTolerance) << "Gyro prediction error too large";
  EXPECT_LT(accel_error, kRelativeErrorTolerance) << "Accel prediction error too large";

  std::cout << "Identity test: gyro_error=" << gyro_error
            << ", accel_error=" << accel_error << std::endl;
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
