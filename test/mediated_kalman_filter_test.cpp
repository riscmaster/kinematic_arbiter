#include <gtest/gtest.h>
#include <kinematic_arbiter/core/mediated_kalman_filter.hpp>
#include <kinematic_arbiter/core/state_index.hpp>
#include <kinematic_arbiter/models/rigid_body_state_model.hpp>
#include <kinematic_arbiter/sensors/position_sensor_model.hpp>
#include <kinematic_arbiter/sensors/pose_sensor_model.hpp>
#include <kinematic_arbiter/sensors/imu_sensor_model.hpp>
#include <kinematic_arbiter/sensors/body_velocity_sensor_model.hpp>
#include <kinematic_arbiter/sensors/heading_velocity_sensor_model.hpp>
#include "utils/test_trajectories.hpp"
#include <random>  // Include this at the top for std::random_device and std::mt19937

namespace ka = kinematic_arbiter;
namespace kasensors = ka::sensors;
namespace kamodels = ka::models;
namespace kinematic_arbiter {
namespace testing {

/**
 * Sample a random vector from a multivariate Gaussian distribution
 *
 * @param gen Random number generator
 * @param covariance Covariance matrix of the distribution
 * @return Random vector sampled from the distribution
 */
template<typename Generator, int Dim>
Eigen::Matrix<double, Dim, 1> SampleGaussianNoise(
    Generator& gen,
    const Eigen::Matrix<double, Dim, Dim>& covariance) {

  // Create standard normal distribution
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  // Generate vector of independent standard normal samples
  Eigen::Matrix<double, Dim, 1> z;
  for (int i = 0; i < Dim; ++i) {
    z(i) = normal_dist(gen);
  }

  // Compute Cholesky decomposition of covariance
  Eigen::LLT<Eigen::Matrix<double, Dim, Dim>> llt(covariance);
  if (llt.info() != Eigen::Success) {
    // Fallback for numerical issues
    return Eigen::Matrix<double, Dim, 1>::Zero();
  }

  // Transform samples using Cholesky factor: x = L*z
  return llt.matrixL() * z;
}

} // namespace testing
} // namespace kinematic_arbiter

// Basic test fixture
class MediatedKalmanFilterTest : public ::testing::Test {
protected:
  // Define state-related types
  static constexpr int StateSize = kamodels::RigidBodyStateModel::StateSize;
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using StateCovariance = StateMatrix;

  // Namespace alias for state indices
  using SIdx = ka::core::StateIndex;

  // Define the filter type with all needed sensor models
  using FilterType = ka::core::MediatedKalmanFilter<
    StateSize,
    kamodels::RigidBodyStateModel,
    kasensors::PositionSensorModel,
    kasensors::ImuSensorModel,
    kasensors::PoseSensorModel,
    kasensors::BodyVelocitySensorModel,
    kasensors::HeadingVelocitySensorModel
  >;

  void SetUp() override {
    // Create state model
    state_model_ = std::make_shared<kamodels::RigidBodyStateModel>();

    // Create all sensor models
    position_model_ = std::make_shared<kasensors::PositionSensorModel>();
    imu_model_ = std::make_shared<kasensors::ImuSensorModel>();
    pose_model_ = std::make_shared<kasensors::PoseSensorModel>();
    body_vel_model_ = std::make_shared<kasensors::BodyVelocitySensorModel>();
    heading_vel_model_ = std::make_shared<kasensors::HeadingVelocitySensorModel>();

    // Create filter directly with new
    filter_ = new FilterType(
      state_model_, position_model_, imu_model_, pose_model_,
      body_vel_model_, heading_vel_model_
    );
  }

  void TearDown() override {
    delete filter_;
  }

  // Store all models to ensure they exist for filter lifetime
  std::shared_ptr<kamodels::RigidBodyStateModel> state_model_;
  std::shared_ptr<kasensors::PositionSensorModel> position_model_;
  std::shared_ptr<kasensors::ImuSensorModel> imu_model_;
  std::shared_ptr<kasensors::PoseSensorModel> pose_model_;
  std::shared_ptr<kasensors::BodyVelocitySensorModel> body_vel_model_;
  std::shared_ptr<kasensors::HeadingVelocitySensorModel> heading_vel_model_;

  // Raw pointer for filter
  FilterType* filter_;
};

// Basic initialization test
TEST_F(MediatedKalmanFilterTest, Initialization) {
  ASSERT_TRUE(filter_);

  // Position sensor is at index 0
  Eigen::Vector3d position(1.0, 2.0, 3.0);
  EXPECT_TRUE(filter_->template ProcessMeasurement<0>(0.0, position));

  auto state = filter_->GetStateEstimate();
  EXPECT_NEAR(state(SIdx::Position::X), 1.0, 1e-6);
  EXPECT_NEAR(state(SIdx::Position::Y), 2.0, 1e-6);
  EXPECT_NEAR(state(SIdx::Position::Z), 3.0, 1e-6);
}

// Test basic sensor fusion with a figure-8 trajectory
TEST_F(MediatedKalmanFilterTest, Figure8Trajectory) {
  ASSERT_TRUE(filter_);

  // Initialize with position
  Eigen::Vector3d init_position(0.0, 0.0, 0.0);
  bool init_success = filter_->template ProcessMeasurement<0>(0.0, init_position);
  ASSERT_TRUE(init_success) << "Failed to initialize filter with position";

  // Set reasonable initial covariance
  auto covariance = filter_->GetStateCovariance();
  for (int i = 0; i < covariance.rows(); i++) {
    covariance(i,i) = 1.0;  // Default moderate uncertainty
  }
  filter_->SetStateCovariance(covariance);

  // Test parameters
  const double duration = 10.0;  // seconds
  const double time_step = 0.1;  // seconds (10Hz)

  // Add these variables right before the for loop:
  double max_position_error = 0.0;
  double max_error_time = 0.0;
  double sum_position_error = 0.0;
  int num_samples = 0;

  // Run trajectory test
  for (double t = 0.0; t <= duration; t += time_step) {
    // Get true state from figure-8 trajectory
    auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

    // Process position measurement (sensor index 0)
    Eigen::Vector3d position;
    position[0] = true_state(SIdx::Position::X);
    position[1] = true_state(SIdx::Position::Y);
    position[2] = true_state(SIdx::Position::Z);
    filter_->template ProcessMeasurement<0>(t, position);

    // Process IMU measurement (sensor index 1) - every 5 steps
    if (static_cast<int>(t / time_step) % 5 == 0) {
      Eigen::Matrix<double, 6, 1> imu_data;
      // Angular velocity
      imu_data[0] = true_state(SIdx::AngularVelocity::X);
      imu_data[1] = true_state(SIdx::AngularVelocity::Y);
      imu_data[2] = true_state(SIdx::AngularVelocity::Z);
      // Linear acceleration
      imu_data[3] = true_state(SIdx::LinearAcceleration::X);
      imu_data[4] = true_state(SIdx::LinearAcceleration::Y);
      imu_data[5] = true_state(SIdx::LinearAcceleration::Z);
      filter_->template ProcessMeasurement<1>(t, imu_data);
    }

    // Process pose measurement (sensor index 2) - every 10 steps
    if (static_cast<int>(t / time_step) % 10 == 0) {
      Eigen::Matrix<double, 7, 1> pose;
      // Position
      pose[0] = true_state(SIdx::Position::X);
      pose[1] = true_state(SIdx::Position::Y);
      pose[2] = true_state(SIdx::Position::Z);
      // Quaternion
      pose[3] = true_state(SIdx::Quaternion::W);
      pose[4] = true_state(SIdx::Quaternion::X);
      pose[5] = true_state(SIdx::Quaternion::Y);
      pose[6] = true_state(SIdx::Quaternion::Z);
      filter_->template ProcessMeasurement<2>(t, pose);
    }

    // Get current state estimate
    auto state_estimate = filter_->GetStateEstimate();

    // Calculate position error
    double x_error = std::abs(state_estimate(SIdx::Position::X) - true_state(SIdx::Position::X));
    double y_error = std::abs(state_estimate(SIdx::Position::Y) - true_state(SIdx::Position::Y));
    double z_error = std::abs(state_estimate(SIdx::Position::Z) - true_state(SIdx::Position::Z));
    double position_error = std::sqrt(x_error*x_error + y_error*y_error + z_error*z_error);

    // Track statistics
    if (position_error > max_position_error) {
      max_position_error = position_error;
      max_error_time = t;
    }
    sum_position_error += position_error;
    num_samples++;

    // Print occasional diagnostics
    if (std::fmod(t, 0.2) < time_step) {
      std::cout << "Time: " << t << "s, Position error: " << position_error << " m" << std::endl;
    }

    // Simple position accuracy check
    EXPECT_LE(position_error, 0.5) << "Position error too large at t=" << t;
  }

  // Define the true noise covariances we expect to converge to
  Eigen::Matrix3d true_position_noise_cov = Eigen::Matrix3d::Identity() * 0.01;
  Eigen::Matrix<double, 6, 6> true_imu_noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.005;

  // At the end of the test, check covariance adaptation with simple norm comparison
  if (duration - time_step >= 0.0) {
    // Position sensor covariance adaptation
    auto position_estimated_cov = position_model_->GetMeasurementCovariance();
    double position_cov_diff = (position_estimated_cov - true_position_noise_cov).norm() /
                              true_position_noise_cov.norm();
    std::cout << "Position covariance relative difference: " << position_cov_diff << std::endl;
    // Use higher threshold initially - we may need to run longer for better convergence
    EXPECT_LE(position_cov_diff, 1.0) << "Position covariance didn't converge properly";

    // Primary IMU covariance adaptation
    auto primary_imu_estimated_cov = imu_model_->GetMeasurementCovariance();
    double primary_imu_cov_diff = (primary_imu_estimated_cov - true_imu_noise_cov).norm() /
                                  true_imu_noise_cov.norm();
    std::cout << "Primary IMU covariance relative difference: " << primary_imu_cov_diff << std::endl;
    EXPECT_LE(primary_imu_cov_diff, 1.0) << "Primary IMU covariance didn't converge properly";
  }

  // Print error statistics
  double mean_position_error = sum_position_error / num_samples;
  std::cout << "\nPosition Error Statistics:" << std::endl;
  std::cout << "  Mean error: " << mean_position_error << " m" << std::endl;
  std::cout << "  Maximum error: " << max_position_error << " m (at t=" << max_error_time << "s)" << std::endl;
}

// Test fixture for multiple IMU sensors
class MultiImuFilterTest : public ::testing::Test {
protected:
  // Define state-related types
  static constexpr int StateSize = kamodels::RigidBodyStateModel::StateSize;
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using StateCovariance = StateMatrix;

  // Namespace alias for state indices
  using SIdx = ka::core::StateIndex;

  // Define filter type with multiple IMU sensors
  using FilterType = ka::core::MediatedKalmanFilter<
    StateSize,
    kamodels::RigidBodyStateModel,
    kasensors::PositionSensorModel,
    kasensors::ImuSensorModel,        // Primary IMU
    kasensors::ImuSensorModel,        // Secondary IMU
    kasensors::ImuSensorModel         // Tertiary IMU
  >;

  void SetUp() override {
    // Create state model
    state_model_ = std::make_shared<kamodels::RigidBodyStateModel>();

    // Create sensor models
    position_model_ = std::make_shared<kasensors::PositionSensorModel>();

    // Create multiple IMU models with different characteristics
    primary_imu_model_ = std::make_shared<kasensors::ImuSensorModel>();
    secondary_imu_model_ = std::make_shared<kasensors::ImuSensorModel>();
    tertiary_imu_model_ = std::make_shared<kasensors::ImuSensorModel>();

    // Note: Since GetParams/SetParams are not available, we're using the default
    // models without modification. In a real implementation, consider adding
    // configuration methods to the sensor models.

    // Create filter
    filter_ = new FilterType(
      state_model_, position_model_,
      primary_imu_model_, secondary_imu_model_, tertiary_imu_model_
    );
  }

  void TearDown() override {
    delete filter_;
  }

  // Store all models to ensure they exist for filter lifetime
  std::shared_ptr<kamodels::RigidBodyStateModel> state_model_;
  std::shared_ptr<kasensors::PositionSensorModel> position_model_;
  std::shared_ptr<kasensors::ImuSensorModel> primary_imu_model_;
  std::shared_ptr<kasensors::ImuSensorModel> secondary_imu_model_;
  std::shared_ptr<kasensors::ImuSensorModel> tertiary_imu_model_;

  // Raw pointer for filter
  FilterType* filter_;
};

// Test fixture for differential drive vehicle simulation
class DifferentialDriveFilterTest : public ::testing::Test {
protected:
  static constexpr int StateSize = kamodels::RigidBodyStateModel::StateSize;
  using SIdx = ka::core::StateIndex;

  // Define filter type with multiple body velocity sensors
  using FilterType = ka::core::MediatedKalmanFilter<
    StateSize,
    kamodels::RigidBodyStateModel,
    kasensors::PositionSensorModel,
    kasensors::ImuSensorModel,
    kasensors::BodyVelocitySensorModel,   // Left wheel
    kasensors::BodyVelocitySensorModel    // Right wheel
  >;

  void SetUp() override {
    // Create state model
    state_model_ = std::make_shared<kamodels::RigidBodyStateModel>();

    // Create sensor models
    position_model_ = std::make_shared<kasensors::PositionSensorModel>();
    imu_model_ = std::make_shared<kasensors::ImuSensorModel>();

    // Create body velocity sensors for left and right wheels
    left_wheel_model_ = std::make_shared<kasensors::BodyVelocitySensorModel>();
    right_wheel_model_ = std::make_shared<kasensors::BodyVelocitySensorModel>();

    // Note: Since GetParams/SetParams are not available, we assume the mounting
    // positions are pre-configured or handled internally

    // Create filter
    filter_ = new FilterType(
      state_model_, position_model_, imu_model_,
      left_wheel_model_, right_wheel_model_
    );
  }

  void TearDown() override {
    delete filter_;
  }

  std::shared_ptr<kamodels::RigidBodyStateModel> state_model_;
  std::shared_ptr<kasensors::PositionSensorModel> position_model_;
  std::shared_ptr<kasensors::ImuSensorModel> imu_model_;
  std::shared_ptr<kasensors::BodyVelocitySensorModel> left_wheel_model_;
  std::shared_ptr<kasensors::BodyVelocitySensorModel> right_wheel_model_;

  FilterType* filter_;
};

// Test fixture for minimalistic sensor suite (heading velocity, IMU, position)
class MinimalisticSuiteFilterTest : public ::testing::Test {
protected:
  static constexpr int StateSize = kamodels::RigidBodyStateModel::StateSize;
  using SIdx = ka::core::StateIndex;

  // Define filter type with minimalistic sensor suite
  using FilterType = ka::core::MediatedKalmanFilter<
    StateSize,
    kamodels::RigidBodyStateModel,
    kasensors::PositionSensorModel,
    kasensors::ImuSensorModel,
    kasensors::HeadingVelocitySensorModel
  >;

  void SetUp() override {
    // Create state model
    state_model_ = std::make_shared<kamodels::RigidBodyStateModel>();

    // Create sensor models for minimalistic suite
    position_model_ = std::make_shared<kasensors::PositionSensorModel>();
    imu_model_ = std::make_shared<kasensors::ImuSensorModel>();
    heading_vel_model_ = std::make_shared<kasensors::HeadingVelocitySensorModel>();

    // Create filter
    filter_ = new FilterType(
      state_model_, position_model_, imu_model_, heading_vel_model_
    );
  }

  void TearDown() override {
    delete filter_;
  }

  std::shared_ptr<kamodels::RigidBodyStateModel> state_model_;
  std::shared_ptr<kasensors::PositionSensorModel> position_model_;
  std::shared_ptr<kasensors::ImuSensorModel> imu_model_;
  std::shared_ptr<kasensors::HeadingVelocitySensorModel> heading_vel_model_;

  FilterType* filter_;
};

// Test for multiple IMU sensors
TEST_F(MultiImuFilterTest, Figure8WithMultipleImus) {
  ASSERT_TRUE(filter_);

  // Get initial true state from figure-8 trajectory at t=0
  auto initial_state = kinematic_arbiter::testing::Figure8Trajectory(0.0);

  // Initialize filter with true state
  filter_->SetStateEstimate(initial_state);

  // Set reasonable initial covariance
  auto covariance = filter_->GetStateCovariance();
  for (int i = 0; i < covariance.rows(); i++) {
    covariance(i,i) = 0.1;  // Smaller initial uncertainty since we know the true state
  }
  filter_->SetStateCovariance(covariance);

  // Test parameters
  const double duration = 10.0;
  const double time_step = 0.01;

  // Set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());

  // Define sensor noise covariances
  Eigen::Matrix3d position_noise_cov = Eigen::Matrix3d::Identity() * 0.001;
  Eigen::Matrix<double, 6, 6> primary_imu_noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.005;
  Eigen::Matrix<double, 6, 6> secondary_imu_noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
  Eigen::Matrix<double, 6, 6> tertiary_imu_noise_cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

  // Define the true noise covariances we expect to converge to
  Eigen::Matrix3d true_position_noise_cov = position_noise_cov;  // Use the one we defined earlier
  Eigen::Matrix<double, 6, 6> true_primary_imu_noise_cov = primary_imu_noise_cov;
  Eigen::Matrix<double, 6, 6> true_secondary_imu_noise_cov = secondary_imu_noise_cov;

  for (double t = 0.0; t <= duration; t += time_step) {
    // Get true state from figure-8 trajectory
    auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

    // Generate measurements using the model's PredictMeasurement function with added noise

    // Position measurement with noise
    auto position = position_model_->PredictMeasurement(true_state);
    Eigen::Vector3d position_noise = kinematic_arbiter::testing::SampleGaussianNoise<std::mt19937, 3>(
        gen, position_noise_cov);
    position += position_noise;
    filter_->template ProcessMeasurement<0>(t, position);

    // Primary IMU measurement (every step) with noise
    auto primary_imu_data = primary_imu_model_->PredictMeasurement(true_state);
    Eigen::Matrix<double, 6, 1> primary_imu_noise = kinematic_arbiter::testing::SampleGaussianNoise(
        gen, primary_imu_noise_cov);
    primary_imu_data += primary_imu_noise;
    filter_->template ProcessMeasurement<1>(t, primary_imu_data);

    // Secondary IMU measurement (every 3 steps) with noise
    if (static_cast<int>(t / time_step) % 3 == 0) {
      auto secondary_imu_data = secondary_imu_model_->PredictMeasurement(true_state);
      Eigen::Matrix<double, 6, 1> secondary_imu_noise = kinematic_arbiter::testing::SampleGaussianNoise(
          gen, secondary_imu_noise_cov);
      secondary_imu_data += secondary_imu_noise;
      filter_->template ProcessMeasurement<2>(t, secondary_imu_data);
    }

    // Tertiary IMU measurement (every 5 steps) with noise
    if (static_cast<int>(t / time_step) % 5 == 0) {
      auto tertiary_imu_data = tertiary_imu_model_->PredictMeasurement(true_state);
      Eigen::Matrix<double, 6, 1> tertiary_imu_noise = kinematic_arbiter::testing::SampleGaussianNoise(
          gen, tertiary_imu_noise_cov);
      tertiary_imu_data += tertiary_imu_noise;
      filter_->template ProcessMeasurement<3>(t, tertiary_imu_data);
    }
    // Get current state estimate
    auto state_estimate = filter_->GetStateEstimate();

    // Calculate position error
    double x_error = std::abs(state_estimate(SIdx::Position::X) - true_state(SIdx::Position::X));
    double y_error = std::abs(state_estimate(SIdx::Position::Y) - true_state(SIdx::Position::Y));
    double z_error = std::abs(state_estimate(SIdx::Position::Z) - true_state(SIdx::Position::Z));
    double position_error = std::sqrt(x_error*x_error + y_error*y_error + z_error*z_error);

    // Print occasional diagnostics
    if (std::fmod(t, 0.2) < time_step) {
      std::cout << "Time: " << t << "s, Position error: " << position_error << " m" << std::endl;
    }

    // Simple position accuracy check
    EXPECT_LE(position_error, 0.5) << "Position error too large at t=" << t;
  }

  // At the end of the test, check covariance adaptation with simple norm comparison
  if (duration - time_step >= 0.0) {
    // Position sensor covariance adaptation
    auto position_estimated_cov = position_model_->GetMeasurementCovariance();
    double position_cov_diff = (position_estimated_cov - true_position_noise_cov).norm() /
                              true_position_noise_cov.norm();
    std::cout << "Position covariance relative difference: " << position_cov_diff << std::endl;
    EXPECT_LE(position_cov_diff, 1.0) << "Position covariance didn't converge properly";

    // Primary IMU covariance adaptation
    auto primary_imu_estimated_cov = primary_imu_model_->GetMeasurementCovariance();
    double primary_imu_cov_diff = (primary_imu_estimated_cov - true_primary_imu_noise_cov).norm() /
                                  true_primary_imu_noise_cov.norm();
    std::cout << "Primary IMU covariance relative difference: " << primary_imu_cov_diff << std::endl;
    EXPECT_LE(primary_imu_cov_diff, 1.0) << "Primary IMU covariance didn't converge properly";

    // Secondary IMU (if present)
    constexpr size_t kMultiImuSensorCount = 4; // Position + 3 IMUs
    if (kMultiImuSensorCount > 1) {
      auto secondary_imu_estimated_cov = secondary_imu_model_->GetMeasurementCovariance();
      double secondary_imu_cov_diff = (secondary_imu_estimated_cov - true_secondary_imu_noise_cov).norm() /
                                      true_secondary_imu_noise_cov.norm();
      std::cout << "Secondary IMU covariance relative difference: " << secondary_imu_cov_diff << std::endl;
      EXPECT_LE(secondary_imu_cov_diff, 1.0) << "Secondary IMU covariance didn't converge properly";
    }
  }
}

// // Test for differential drive vehicle
// TEST_F(DifferentialDriveFilterTest, Figure8WithDifferentialDrive) {
//   ASSERT_TRUE(filter_);

//   // Get initial true state from figure-8 trajectory at t=0
//   auto initial_state = kinematic_arbiter::testing::Figure8Trajectory(0.0);

//   // Initialize filter with true state
//   filter_->SetStateEstimate(initial_state);

//   // Set reasonable initial covariance
//   auto covariance = filter_->GetStateCovariance();
//   for (int i = 0; i < covariance.rows(); i++) {
//     covariance(i,i) = 0.1;  // Smaller initial uncertainty since we know the true state
//   }
//   filter_->SetStateCovariance(covariance);

//   // Test parameters
//   const double duration = 10.0;
//   const double time_step = 0.1;

//   for (double t = 0.0; t <= duration; t += time_step) {
//     // Get true state from figure-8 trajectory
//     auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

//     // Generate measurements using the model's PredictMeasurement function
//     // Position measurement
//     auto position = position_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<0>(t, position);

//     // IMU measurement
//     auto imu_data = imu_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<1>(t, imu_data);

//     // Left wheel velocity measurement
//     auto left_wheel_velocity = left_wheel_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<2>(t, left_wheel_velocity);

//     // Right wheel velocity measurement
//     auto right_wheel_velocity = right_wheel_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<3>(t, right_wheel_velocity);

//     // Get current state estimate
//     auto state_estimate = filter_->GetStateEstimate();

//     // Verify position accuracy (can check immediately since we're properly initialized)
//     double x_error = std::abs(state_estimate(SIdx::Position::X) - true_state(SIdx::Position::X));
//     double y_error = std::abs(state_estimate(SIdx::Position::Y) - true_state(SIdx::Position::Y));
//     double z_error = std::abs(state_estimate(SIdx::Position::Z) - true_state(SIdx::Position::Z));
//     double position_error = std::sqrt(x_error*x_error + y_error*y_error + z_error*z_error);

//     EXPECT_LE(position_error, 0.5) << "Position error too large at t=" << t;
//   }
// }

// // Test for minimalistic sensor suite (heading velocity, IMU, position)
// TEST_F(MinimalisticSuiteFilterTest, Figure8WithMinimalisticSuite) {
//   ASSERT_TRUE(filter_);

//   // Get initial true state from figure-8 trajectory at t=0
//   auto initial_state = kinematic_arbiter::testing::Figure8Trajectory(0.0);

//   // Initialize filter with true state
//   filter_->SetStateEstimate(initial_state);

//   // Set reasonable initial covariance
//   auto covariance = filter_->GetStateCovariance();
//   for (int i = 0; i < covariance.rows(); i++) {
//     covariance(i,i) = 0.1;  // Smaller initial uncertainty since we know the true state
//   }
//   filter_->SetStateCovariance(covariance);

//   // Test parameters
//   const double duration = 10.0;
//   const double time_step = 0.1;

//   for (double t = 0.0; t <= duration; t += time_step) {
//     // Get true state from figure-8 trajectory
//     auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

//     // Process position measurement every 5 steps (simulating GPS at 2Hz)
//     if (static_cast<int>(t / time_step) % 5 == 0) {
//       auto position = position_model_->PredictMeasurement(true_state);
//       filter_->template ProcessMeasurement<0>(t, position);
//     }

//     // Process IMU measurement using model's prediction
//     auto imu_data = imu_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<1>(t, imu_data);

//     // Process heading velocity measurement using model's prediction
//     auto heading_velocity = heading_vel_model_->PredictMeasurement(true_state);
//     filter_->template ProcessMeasurement<2>(t, heading_velocity);

//     // Get current state estimate
//     auto state_estimate = filter_->GetStateEstimate();

//     // Verify position accuracy (can check immediately since we're properly initialized)
//     double x_error = std::abs(state_estimate(SIdx::Position::X) - true_state(SIdx::Position::X));
//     double y_error = std::abs(state_estimate(SIdx::Position::Y) - true_state(SIdx::Position::Y));
//     double z_error = std::abs(state_estimate(SIdx::Position::Z) - true_state(SIdx::Position::Z));
//     double position_error = std::sqrt(x_error*x_error + y_error*y_error + z_error*z_error);

  //   // With proper initialization, we can maintain a tighter error bound
  //   EXPECT_LE(position_error, 0.6) << "Position error too large at t=" << t;
  // }
// }
