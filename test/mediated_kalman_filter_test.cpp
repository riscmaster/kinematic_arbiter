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
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace ka = kinematic_arbiter;
namespace kasensors = ka::sensors;
namespace kamodels = ka::models;

// Define the correct state indices namespace alias that all tests can use
using SIdx = kinematic_arbiter::core::StateIndex;

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

// Test fixture
class MediatedKalmanFilterTest : public ::testing::Test {
protected:
  // Updated FilterType to use just StateSize and ProcessModel
  using FilterType = kinematic_arbiter::core::MediatedKalmanFilter<19,
    kinematic_arbiter::models::RigidBodyStateModel>;

  // Type aliases matching those in MediatedKalmanFilter
  using StateVector = Eigen::Matrix<double, 19, 1>;
  using StateMatrix = Eigen::Matrix<double, 19, 19>;

  void SetUp() override {
    // Create models
    state_model_ = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();
    position_sensor_model_ = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();
    primary_imu_model_ = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();
    pose_model_ = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();
    body_vel_model_ = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();
    heading_vel_model_ = std::make_shared<kinematic_arbiter::sensors::HeadingVelocitySensorModel>();

    // Create filter with process model
    filter_ = new FilterType(state_model_);

    // Register each sensor and store its index
    position_idx_ = filter_->AddSensor(position_sensor_model_);
    imu_idx_ = filter_->AddSensor(primary_imu_model_);
    pose_idx_ = filter_->AddSensor(pose_model_);
    body_vel_idx_ = filter_->AddSensor(body_vel_model_);
    heading_vel_idx_ = filter_->AddSensor(heading_vel_model_);
  }

  void TearDown() override {
    delete filter_;
  }

  std::shared_ptr<kinematic_arbiter::models::RigidBodyStateModel> state_model_;
  std::shared_ptr<kinematic_arbiter::sensors::PositionSensorModel> position_sensor_model_;
  std::shared_ptr<kinematic_arbiter::sensors::ImuSensorModel> primary_imu_model_;
  std::shared_ptr<kinematic_arbiter::sensors::PoseSensorModel> pose_model_;
  std::shared_ptr<kinematic_arbiter::sensors::BodyVelocitySensorModel> body_vel_model_;
  std::shared_ptr<kinematic_arbiter::sensors::HeadingVelocitySensorModel> heading_vel_model_;

  FilterType* filter_;

  // Store sensor indices for use in tests
  size_t position_idx_;
  size_t imu_idx_;
  size_t pose_idx_;
  size_t body_vel_idx_;
  size_t heading_vel_idx_;
};

// Basic initialization test
TEST_F(MediatedKalmanFilterTest, Initialization) {
  ASSERT_TRUE(filter_);

  // Create a position measurement to initialize the position
  Eigen::Vector3d position(10.0, 20.0, 30.0);

  // Process the measurement using the position sensor index
  EXPECT_TRUE(filter_->ProcessMeasurementByIndex(position_idx_, 0.0, position));

  auto state = filter_->GetStateEstimate();

  // Use the SIdx alias from the fixture class
  EXPECT_NEAR(10.0, state(SIdx::Position::X), 1e-12);
  EXPECT_NEAR(20.0, state(SIdx::Position::Y), 1e-12);
  EXPECT_NEAR(30.0, state(SIdx::Position::Z), 1e-12);
}

// Test prediction and single sensor with a figure-8 trajectory
TEST_F(MediatedKalmanFilterTest, Figure8TrajectoryDetailedAnalysis) {
  ASSERT_TRUE(filter_);

  // Test parameters
  const double duration = 10.0;
  const double time_step = 0.05; // 20Hz
  const int num_steps = static_cast<int>(duration / time_step) + 1;
  constexpr int num_states = 19; // RigidBodyStateModel has 19 states

  // Arrays to store errors for each state component
  std::vector<std::vector<double>> prediction_errors(num_states);
  std::vector<std::vector<double>> body_velocity_errors(num_states);
  std::vector<double> timestamps;

  // Initialize arrays
  for (int i = 0; i < num_states; ++i) {
    prediction_errors[i].reserve(num_steps);
    body_velocity_errors[i].reserve(num_steps);
  }
  timestamps.reserve(num_steps);

  // Create a filter for prediction-only run
  auto prediction_filter = std::make_shared<FilterType>(state_model_);

  // Run 1: Pure prediction
  {
    // Get initial true state
    StateVector initial_state = kinematic_arbiter::testing::Figure8Trajectory(0.0);

    // Initialize filter with true state
    prediction_filter->SetStateEstimate(initial_state);

    // Set reasonable initial covariance
    auto covariance = prediction_filter->GetStateCovariance();
    for (int i = 0; i < covariance.rows(); i++) {
      covariance(i,i) = 0.1;  // Smaller initial uncertainty since we know the true state
    }
    prediction_filter->SetStateCovariance(covariance);

    for (double t = 0.0; t <= duration; t += time_step) {
      // Store timestamp
      timestamps.push_back(t);

      // Get true state from figure-8 trajectory
      auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

      // Predict to current time (except for initial time)
      if (t > 0) {
        // Use the new PredictNewReference method for dead reckoning
        prediction_filter->PredictNewReference(t);
      }

      // Get current state estimate
      auto state_estimate = prediction_filter->GetStateEstimate();

      // Store errors for each state component
      for (int i = 0; i < num_states; ++i) {
        double error = state_estimate(i) - true_state(i);
        // Handle quaternion sign ambiguity
        if (i >= SIdx::Quaternion::W && i <= SIdx::Quaternion::Z) {
          // If dot product is negative, one quaternion is negative of the other
          double dot_product = state_estimate(SIdx::Quaternion::W) * true_state(SIdx::Quaternion::W) +
                              state_estimate(SIdx::Quaternion::X) * true_state(SIdx::Quaternion::X) +
                              state_estimate(SIdx::Quaternion::Y) * true_state(SIdx::Quaternion::Y) +
                              state_estimate(SIdx::Quaternion::Z) * true_state(SIdx::Quaternion::Z);
          if (dot_product < 0) {
            error = -state_estimate(i) - true_state(i);
          }
        }
        prediction_errors[i].push_back(error);
      }
    }
  }

  // Run 2: Body velocity sensor
  {
    // Create a filter with just the process model
    auto velocity_filter = std::make_shared<FilterType>(state_model_);

    // Add the body velocity sensor
    size_t body_vel_idx = velocity_filter->AddSensor(body_vel_model_);

    // Reset filter with perfect initial state
    StateVector initial_state = kinematic_arbiter::testing::Figure8Trajectory(0.0);

    // Initialize filter with true state
    velocity_filter->SetStateEstimate(initial_state);

    // Set reasonable initial covariance
    auto covariance = velocity_filter->GetStateCovariance();
    for (int i = 0; i < covariance.rows(); i++) {
      covariance(i,i) = 0.1;  // Smaller initial uncertainty
    }
    velocity_filter->SetStateCovariance(covariance);

    for (double t = 0.0; t <= duration; t += time_step) {
      // Get true state from figure-8 trajectory
      auto true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

      // Predict to current time (except for initial time)
      if (t > 0) {
        velocity_filter->PredictNewReference(t);
      }

      // Get body velocity sensor and generate perfect measurement
      auto body_vel_sensor = velocity_filter->GetSensorByIndex<kinematic_arbiter::sensors::BodyVelocitySensorModel>(body_vel_idx);
      auto true_measurement = body_vel_sensor->PredictMeasurement(true_state);

      // Process measurement using body velocity sensor index
      velocity_filter->ProcessMeasurementByIndex(body_vel_idx, t, true_measurement);

      // Get current state estimate
      auto state_estimate = velocity_filter->GetStateEstimate();

      // Store errors for each state component
      for (int i = 0; i < num_states; ++i) {
        double error = state_estimate(i) - true_state(i);
        // Handle quaternion sign ambiguity
        if (i >= SIdx::Quaternion::W && i <= SIdx::Quaternion::Z) {
          // If dot product is negative, one quaternion is negative of the other
          double dot_product = state_estimate(SIdx::Quaternion::W) * true_state(SIdx::Quaternion::W) +
                              state_estimate(SIdx::Quaternion::X) * true_state(SIdx::Quaternion::X) +
                              state_estimate(SIdx::Quaternion::Y) * true_state(SIdx::Quaternion::Y) +
                              state_estimate(SIdx::Quaternion::Z) * true_state(SIdx::Quaternion::Z);
          if (dot_product < 0) {
            error = -state_estimate(i) - true_state(i);
          }
        }
        body_velocity_errors[i].push_back(error);
      }
    }
  }

  // Function to compute statistics for each state component
  auto compute_stats = [this, &timestamps](const std::vector<std::vector<double>>& errors, const std::string& run_name) {
    std::cout << "\n===== Statistics for " << run_name << " =====\n";

    for (size_t state_idx = 0; state_idx < errors.size(); ++state_idx) {
      const auto& state_errors = errors[state_idx];

      double max_error = 0.0;
      double max_error_time = 0.0;
      double sum = 0.0;
      double sum_squared = 0.0;

      for (size_t i = 0; i < state_errors.size(); ++i) {
        double error = std::abs(state_errors[i]);
        if (error > max_error) {
          max_error = error;
          max_error_time = timestamps[i];
        }
        sum += state_errors[i];
        sum_squared += state_errors[i] * state_errors[i];
      }

      double mean_error = sum / state_errors.size();
      double variance = (sum_squared / state_errors.size()) - (mean_error * mean_error);
      double std_dev = std::sqrt(variance);
      double three_sigma = 3 * std_dev;

      std::string state_name = "State[" + std::to_string(state_idx) + "]";

      if (state_idx >= SIdx::Position::X && state_idx <= SIdx::Position::Z) {
        state_name = "Position." + std::string(1, 'X' + (state_idx - SIdx::Position::X));
      } else if (state_idx >= SIdx::Quaternion::W && state_idx <= SIdx::Quaternion::Z) {
        char component;
        if (state_idx == SIdx::Quaternion::W) {
          component = 'W';
        } else {
          component = 'X' + (state_idx - SIdx::Quaternion::X);
        }
        state_name = "Quaternion." + std::string(1, component);
      } else if (state_idx >= SIdx::LinearVelocity::X && state_idx <= SIdx::LinearVelocity::Z) {
        state_name = "LinVel." + std::string(1, 'X' + (state_idx - SIdx::LinearVelocity::X));
      } else if (state_idx >= SIdx::AngularVelocity::X && state_idx <= SIdx::AngularVelocity::Z) {
        state_name = "AngVel." + std::string(1, 'X' + (state_idx - SIdx::AngularVelocity::X));
      } else if (state_idx >= SIdx::LinearAcceleration::X && state_idx <= SIdx::LinearAcceleration::Z) {
        state_name = "LinAcc." + std::string(1, 'X' + (state_idx - SIdx::LinearAcceleration::X));
      } else if (state_idx >= SIdx::AngularAcceleration::X && state_idx <= SIdx::AngularAcceleration::Z) {
        state_name = "AngAcc." + std::string(1, 'X' + (state_idx - SIdx::AngularAcceleration::X));
      }

      std::cout << std::setw(20) << state_name << ": "
                << "max=" << std::setw(10) << max_error << " (t=" << max_error_time << "s), "
                << "mean=" << std::setw(10) << mean_error << ", "
                << "3σ=" << std::setw(10) << three_sigma << "\n";
    }
  };

  // Compute and print statistics for both runs
  compute_stats(prediction_errors, "Pure Prediction");
  compute_stats(body_velocity_errors, "Body Velocity Sensor");

  // Verify that key states have reasonable errors
  double max_position_error_prediction = 0.0;
  double max_position_error_body_vel = 0.0;

  for (int i = SIdx::Position::X; i <= SIdx::Position::Z; ++i) {
    for (double e : prediction_errors[i]) {
      max_position_error_prediction = std::max(max_position_error_prediction, std::abs(e));
    }
    for (double e : body_velocity_errors[i]) {
      max_position_error_body_vel = std::max(max_position_error_body_vel, std::abs(e));
    }
  }

  // Simple verification that body velocity measurements improve prediction
  EXPECT_LE(max_position_error_body_vel, max_position_error_prediction)
    << "Body velocity measurements should improve position tracking";
}
