#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "test/utils/test_trajectories.hpp"

// Use the correct state index type
using SIdx = kinematic_arbiter::core::StateIndex;

namespace kinematic_arbiter {

// Test fixture
class MediatedKalmanFilterTest : public ::testing::Test {
protected:
  // FilterType definition
  using FilterType = kinematic_arbiter::core::MediatedKalmanFilter<19,
    kinematic_arbiter::models::RigidBodyStateModel>;

  // Type aliases
  using StateVector = Eigen::Matrix<double, 19, 1>;
  using StateMatrix = Eigen::Matrix<double, 19, 19>;
  using StateFlags = Eigen::Array<bool, 19, 1>;

  void SetUp() override {
    // Create models
    state_model_ = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();
    body_vel_model_ = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();
  }

  std::shared_ptr<kinematic_arbiter::models::RigidBodyStateModel> state_model_;
  std::shared_ptr<kinematic_arbiter::sensors::BodyVelocitySensorModel> body_vel_model_;

  // Add this function inside the class
  void PrintFilterDiagnostics(
      const std::shared_ptr<FilterType>& filter,
      const std::shared_ptr<sensors::BodyVelocitySensorModel>& sensor,
      double t,
      const StateVector& true_state,
      const StateVector& predicted_state,
      const StateMatrix& predicted_cov,
      const StateVector& updated_state,
      const Eigen::Matrix<double, 6, 1>& measurement,
      const core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>::MeasurementAuxData& aux_data,
      double vel_error_predicted,
      double vel_error_updated,
      const std::string& failure_reason = "VELOCITY ERROR WORSE") {

    // Calculate Kalman gain for diagnostics
    Eigen::MatrixXd K = predicted_cov * aux_data.jacobian.transpose() *
                         aux_data.innovation_covariance.inverse();

    // Extract velocity blocks
    auto process_noise = filter->GetProcessModel()->GetProcessNoiseCovariance(1.0);
    Eigen::Matrix3d vel_cov_before = predicted_cov.block<3,3>(core::StateIndex::LinearVelocity::X, core::StateIndex::LinearVelocity::X);
    Eigen::Matrix3d process_noise_vel = process_noise.block<3,3>(core::StateIndex::LinearVelocity::X, core::StateIndex::LinearVelocity::X);
    Eigen::Matrix3d vel_cov_after = filter->GetStateCovariance().block<3,3>(core::StateIndex::LinearVelocity::X, core::StateIndex::LinearVelocity::X);

    std::cerr << "\n========= FILTER TEST FAILURE at t=" << t << ": " << failure_reason << " =========\n\n"
              << "STATE VECTORS (3D velocity components):\n"
              << "-----------------------------------------\n"
              << "True velocity:      " << true_state.segment<3>(core::StateIndex::LinearVelocity::X).transpose() << "\n"
              << "Predicted velocity: " << predicted_state.segment<3>(core::StateIndex::LinearVelocity::X).transpose() << "\n"
              << "Updated velocity:   " << updated_state.segment<3>(core::StateIndex::LinearVelocity::X).transpose() << "\n\n"

              << "ERRORS:\n"
              << "-----------------------------------------\n"
              << "Prediction error:   " << vel_error_predicted << "\n"
              << "Update error:       " << vel_error_updated;

    // Only show (WORSE!) if the velocity error actually got worse
    if (vel_error_updated > vel_error_predicted) {
      std::cerr << " (WORSE!)";
    }
    std::cerr << "\n\n";

    std::cerr << "MEASUREMENTS (3D):\n"
              << "-----------------------------------------\n"
              << "Actual measurement:     " << measurement.transpose() << "\n"
              << "Predicted measurement:  " << sensor->PredictMeasurement(predicted_state).transpose() << "\n"
              << "Innovation/residual:    " << aux_data.innovation.segment(0, 3).transpose() << "\n"
              << "Innovation norm:        " << aux_data.innovation.segment(0, 3).norm() << "\n\n"

              << "JACOBIAN (H matrix, showing velocity rows):\n"
              << "-----------------------------------------\n";

    // Print first 3 rows of jacobian
    for (int i = 0; i < 3 && i < aux_data.jacobian.rows(); i++) {
      for (int j = 0; j < aux_data.jacobian.cols(); j++) {
        std::cerr << aux_data.jacobian(i, j) << " ";
      }
      std::cerr << "\n";
    }
    std::cerr << "\n";

    std::cerr << "PROCESS NOISE (Q, velocity block with dt=1):\n"
              << "-----------------------------------------\n"
              << process_noise_vel << "\n\n"

              << "COVARIANCE MATRICES (3x3 blocks):\n"
              << "-----------------------------------------\n"
              << "State covariance BEFORE (P-, velocity block):\n" << vel_cov_before << "\n\n"
              << "State covariance AFTER (P+, velocity block):\n" << vel_cov_after << "\n\n"
              << "Measurement covariance (R):\n" << sensor->GetMeasurementCovariance() << "\n\n"
              << "Innovation covariance (S = H*P*H' + R):\n";

    // Print innovation covariance (3x3 top-left corner if larger)
    int rows = std::min(3, (int)aux_data.innovation_covariance.rows());
    int cols = std::min(3, (int)aux_data.innovation_covariance.cols());
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        std::cerr << aux_data.innovation_covariance(i, j) << " ";
      }
      std::cerr << "\n";
    }
    std::cerr << "\n\n";

    std::cerr << "KALMAN GAIN (K = P*H'*S^-1, velocity rows):\n"
              << "-----------------------------------------\n";

    // Print Kalman gain (velocity rows only)
    for (int i = core::StateIndex::LinearVelocity::X; i < core::StateIndex::LinearVelocity::X + 3; i++) {
      for (int j = 0; j < std::min(3, (int)K.cols()); j++) {
        std::cerr << K(i, j) << " ";
      }
      std::cerr << "\n";
    }
    std::cerr << "\n\n";

    std::cerr << "KALMAN FILTER EQUATIONS:\n"
              << "-----------------------------------------\n"
              << "Innovation: y = z - h(x)\n"
              << "Kalman gain: K = P-*H'*S^-1\n"
              << "State update: x+ = x- + K*y\n"
              << "Covariance update: P+ = (I - K*H)*P-*((I - K*H)ᵀ) + K*R*Kᵀ\n\n"

              << "NUMERICAL CHECKS:\n"
              << "-----------------------------------------\n"
              << "Innovation covariance invertible: "
              << (aux_data.innovation_covariance.determinant() > 1e-10 ? "YES" : "NO") << "\n"
              << "Innovation covariance determinant: " << aux_data.innovation_covariance.determinant() << "\n"
              << "Condition number (approx): "
              << aux_data.innovation_covariance.norm() * aux_data.innovation_covariance.inverse().norm() << "\n"
              << "State cov trace before: " << vel_cov_before.trace() << "\n"
              << "State cov trace after: " << vel_cov_after.trace() << "\n"
              << "Measurement covariance norm: " << sensor->GetMeasurementCovariance().norm() << "\n"
              << "Measurement covariance trace: " << sensor->GetMeasurementCovariance().trace() << "\n";
  }

  template<typename SensorModel>
  void TestSensorImprovesStateEstimates(std::shared_ptr<SensorModel> sensor) {
      using namespace core;

      // Create filter
      auto filter = std::make_shared<FilterType>(state_model_);
      size_t sensor_idx = filter->AddSensor(sensor);

      // Test parameters
      const double dt = 0.05;
      double t = 0.0;

      // Initialize with true state
      StateVector initial_state = kinematic_arbiter::testing::Figure8Trajectory(t);
      filter->SetStateEstimate(initial_state);

      // Set initial covariance
      StateMatrix covariance = StateMatrix::Identity() * 0.1;
      filter->SetStateCovariance(covariance);

      // Get initializable states from the sensor
      StateFlags initializable_flags = sensor->GetInitializableStates();

      // Function to check for NaNs
      auto hasNaN = [](const auto& mat) -> bool {
          return !(mat.array() == mat.array()).all();
      };

      // Last successful state for diagnostics in error messages
      StateVector last_good_state = initial_state;
      StateMatrix last_good_cov = covariance;

      // Track covariance history for checks
      std::vector<double> measurement_cov_norm_history;

      // Store initial measurement covariance norm
      double initial_measurement_cov_norm = sensor->GetMeasurementCovariance().norm();
      measurement_cov_norm_history.push_back(initial_measurement_cov_norm);

      // Set ceiling based on initial covariance norm
      const double kMeasCovCeiling = initial_measurement_cov_norm * 1.5;
      const double kMeasCovFloor = 0.1; // Floor value based on observed behavior

      // Test for 10 seconds (200 steps at 0.05s per step)
      for (int i = 0; i < 200; i++) {
          t += dt;

          // Check for NaNs before prediction
          if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
              FAIL() << "NaN detected in state or covariance before prediction at t=" << t;
          }

          // Get true state
          StateVector true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

          try {
              // Predict forward
              filter->PredictNewReference(t);

              // Check for NaNs after prediction
              if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
                  FAIL() << "NaN detected in state or covariance after prediction at t=" << t;
              }

              StateVector predicted_state = filter->GetStateEstimate();
              StateMatrix predicted_cov = filter->GetStateCovariance();

              // Get measurement
              auto measurement = sensor->PredictMeasurement(true_state);

              // Compute auxiliary data
              auto aux_data = sensor->ComputeAuxiliaryData(
                  predicted_state,
                  predicted_cov,
                  measurement);

              // Check for NaNs in auxiliary data
              if (hasNaN(aux_data.innovation) || hasNaN(aux_data.jacobian) || hasNaN(aux_data.innovation_covariance)) {
                  FAIL() << "NaN detected in auxiliary data at t=" << t;
              }

              // Process measurement
              filter->ProcessMeasurementByIndex(sensor_idx, t, measurement);

              // Check for NaNs after update
              if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
                  FAIL() << "NaN detected in state or covariance after update at t=" << t;
              }

              // Get updated state
              StateVector updated_state = filter->GetStateEstimate();
              StateMatrix updated_cov = filter->GetStateCovariance();

              // Check each state triplet (X/Y/Z) that is initializable
              // Position
              if (initializable_flags[StateIndex::Position::X]) {
                  CheckStateTripleImproved(predicted_state, updated_state, true_state, predicted_cov, updated_cov,
                                          StateIndex::Position::Begin(), t, "Position");
              }

              // Linear Velocity
              if (initializable_flags[StateIndex::LinearVelocity::X]) {
                  CheckStateTripleImproved(predicted_state, updated_state, true_state, predicted_cov, updated_cov,
                                          StateIndex::LinearVelocity::Begin(), t, "Linear Velocity");
              }

              // Angular Velocity
              if (initializable_flags[StateIndex::AngularVelocity::X]) {
                  CheckStateTripleImproved(predicted_state, updated_state, true_state, predicted_cov, updated_cov,
                                          StateIndex::AngularVelocity::Begin(), t, "Angular Velocity");
              }

              // Linear Acceleration
              if (initializable_flags[StateIndex::LinearAcceleration::X]) {
                  CheckStateTripleImproved(predicted_state, updated_state, true_state, predicted_cov, updated_cov,
                                          StateIndex::LinearAcceleration::Begin(), t, "Linear Acceleration");
              }

              // Angular Acceleration
              if (initializable_flags[StateIndex::AngularAcceleration::X]) {
                  CheckStateTripleImproved(predicted_state, updated_state, true_state, predicted_cov, updated_cov,
                                          StateIndex::AngularAcceleration::Begin(), t, "Angular Acceleration");
              }

              // Store measurement covariance norm
              double current_measurement_cov_norm = sensor->GetMeasurementCovariance().norm();
              measurement_cov_norm_history.push_back(current_measurement_cov_norm);

              // Check that measurement covariance is decreasing until it reaches the floor
              if (i > 0 &&
                  measurement_cov_norm_history[i] > kMeasCovFloor && // Only enforce decrease above floor
                  current_measurement_cov_norm > measurement_cov_norm_history[i] * 1.05) { // Allow 5% fluctuation

                  FAIL() << "Measurement covariance norm increased significantly at t=" << t
                         << " (previous: " << measurement_cov_norm_history[i]
                         << ", current: " << current_measurement_cov_norm
                         << ", ratio: " << (current_measurement_cov_norm / measurement_cov_norm_history[i]) << ")";
              }

              // Ensure covariance stays within reasonable bounds
              if (current_measurement_cov_norm > kMeasCovCeiling) {
                  FAIL() << "Measurement covariance norm exceeded maximum at t=" << t
                         << " (maximum: " << kMeasCovCeiling
                         << ", current: " << current_measurement_cov_norm << ")";
              }

              // Update last good state
              last_good_state = updated_state;
              last_good_cov = updated_cov;
          }
          catch (const std::exception& e) {
              FAIL() << "Exception during filter operation at t=" << t << ": " << e.what();
          }
      }

      SUCCEED() << "All initializable states improved";
  }

  // Helper function to check if a triple of states (X/Y/Z) improved
  void CheckStateTripleImproved(
      const StateVector& predicted_state,
      const StateVector& updated_state,
      const StateVector& true_state,
      const StateMatrix& predicted_cov,
      const StateMatrix& updated_cov,
      int start_idx,
      double t,
      const std::string& state_name) {

      // Check that estimate improved
      double error_before = (predicted_state.segment<3>(start_idx) - true_state.segment<3>(start_idx)).norm();
      double error_after = (updated_state.segment<3>(start_idx) - true_state.segment<3>(start_idx)).norm();

      if (error_after > error_before) {
          FAIL() << state_name << " estimate did not improve at t=" << t
                 << " (before: " << error_before
                 << ", after: " << error_after << ")";
      }

      // Check that covariance decreased
      double cov_trace_before = predicted_cov.block<3,3>(start_idx, start_idx).trace();
      double cov_trace_after = updated_cov.block<3,3>(start_idx, start_idx).trace();

      if (cov_trace_after > cov_trace_before * 1.001) {
          FAIL() << state_name << " covariance did not decrease at t=" << t
                 << " (before: " << cov_trace_before
                 << ", after: " << cov_trace_after << ")";
      }
  }
};

// Test using the template method inside the fixture
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates) {
    // Test with automatic state detection based on sensor capabilities
    TestSensorImprovesStateEstimates(body_vel_model_);
}

} // namespace kinematic_arbiter
