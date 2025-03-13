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
};

// Velocity sensor measurement test - single focused test
TEST_F(MediatedKalmanFilterTest, VelocitySensorShouldImproveVelocityEstimates) {
  // Create filter
  auto filter = std::make_shared<FilterType>(state_model_);

  // Add body velocity sensor
  size_t vel_idx = filter->AddSensor(body_vel_model_);

  // Test parameters
  const double dt = 0.05;
  double t = 0.0;

  // Initialize with true state
  StateVector initial_state = kinematic_arbiter::testing::Figure8Trajectory(t);
  filter->SetStateEstimate(initial_state);

  // Set initial covariance
  StateMatrix covariance = StateMatrix::Identity() * 0.1;
  filter->SetStateCovariance(covariance);

  // Function to check for NaNs
  auto hasNaN = [](const auto& mat) -> bool {
    return !(mat.array() == mat.array()).all();
  };

  // Last successful state and time for diagnostics
  StateVector last_good_state = initial_state;
  StateMatrix last_good_cov = covariance;
  double last_good_time = t;

  // Track covariance history for checks
  std::vector<double> vel_cov_trace_history;
  std::vector<double> measurement_cov_norm_history;

  // Store initial measurement covariance norm
  auto vel_sensor = filter->GetSensorByIndex<kinematic_arbiter::sensors::BodyVelocitySensorModel>(vel_idx);
  double initial_measurement_cov_norm = vel_sensor->GetMeasurementCovariance().norm();
  measurement_cov_norm_history.push_back(initial_measurement_cov_norm);

  // Set ceiling based on initial covariance norm (allowing some margin)
  const double kMeasCovCeiling = initial_measurement_cov_norm * 1.5;

  // Test for 10 seconds
  for (int i = 0; i < 200; i++) {
    // Move to next time step
    t += dt;

    // Check for NaNs in state or covariance before prediction
    if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
      std::cerr << "\n========= NaN DETECTED BEFORE PREDICTION at t=" << t << " =========\n";
      std::cerr << "Last good state time: " << last_good_time << "\n";
      std::cerr << "Last good velocity: " << last_good_state.segment<3>(SIdx::LinearVelocity::X).transpose() << "\n";
      std::cerr << "Last good velocity covariance:\n"
                << last_good_cov.block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X) << "\n";
      FAIL() << "NaN detected in state or covariance before prediction at t=" << t;
    }

    // Get true state
    StateVector true_state = kinematic_arbiter::testing::Figure8Trajectory(t);

    try {
      // Predict forward
      filter->PredictNewReference(t);

      // Check for NaNs after prediction
      if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
        std::cerr << "\n========= NaN DETECTED AFTER PREDICTION at t=" << t << " =========\n";
        std::cerr << "Last good state time: " << last_good_time << "\n";
        std::cerr << "Last good velocity: " << last_good_state.segment<3>(SIdx::LinearVelocity::X).transpose() << "\n";
        std::cerr << "Last good velocity covariance:\n"
                  << last_good_cov.block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X) << "\n";
        FAIL() << "NaN detected in state or covariance after prediction at t=" << t;
      }

      StateVector predicted_state = filter->GetStateEstimate();
      StateMatrix predicted_cov = filter->GetStateCovariance();

      // Calculate velocity error after prediction
      double vel_error_predicted = (predicted_state.segment<3>(SIdx::LinearVelocity::X) -
                                 true_state.segment<3>(SIdx::LinearVelocity::X)).norm();

      // Get body velocity sensor and measurement
      auto vel_sensor = filter->GetSensorByIndex<kinematic_arbiter::sensors::BodyVelocitySensorModel>(vel_idx);
      auto measurement = vel_sensor->PredictMeasurement(true_state);

      auto measurement_cov = vel_sensor->GetMeasurementCovariance();

      // Compute auxiliary data for NaN checks
      auto aux_data = vel_sensor->ComputeAuxiliaryData(
        predicted_state,
        predicted_cov,
        measurement);

      // Check for NaNs in auxiliary data
      if (hasNaN(aux_data.innovation) || hasNaN(aux_data.jacobian) || hasNaN(aux_data.innovation_covariance)) {
        std::cerr << "\n========= NaN DETECTED IN AUXILIARY DATA at t=" << t << " =========\n";
        std::cerr << "Last good state time: " << last_good_time << "\n";
        std::cerr << "Predicted velocity: " << predicted_state.segment<3>(SIdx::LinearVelocity::X).transpose() << "\n";
        std::cerr << "Predicted velocity covariance:\n"
                  << predicted_cov.block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X) << "\n";
        std::cerr << "Measurement: " << measurement.transpose() << "\n";

        // Check which parts contain NaN
        std::cerr << "NaN location(s):\n";
        std::cerr << "  Innovation: " << (hasNaN(aux_data.innovation) ? "YES" : "NO") << "\n";
        std::cerr << "  Jacobian: " << (hasNaN(aux_data.jacobian) ? "YES" : "NO") << "\n";
        std::cerr << "  Innovation covariance: " << (hasNaN(aux_data.innovation_covariance) ? "YES" : "NO") << "\n";

        FAIL() << "NaN detected in auxiliary data at t=" << t;
      }

      // Process measurement
      filter->ProcessMeasurementByIndex(vel_idx, t, measurement);

      // Check for NaNs after update
      if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
        std::cerr << "\n========= NaN DETECTED AFTER UPDATE at t=" << t << " =========\n";
        std::cerr << "Last good state time: " << last_good_time << "\n";

        // Print more comprehensive information about the state before NaN
        std::cerr << "PREDICTED STATE (before update):\n";
        std::cerr << "Position: " << predicted_state.segment<3>(SIdx::Position::X).transpose() << "\n";
        std::cerr << "Velocity: " << predicted_state.segment<3>(SIdx::LinearVelocity::X).transpose() << "\n";

        // Show measurement and innovation
        std::cerr << "MEASUREMENT:\n" << measurement.transpose() << "\n\n";
        std::cerr << "INNOVATION:\n" << aux_data.innovation.transpose() << "\n\n";

        // Show detailed covariance blocks
        std::cerr << "STATE COVARIANCE (key blocks before update):\n";
        std::cerr << "Position covariance:\n"
                  << predicted_cov.block<3,3>(SIdx::Position::X, SIdx::Position::X) << "\n\n";
        std::cerr << "Velocity covariance:\n"
                  << predicted_cov.block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X) << "\n\n";
        std::cerr << "Position-velocity cross covariance:\n"
                  << predicted_cov.block<3,3>(SIdx::Position::X, SIdx::LinearVelocity::X) << "\n\n";

        // Show sensor information
        std::cerr << "SENSOR INFORMATION:\n";
        std::cerr << "Measurement covariance (R):\n" << vel_sensor->GetMeasurementCovariance() << "\n\n";
        std::cerr << "Innovation covariance (S):\n" << aux_data.innovation_covariance << "\n\n";
        std::cerr << "Jacobian (H):\n" << aux_data.jacobian << "\n\n";

        // Show which components have NaN
        std::cerr << "NaN DETECTION:\n";

        // Check state for NaNs
        bool nan_in_state = false;
        const StateVector& current_state = filter->GetStateEstimate();
        for (int i = 0; i < current_state.size(); i++) {
          if (std::isnan(current_state[i])) {
            if (!nan_in_state) {
              std::cerr << "State contains NaN at indices: ";
              nan_in_state = true;
            }
            std::cerr << i << " ";
          }
        }
        if (nan_in_state) std::cerr << "\n";
        else std::cerr << "No NaNs detected in state vector\n";

        // Check covariance for NaNs
        bool nan_in_cov = false;
        int nan_count = 0; // Declare nan_count variable
        const StateMatrix& current_cov = filter->GetStateCovariance();
        for (int i = 0; i < current_cov.rows(); i++) {
          for (int j = 0; j < current_cov.cols(); j++) {
            if (std::isnan(current_cov(i,j))) {
              if (!nan_in_cov) {
                std::cerr << "Covariance contains NaN at indices: ";
                nan_in_cov = true;
              }
              std::cerr << "(" << i << "," << j << ") ";
              // Only show first 10 NaN locations to avoid flooding output
              if (++nan_count >= 10) {
                std::cerr << "... (more NaNs present)";
                i = current_cov.rows(); // Break out of both loops
                break;
              }
            }
          }
        }
        if (nan_in_cov) std::cerr << "\n";
        else std::cerr << "No NaNs detected in covariance matrix\n";

        FAIL() << "NaN detected in state or covariance after update at t=" << t;
      }

      // Get updated state and calculate error
      StateVector updated_state = filter->GetStateEstimate();
      double vel_error_updated = (updated_state.segment<3>(SIdx::LinearVelocity::X) -
                                true_state.segment<3>(SIdx::LinearVelocity::X)).norm();

      // Store velocity covariance trace
      double current_trace = filter->GetStateCovariance().block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X).trace();
      vel_cov_trace_history.push_back(current_trace);

      // Store measurement covariance norm
      double current_measurement_cov_norm = vel_sensor->GetMeasurementCovariance().norm();
      measurement_cov_norm_history.push_back(current_measurement_cov_norm);

      // Verify update improved velocity estimate
      if (vel_error_updated > vel_error_predicted) {
        PrintFilterDiagnostics(
            filter,
            vel_sensor,
            t,
            true_state,
            predicted_state,
            predicted_cov,
            updated_state,
            measurement,
            aux_data,
            vel_error_predicted,
            vel_error_updated,
            "VELOCITY ERROR WORSE"
        );

        FAIL() << "Velocity measurement update at t=" << t
               << " made velocity error worse (predicted: " << vel_error_predicted
               << ", updated: " << vel_error_updated << ")";
      }

      // Check that update step decreased covariance
      double predicted_trace = predicted_cov.block<3,3>(core::StateIndex::LinearVelocity::X, core::StateIndex::LinearVelocity::X).trace();
      if (current_trace > predicted_trace * 1.001) {
        PrintFilterDiagnostics(
            filter,
            vel_sensor,
            t,
            true_state,
            predicted_state,
            predicted_cov,
            updated_state,
            measurement,
            aux_data,
            vel_error_predicted,
            vel_error_updated,
            "UPDATE DIDN'T DECREASE COVARIANCE"
        );

        FAIL() << "Velocity covariance didn't decrease during update at t=" << t
                << " (predicted: " << predicted_trace
                << ", after update: " << current_trace << ")";
      }

      // Check that measurement covariance is decreasing until it reaches a reasonable floor
      const double kMeasCovFloor = 0.1; // Floor value based on observed behavior
      if (i > 0 &&
          measurement_cov_norm_history[i] > kMeasCovFloor && // Only enforce decrease above floor
          current_measurement_cov_norm > measurement_cov_norm_history[i] * 1.05) { // Allow 5% fluctuation
        PrintFilterDiagnostics(
            filter,
            vel_sensor,
            t,
            true_state,
            predicted_state,
            predicted_cov,
            updated_state,
            measurement,
            aux_data,
            vel_error_predicted,
            vel_error_updated,
            "MEASUREMENT COVARIANCE INCREASED SIGNIFICANTLY"
        );

        FAIL() << "Measurement covariance norm increased significantly at t=" << t
               << " (previous: " << measurement_cov_norm_history[i]
               << ", current: " << current_measurement_cov_norm
               << ", ratio: " << (current_measurement_cov_norm / measurement_cov_norm_history[i]) << ")";
      }

      // Ensure covariance stays within reasonable bounds
      if (current_measurement_cov_norm > kMeasCovCeiling) {
        PrintFilterDiagnostics(
            filter,
            vel_sensor,
            t,
            true_state,
            predicted_state,
            predicted_cov,
            updated_state,
            measurement,
            aux_data,
            vel_error_predicted,
            vel_error_updated,
            "MEASUREMENT COVARIANCE EXCEEDED MAXIMUM"
        );

        FAIL() << "Measurement covariance norm exceeded maximum at t=" << t
               << " (maximum: " << kMeasCovCeiling
               << ", current: " << current_measurement_cov_norm << ")";
      }

      // Update last good state and time
      last_good_state = updated_state;
      last_good_cov = filter->GetStateCovariance();
      last_good_time = t;
    }
    catch (const std::exception& e) {
      std::cerr << "\n========= EXCEPTION at t=" << t << " =========\n";
      std::cerr << "Last good state time: " << last_good_time << "\n";
      std::cerr << "Last good velocity: " << last_good_state.segment<3>(SIdx::LinearVelocity::X).transpose() << "\n";
      std::cerr << "Last good velocity covariance:\n"
                << last_good_cov.block<3,3>(SIdx::LinearVelocity::X, SIdx::LinearVelocity::X) << "\n";
      std::cerr << "Exception: " << e.what() << "\n";
      FAIL() << "Exception during filter operation at t=" << t << ": " << e.what();
    }
  }

  // Print measurement covariance reduction statistics
  double initial_meas_cov_norm = measurement_cov_norm_history.front();
  double final_meas_cov_norm = measurement_cov_norm_history.back();
  double meas_cov_percent_reduction = (1.0 - final_meas_cov_norm/initial_meas_cov_norm) * 100;

  std::cerr << "\n========= MEASUREMENT COVARIANCE REDUCTION SUMMARY =========\n";
  std::cerr << "Initial measurement covariance norm: " << initial_meas_cov_norm << "\n";
  std::cerr << "Final measurement covariance norm: " << final_meas_cov_norm << "\n";
  std::cerr << "Percent reduction: " << meas_cov_percent_reduction << "%\n";

  SUCCEED() << "All velocity measurements improved the velocity estimate";
}

} // namespace kinematic_arbiter
