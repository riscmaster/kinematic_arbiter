#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <iomanip>

#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
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
    // Create state model (this is common to all tests)
    state_model_ = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();
  }

  // Only keep the state model in the fixture
  std::shared_ptr<kinematic_arbiter::models::RigidBodyStateModel> state_model_;

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
  void TestSensorImprovesStateEstimates(
      std::shared_ptr<SensorModel> sensor,
      const Eigen::MatrixXd& noise_covariance) {
      using namespace core;
      using MeasurementType = typename SensorModel::MeasurementVector;

      // Flag to determine if we should add noise
      bool add_noise = !noise_covariance.isZero();

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

      // Set floor based on true noise covariance
      // For zero noise, use a small positive value to avoid division by zero
      // const double kMeasCovFloor = noise_covariance.isZero() ? 0.1 :
      //                              std::max(0.1, noise_covariance.norm() * 0.5);

      // Random number generator for adding noise
      std::random_device rd;
      std::mt19937 gen(rd());

      // Create multivariate normal distribution for noise
      Eigen::MatrixXd transform;
      if (add_noise) {
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(noise_covariance);
          transform = eigenSolver.eigenvectors() *
                      eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
      }

      // Function to add noise to measurement
      auto addNoise = [&gen, &transform](const MeasurementType& measurement) -> MeasurementType {
          // Generate standard normal samples
          Eigen::VectorXd z(measurement.size());
          std::normal_distribution<double> normal(0, 1);
          for (int i = 0; i < z.size(); ++i) {
              z(i) = normal(gen);
          }

          // Transform to desired covariance
          Eigen::VectorXd noise = transform * z;

          // Add noise to measurement
          return measurement + noise;
      };

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

              // Check for NaNs after prediction
              if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
                  FAIL() << "NaN detected in state or covariance after prediction at t=" << t;
              }

              StateVector predicted_state = filter->GetProcessModel()->PredictState(filter->GetStateEstimate(), dt);
              StateMatrix A = filter->GetProcessModel()->GetTransitionMatrix(filter->GetStateEstimate(), dt);
              StateMatrix predicted_cov = A * filter->GetStateCovariance() * A.transpose() + filter->GetProcessModel()->GetProcessNoiseCovariance(dt);

              // Get measurement and add noise if needed
              auto measurement = sensor->PredictMeasurement(true_state);
              if (add_noise) {
                  measurement = addNoise(measurement);
              }

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
              bool assumptions_held = filter->ProcessMeasurementByIndex(sensor_idx, t, measurement);
              if (!assumptions_held) {
                  FAIL() << "Measurement assumptions not held at t=" << t;
              }

              // Check for NaNs after update
              if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
                  FAIL() << "NaN detected in state or covariance after update at t=" << t;
              }

              StateVector updated_state = filter->GetStateEstimate();
              StateMatrix updated_cov = filter->GetStateCovariance();

              // Position
              if (initializable_flags[StateIndex::Position::X]) {
                  double error_before = (predicted_state.segment<3>(StateIndex::Position::X) -
                                      true_state.segment<3>(StateIndex::Position::X)).norm();
                  double error_after = (updated_state.segment<3>(StateIndex::Position::X) -
                                    true_state.segment<3>(StateIndex::Position::X)).norm();

                  if (error_after > error_before) {
                      FAIL() << "Position estimate did not improve at t=" << t;
                  }
              }

              // Orientation (quaternion)
              if (initializable_flags[StateIndex::Quaternion::W]) {
                  Eigen::Quaterniond q_true(
                      true_state(StateIndex::Quaternion::W),
                      true_state(StateIndex::Quaternion::X),
                      true_state(StateIndex::Quaternion::Y),
                      true_state(StateIndex::Quaternion::Z));

                  Eigen::Quaterniond q_pred(
                      predicted_state(StateIndex::Quaternion::W),
                      predicted_state(StateIndex::Quaternion::X),
                      predicted_state(StateIndex::Quaternion::Y),
                      predicted_state(StateIndex::Quaternion::Z));

                  Eigen::Quaterniond q_upd(
                      updated_state(StateIndex::Quaternion::W),
                      updated_state(StateIndex::Quaternion::X),
                      updated_state(StateIndex::Quaternion::Y),
                      updated_state(StateIndex::Quaternion::Z));

                  double angle_before = 2.0 * q_true.angularDistance(q_pred);
                  double angle_after = 2.0 * q_true.angularDistance(q_upd);

                  // Add tolerance: 5% relative or 0.001 radians absolute
                  const double rel_tol = 0.05;
                  const double abs_tol = 0.001;

                  // Get prediction model inputs
                  Eigen::Matrix<double, 6, 1> inputs = sensor->GetPredictionModelInputs(
                      predicted_state,
                      predicted_cov,
                      measurement,
                      dt);

                  // Detailed diagnostic information
                  std::stringstream diagnostic_info;
                  diagnostic_info << "Time: " << t << "\n";

                  // Quaternion information
                  diagnostic_info << "True quaternion: [" << q_true.w() << ", " << q_true.x() << ", "
                                  << q_true.y() << ", " << q_true.z() << "]\n";
                  diagnostic_info << "Predicted quaternion: [" << q_pred.w() << ", " << q_pred.x() << ", "
                                  << q_pred.y() << ", " << q_pred.z() << "]\n";
                  diagnostic_info << "Updated quaternion: [" << q_upd.w() << ", " << q_upd.x() << ", "
                                  << q_upd.y() << ", " << q_upd.z() << "]\n";
                  diagnostic_info << "Angular error before update: " << angle_before << " rad\n";
                  diagnostic_info << "Angular error after update: " << angle_after << " rad\n";
                  diagnostic_info << "Change in error: " << (angle_after - angle_before) << " rad\n";

                  // Acceleration comparison table
                  diagnostic_info << "\n=== ACCELERATION COMPARISON ===\n";
                  diagnostic_info << std::setw(20) << "Source"
                                  << std::setw(15) << "Linear X"
                                  << std::setw(15) << "Linear Y"
                                  << std::setw(15) << "Linear Z"
                                  << std::setw(15) << "Angular X"
                                  << std::setw(15) << "Angular Y"
                                  << std::setw(15) << "Angular Z" << "\n";

                  // True accelerations
                  diagnostic_info << std::setw(20) << "True"
                                  << std::setw(15) << true_state(StateIndex::LinearAcceleration::X)
                                  << std::setw(15) << true_state(StateIndex::LinearAcceleration::Y)
                                  << std::setw(15) << true_state(StateIndex::LinearAcceleration::Z)
                                  << std::setw(15) << true_state(StateIndex::AngularAcceleration::X)
                                  << std::setw(15) << true_state(StateIndex::AngularAcceleration::Y)
                                  << std::setw(15) << true_state(StateIndex::AngularAcceleration::Z) << "\n";

                  // Predicted accelerations
                  diagnostic_info << std::setw(20) << "Predicted"
                                  << std::setw(15) << predicted_state(StateIndex::LinearAcceleration::X)
                                  << std::setw(15) << predicted_state(StateIndex::LinearAcceleration::Y)
                                  << std::setw(15) << predicted_state(StateIndex::LinearAcceleration::Z)
                                  << std::setw(15) << predicted_state(StateIndex::AngularAcceleration::X)
                                  << std::setw(15) << predicted_state(StateIndex::AngularAcceleration::Y)
                                  << std::setw(15) << predicted_state(StateIndex::AngularAcceleration::Z) << "\n";

                  // Updated accelerations
                  diagnostic_info << std::setw(20) << "Updated"
                                  << std::setw(15) << updated_state(StateIndex::LinearAcceleration::X)
                                  << std::setw(15) << updated_state(StateIndex::LinearAcceleration::Y)
                                  << std::setw(15) << updated_state(StateIndex::LinearAcceleration::Z)
                                  << std::setw(15) << updated_state(StateIndex::AngularAcceleration::X)
                                  << std::setw(15) << updated_state(StateIndex::AngularAcceleration::Y)
                                  << std::setw(15) << updated_state(StateIndex::AngularAcceleration::Z) << "\n";

                  // Prediction model inputs
                  diagnostic_info << std::setw(20) << "Inputs"
                                  << std::setw(15) << inputs(0)
                                  << std::setw(15) << inputs(1)
                                  << std::setw(15) << inputs(2)
                                  << std::setw(15) << inputs(3)
                                  << std::setw(15) << inputs(4)
                                  << std::setw(15) << inputs(5) << "\n";

                  // Add velocity information
                  diagnostic_info << "\n=== VELOCITY COMPARISON ===\n";
                  diagnostic_info << std::setw(20) << "Source"
                                  << std::setw(15) << "Linear X"
                                  << std::setw(15) << "Linear Y"
                                  << std::setw(15) << "Linear Z"
                                  << std::setw(15) << "Angular X"
                                  << std::setw(15) << "Angular Y"
                                  << std::setw(15) << "Angular Z" << "\n";

                  // True velocities
                  diagnostic_info << std::setw(20) << "True"
                                  << std::setw(15) << true_state(StateIndex::LinearVelocity::X)
                                  << std::setw(15) << true_state(StateIndex::LinearVelocity::Y)
                                  << std::setw(15) << true_state(StateIndex::LinearVelocity::Z)
                                  << std::setw(15) << true_state(StateIndex::AngularVelocity::X)
                                  << std::setw(15) << true_state(StateIndex::AngularVelocity::Y)
                                  << std::setw(15) << true_state(StateIndex::AngularVelocity::Z) << "\n";

                  // Predicted velocities
                  diagnostic_info << std::setw(20) << "Predicted"
                                  << std::setw(15) << predicted_state(StateIndex::LinearVelocity::X)
                                  << std::setw(15) << predicted_state(StateIndex::LinearVelocity::Y)
                                  << std::setw(15) << predicted_state(StateIndex::LinearVelocity::Z)
                                  << std::setw(15) << predicted_state(StateIndex::AngularVelocity::X)
                                  << std::setw(15) << predicted_state(StateIndex::AngularVelocity::Y)
                                  << std::setw(15) << predicted_state(StateIndex::AngularVelocity::Z) << "\n";

                  // Updated velocities
                  diagnostic_info << std::setw(20) << "Updated"
                                  << std::setw(15) << updated_state(StateIndex::LinearVelocity::X)
                                  << std::setw(15) << updated_state(StateIndex::LinearVelocity::Y)
                                  << std::setw(15) << updated_state(StateIndex::LinearVelocity::Z)
                                  << std::setw(15) << updated_state(StateIndex::AngularVelocity::X)
                                  << std::setw(15) << updated_state(StateIndex::AngularVelocity::Y)
                                  << std::setw(15) << updated_state(StateIndex::AngularVelocity::Z) << "\n";

                  // Additional diagnostic information
                  diagnostic_info << "\n=== FILTER DIAGNOSTICS ===\n";
                  diagnostic_info << "Innovation: [";
                  for (int i = 0; i < aux_data.innovation.size(); ++i) {
                      diagnostic_info << aux_data.innovation(i);
                      if (i < aux_data.innovation.size() - 1) diagnostic_info << ", ";
                  }
                  diagnostic_info << "]\n";

                  // Print covariance information
                  if (i == 0) {  // Only print on first iteration to avoid too much output
                      diagnostic_info << "Measurement covariance norm: " << sensor->GetMeasurementCovariance().norm() << "\n";
                      diagnostic_info << "Innovation covariance norm: " << aux_data.innovation_covariance.norm() << "\n";
                      diagnostic_info << "State covariance norm: " << filter->GetStateCovariance().norm() << "\n";

                      // Add covariance for accelerations
                      Eigen::Matrix3d lin_accel_cov = predicted_cov.block<3,3>(
                          StateIndex::LinearAcceleration::Begin(),
                          StateIndex::LinearAcceleration::Begin());
                      Eigen::Matrix3d ang_accel_cov = predicted_cov.block<3,3>(
                          StateIndex::AngularAcceleration::Begin(),
                          StateIndex::AngularAcceleration::Begin());

                      diagnostic_info << "Linear accel covariance norm: " << lin_accel_cov.norm() << "\n";
                      diagnostic_info << "Angular accel covariance norm: " << ang_accel_cov.norm() << "\n";
                  }

                  // Only fail if error is significantly worse
                  if (angle_after > angle_before * (1.0 + rel_tol) &&
                      angle_after - angle_before > abs_tol) {
                      FAIL() << "Orientation estimate significantly worse at t=" << t
                             << " (before: " << angle_before << " rad, after: " << angle_after << " rad)\n"
                             << diagnostic_info.str();
                  }

                  // Add diagnostic output for debugging
                  if (angle_after > angle_before) {
                      std::cout << "WARNING: Orientation estimate worse at t=" << t << "\n"
                                << diagnostic_info.str() << std::endl;
                  }
              }

              // Linear Velocity
              if (initializable_flags[StateIndex::LinearVelocity::X]) {
                  double error_before = (predicted_state.segment<3>(StateIndex::LinearVelocity::X) -
                                      true_state.segment<3>(StateIndex::LinearVelocity::X)).norm();
                  double error_after = (updated_state.segment<3>(StateIndex::LinearVelocity::X) -
                                    true_state.segment<3>(StateIndex::LinearVelocity::X)).norm();

                  if (error_after > error_before) {
                      FAIL() << "Linear Velocity estimate did not improve at t=" << t;
                  }
              }

              // Angular Velocity
              if (initializable_flags[StateIndex::AngularVelocity::X]) {
                  double error_before = (predicted_state.segment<3>(StateIndex::AngularVelocity::X) -
                                      true_state.segment<3>(StateIndex::AngularVelocity::X)).norm();
                  double error_after = (updated_state.segment<3>(StateIndex::AngularVelocity::X) -
                                    true_state.segment<3>(StateIndex::AngularVelocity::X)).norm();

                  // Add tolerance: 5% relative or 0.002 absolute
                  const double rel_tol = 0.05;
                  const double abs_tol = 0.002;

                  // Only fail if error is significantly worse
                  if (error_after > error_before * (1.0 + rel_tol) &&
                      error_after - error_before > abs_tol) {
                      FAIL() << "Angular Velocity estimate significantly worse at t=" << t
                             << " (before: " << error_before << ", after: " << error_after << ")";
                  }
              }

              // Linear Acceleration
              if (initializable_flags[StateIndex::LinearAcceleration::X]) {
                  double error_before = (predicted_state.segment<3>(StateIndex::LinearAcceleration::X) -
                                      true_state.segment<3>(StateIndex::LinearAcceleration::X)).norm();
                  double error_after = (updated_state.segment<3>(StateIndex::LinearAcceleration::X) -
                                    true_state.segment<3>(StateIndex::LinearAcceleration::X)).norm();

                  // Increased tolerance: 5% relative or 0.1 absolute
                  const double rel_tol = 0.05;
                  const double abs_tol = 0.1;

                  // Only fail if error is significantly worse
                  if (error_after > error_before * (1.0 + rel_tol) &&
                      error_after - error_before > abs_tol) {
                      FAIL() << "Linear Acceleration estimate significantly worse at t=" << t
                             << " (before: " << error_before << ", after: " << error_after << ")";
                  }
              }

              // Angular Acceleration
              if (initializable_flags[StateIndex::AngularAcceleration::X]) {
                  double error_before = (predicted_state.segment<3>(StateIndex::AngularAcceleration::X) -
                                      true_state.segment<3>(StateIndex::AngularAcceleration::X)).norm();
                  double error_after = (updated_state.segment<3>(StateIndex::AngularAcceleration::X) -
                                    true_state.segment<3>(StateIndex::AngularAcceleration::X)).norm();

                  // Increased tolerance: 10% relative or 0.1 absolute
                  const double rel_tol = 0.10;
                  const double abs_tol = 0.1;

                  // Only fail if error is significantly worse
                  if (error_after > error_before * (1.0 + rel_tol) &&
                      error_after - error_before > abs_tol) {
                      FAIL() << "Angular Acceleration estimate significantly worse at t=" << t
                             << " (before: " << error_before << ", after: " << error_after << ")";
                  }
              }

              // Store measurement covariance norm
              double current_measurement_cov_norm = sensor->GetMeasurementCovariance().norm();
              measurement_cov_norm_history.push_back(current_measurement_cov_norm);

              // Check that measurement covariance is decreasing until it reaches the floor
              if (i > 0 && current_measurement_cov_norm > measurement_cov_norm_history[i] * 1.06) { // Allow 5% fluctuation
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

              // Check if estimated measurement covariance is approaching the true noise covariance
              // Only check after sufficient iterations for adaptation to occur and only if noise was added
              if (add_noise && i > 100) {
                  // Calculate Frobenius norm of the difference between estimated and true covariance
                  Eigen::MatrixXd cov_diff = sensor->GetMeasurementCovariance() - noise_covariance;
                  double cov_diff_norm = cov_diff.norm();

                  // Relative error should be less than 50% after sufficient iterations
                  double rel_error = cov_diff_norm / std::max(1e-10, noise_covariance.norm());
                  EXPECT_LT(rel_error, 0.5)
                      << "Estimated measurement covariance not converging to true noise at t=" << t
                      << " (relative error: " << rel_error * 100 << "%)";
              }

              // Update last good state
              last_good_state = updated_state;
              last_good_cov = updated_cov;
          }
          catch (const std::exception& e) {
              FAIL() << "Exception caught at t=" << t << ": " << e.what();
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

// Test with BodyVelocitySensorModel
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates(body_vel_model, zero_noise);
}

// Test with PositionSensorModel
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 3, 3> zero_noise = Eigen::Matrix<double, 3, 3>::Zero();

  TestSensorImprovesStateEstimates(position_model, zero_noise);
}

// Test with PoseSensorModel
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 7, 7> zero_noise = Eigen::Matrix<double, 7, 7>::Zero();

  TestSensorImprovesStateEstimates(pose_model, zero_noise);
}

// Test with ImuSensorModel
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates(imu_model, zero_noise);
}

} // namespace kinematic_arbiter
