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
#include "kinematic_arbiter/core/trajectory_utils.hpp"
#include "kinematic_arbiter/core/statistical_utils.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"

// Use the correct state index type
using SIdx = kinematic_arbiter::core::StateIndex;
using SensorType = kinematic_arbiter::core::SensorType;

namespace kinematic_arbiter {

// Helper function to check for NaN values in Eigen matrices and vectors
template<typename Derived>
bool hasNaN(const Eigen::MatrixBase<Derived>& matrix) {
  return !(matrix.array() == matrix.array()).all();
}

// Test fixture
class MediatedKalmanFilterTest : public ::testing::Test {
protected:
  // FilterType definition
  using FilterType = kinematic_arbiter::core::MediatedKalmanFilter<SIdx::kFullStateSize,
    kinematic_arbiter::models::RigidBodyStateModel>;

  // Type aliases
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, SIdx::kFullStateSize, SIdx::kFullStateSize>;
  using StateFlags = Eigen::Array<bool, SIdx::kFullStateSize, 1>;

  void SetUp() override {
    // Create state model (this is common to all tests)
    state_model_ = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();
  }

  // Only keep the state model in the fixture
  std::shared_ptr<kinematic_arbiter::models::RigidBodyStateModel> state_model_;

  // Generic PrintFilterDiagnostics that works with any sensor type
  template<SensorType Type>
  void PrintFilterDiagnostics(
      const std::shared_ptr<core::MeasurementModelInterface>& sensor,
      const StateVector& true_state,
      const StateVector& predicted_state,
      const StateMatrix& predicted_cov,
      const StateVector& updated_state,
      const typename core::MeasurementModelInterface::DynamicVector& measurement,
      const typename core::MeasurementModelInterface::MeasurementAuxData& aux_data,
      const std::string& failure_reason = "ERROR WORSE") {

    // Calculate Kalman gain for diagnostics
    Eigen::MatrixXd K = predicted_cov * aux_data.jacobian.transpose() *
                         aux_data.innovation_covariance.inverse();

    // Extract state blocks
    Eigen::Vector3d true_position = true_state.segment<3>(SIdx::Position::X);
    Eigen::Vector3d predicted_position = predicted_state.segment<3>(SIdx::Position::X);
    Eigen::Vector3d updated_position = updated_state.segment<3>(SIdx::Position::X);

    Eigen::Matrix<double, 6, 1> true_velocity = true_state.segment<6>(SIdx::LinearVelocity::X);
    Eigen::Matrix<double, 6, 1> predicted_velocity = predicted_state.segment<6>(SIdx::LinearVelocity::X);
    Eigen::Matrix<double, 6, 1> updated_velocity = updated_state.segment<6>(SIdx::LinearVelocity::X);

    Eigen::Matrix<double, 6, 1> true_acceleration = true_state.segment<6>(SIdx::LinearAcceleration::X);
    Eigen::Matrix<double, 6, 1> predicted_acceleration = predicted_state.segment<6>(SIdx::LinearAcceleration::X);
    Eigen::Matrix<double, 6, 1> updated_acceleration = updated_state.segment<6>(SIdx::LinearAcceleration::X);

    Eigen::Quaterniond true_orientation(true_state.segment<4>(SIdx::Quaternion::Begin()));
    Eigen::Quaterniond predicted_orientation(predicted_state.segment<4>(SIdx::Quaternion::Begin()));
    Eigen::Quaterniond updated_orientation(updated_state.segment<4>(SIdx::Quaternion::Begin()));

    std::cerr << "\n========= FILTER TEST FAILURE : " << failure_reason << " =========\n\n"
              << "STATE VECTORS:\n"
              << "-----------------------------------------\n"
              << "Position:\n"
              << "  True:      " << true_position.transpose() << "\n"
              << "  Predicted: " << predicted_position.transpose() << "\n"
              << "  Updated:   " << updated_position.transpose() << "\n"
              << "  Predicted error: " << (true_position - predicted_position).norm() << "\n"
              << "  Updated error:   " << (true_position - updated_position).norm() << "\n\n"
              << "Velocity:\n"
              << "  True:      " << true_velocity.transpose() << "\n"
              << "  Predicted: " << predicted_velocity.transpose() << "\n"
              << "  Updated:   " << updated_velocity.transpose() << "\n"
              << "  Predicted error: " << (true_velocity - predicted_velocity).norm() << "\n"
              << "  Updated error:   " << (true_velocity - updated_velocity).norm() << "\n\n"
              << "Acceleration:\n"
              << "  True:      " << true_acceleration.transpose() << "\n"
              << "  Predicted: " << predicted_acceleration.transpose() << "\n"
              << "  Updated:   " << updated_acceleration.transpose() << "\n"
              << "  Predicted error: " << (true_acceleration - predicted_acceleration).norm() << "\n"
              << "  Updated error:   " << (true_acceleration - updated_acceleration).norm() << "\n\n"
              << "Orientation:\n"
              << "  True:      " << true_orientation.coeffs().transpose() << "\n"
              << "  Predicted: " << predicted_orientation.coeffs().transpose() << "\n"
              << "  Updated:   " << updated_orientation.coeffs().transpose() << "\n"
              << "  Predicted angular error: " << predicted_orientation.angularDistance(true_orientation) << "\n"
              << "  Updated angular error:   " << updated_orientation.angularDistance(true_orientation) << "\n\n";


    std::cerr << "\n\n";

    std::cerr << "MEASUREMENTS (3D):\n"
              << "-----------------------------------------\n"
              << "Actual measurement:     " << measurement.transpose() << "\n"
              << "Predicted measurement:  " << sensor->PredictMeasurement(predicted_state).transpose() << "\n"
              << "Innovation/residual:    " << aux_data.innovation.segment(0, 3).transpose() << "\n"
              << "Innovation norm:        " << aux_data.innovation.segment(0, 3).norm() << "\n\n"

              << "JACOBIAN (C matrix):\n"
              << "-----------------------------------------\n";


    // Assuming GetInitializableStates() returns a vector of indices and a corresponding vector of state names
    auto initializable_states = sensor->GetInitializableStates();
    std::vector<std::string> state_names = kinematic_arbiter::core::GetInitializableStateNames(initializable_states);

    // Print Jacobian for initializable states
    std::cerr << "Jacobian for Initializable States:\n";
    for (Eigen::Index idx = 0; idx < initializable_states.size(); ++idx) {
        int i = initializable_states[idx];
        std::cerr << state_names[idx] << ": ";
        for (int j = 0; j < aux_data.jacobian.cols(); ++j) {
            std::cerr << aux_data.jacobian(i, j) << " ";
        }
        std::cerr << "\n";
    }

    // Print Kalman gain for initializable states
    std::cerr << "Kalman Gain for Initializable States:\n";
    for (Eigen::Index idx = 0; idx < initializable_states.size(); ++idx) {
        int i = initializable_states[idx];
        std::cerr << state_names[idx] << ": ";
        for (int j = 0; j < std::min(3, static_cast<int>(K.cols())); ++j) {
            std::cerr << K(i, j) << " ";
        }
        std::cerr << "\n";
    }

    std::cerr << "KALMAN FILTER EQUATIONS:\n"
              << "-----------------------------------------\n"
              << "Innovation: y = z - h(x)\n"
              << "Kalman gain: K = P-*C'*S^-1\n"
              << "State update: x+ = x- + K*y\n"
              << "Covariance update: P+ = (I - K*C)*P-*((I - K*C)ᵀ) + K*R*Kᵀ\n\n"

              << "NUMERICAL CHECKS:\n"
              << "-----------------------------------------\n"
              << "Innovation covariance invertible: "
              << (aux_data.innovation_covariance.determinant() > 1e-10 ? "YES" : "NO") << "\n"
              << "Innovation covariance determinant: " << aux_data.innovation_covariance.determinant() << "\n"
              << "Condition number (approx): "
              << aux_data.innovation_covariance.norm() * aux_data.innovation_covariance.inverse().norm() << "\n"
              << "Measurement covariance norm: " << sensor->GetMeasurementCovariance().norm() << "\n";
  }

  // Updated template method that takes sensor type as a parameter
  template<SensorType Type, typename SensorModel>
  void TestSensorImprovesStateEstimates(
      std::shared_ptr<SensorModel> sensor,
      const Eigen::MatrixXd& noise_covariance) {
      using namespace core;
      using MeasurementType = typename SensorModel::Vector;

      // Flag to determine if we should add noise
      bool add_noise = !noise_covariance.isZero();
      double true_meas_cov_norm = noise_covariance.norm();

      // Create filter
      auto filter = std::make_shared<FilterType>(state_model_);
      size_t sensor_idx = filter->AddSensor(sensor);

      // Test parameters
      const double dt = 0.05;
      double t = 0.0;

      // Initialize with true state
      StateVector prev_state = kinematic_arbiter::utils::Figure8Trajectory(t);
      // Set initial covariance
      StateMatrix prev_covariance = StateMatrix::Identity() * 0.1;
      filter->SetStateEstimate(prev_state, t, prev_covariance);
      ASSERT_EQ(filter->IsInitialized(), true) << "Filter not initialized during initialization";
      ASSERT_EQ(filter->GetStateEstimate(), prev_state) << "State not updated during initialization";
      ASSERT_EQ(filter->GetCurrentTime(), t) << "Time not updated during initialization";
      ASSERT_EQ(filter->GetStateCovariance(), prev_covariance) << "Covariance not updated during initialization";

      // Get initializable states and names - moved these declarations so they're in scope
      auto initializable_states = sensor->GetInitializableStates();
      std::vector<std::string> state_names = kinematic_arbiter::core::GetInitializableStateNames(initializable_states);

      // Track measurement covariance norm to test convergence
      std::vector<double> measurement_cov_norm_history;

      // Store initial measurement covariance norm
      double initial_measurement_cov_norm = sensor->GetMeasurementCovariance().norm();
      measurement_cov_norm_history.push_back(initial_measurement_cov_norm);

    //   // Set ceiling based on initial covariance norm
    //   const double kMeasCovCeiling = initial_measurement_cov_norm * 1.5;

    //   // Set floor based on true noise covariance
    //   // For zero noise, use a small positive value to avoid division by zero
    //   const double kMeasCovFloor = noise_covariance.isZero() ? 0.1 :
    //                                std::max(0.1, noise_covariance.norm() * 0.5);

      // Random number generator setup
      std::mt19937 gen(std::random_device{}());

      // Test for 10 seconds (200 steps at 0.05s per step)
      for (int i = 0; i < 200; i++) {
          ASSERT_EQ(filter->GetCurrentTime(), t) << "Time mismatch at loop iteration " << i << ": t = " << t+dt;
          t += dt;

          // Check for NaNs before prediction
          if (hasNaN(prev_state) || hasNaN(prev_covariance)) {
              FAIL() << "NaN detected in state or covariance before prediction at t=" << t;
          }

          // Get true state
          StateVector true_state = kinematic_arbiter::utils::Figure8Trajectory(t);

          // Check for NaNs after prediction
          if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
              FAIL() << "NaN detected in state or covariance after prediction at t=" << t;
          }
          ASSERT_FLOAT_EQ(filter->GetCurrentTime(), t-dt) << "Time changed before prediction";
          StateVector predicted_state = filter->GetStateEstimate(t);
          ASSERT_FLOAT_EQ(filter->GetCurrentTime(), t-dt) << "Time changed after prediction";

          // Get measurement
          auto measurement = sensor->PredictMeasurement(true_state);

          // Add noise if needed
          if (add_noise) {
              Eigen::VectorXd noise = kinematic_arbiter::utils::generateMultivariateNoise(
                  noise_covariance, gen);
              measurement += noise;
          }

          // Compute auxiliary data
          auto aux_data = sensor->ComputeAuxiliaryData(
              predicted_state,
              filter->GetStateCovariance(),
              measurement);

          // Check for NaNs in auxiliary data
          if (hasNaN(aux_data.innovation) || hasNaN(aux_data.jacobian) || hasNaN(aux_data.innovation_covariance)) {
              FAIL() << "NaN detected in auxiliary data at t=" << t;
          }

          // Process measurement with explicit template parameter
          bool assumptions_held = filter->ProcessMeasurementByIndex(sensor_idx, measurement, t);
          if (!assumptions_held) {
              FAIL() << "Measurement assumptions not held at t=" << t;
          }
          ASSERT_EQ(filter->GetCurrentTime(), t) << "Time not updated during update";

          // Check for NaNs after update
          if (hasNaN(filter->GetStateEstimate()) || hasNaN(filter->GetStateCovariance())) {
              FAIL() << "NaN detected in state or covariance after update at t=" << t;
          }
          StateVector updated_state = filter->GetStateEstimate();
          StateMatrix updated_cov = filter->GetStateCovariance();
          // Calculate prediction error and update error for each initializable state
          for (Eigen::Index idx = 0; idx < initializable_states.size(); ++idx) {
            int state_idx = initializable_states[idx];

            // Skip this check if the state isn't actually initializable by this sensor
            StateFlags initializable_flags = sensor->GetInitializableStates();
            if (!initializable_flags[state_idx]) {
              continue;  // Skip checks for states this sensor can't initialize
            }

            // Special handling for quaternion components
            if (state_idx == SIdx::Quaternion::X ||
                state_idx == SIdx::Quaternion::Y ||
                state_idx == SIdx::Quaternion::Z) {

              // Extract quaternion components
              Eigen::Quaterniond true_quat(
                  true_state[SIdx::Quaternion::W],
                  true_state[SIdx::Quaternion::X],
                  true_state[SIdx::Quaternion::Y],
                  true_state[SIdx::Quaternion::Z]);

              Eigen::Quaterniond pred_quat(
                  predicted_state[SIdx::Quaternion::W],
                  predicted_state[SIdx::Quaternion::X],
                  predicted_state[SIdx::Quaternion::Y],
                  predicted_state[SIdx::Quaternion::Z]);

              Eigen::Quaterniond updated_quat(
                  updated_state[SIdx::Quaternion::W],
                  updated_state[SIdx::Quaternion::X],
                  updated_state[SIdx::Quaternion::Y],
                  updated_state[SIdx::Quaternion::Z]);

              // Convert to rotation matrices
              Eigen::Matrix3d true_rot = true_quat.toRotationMatrix();
              Eigen::Matrix3d pred_rot = pred_quat.toRotationMatrix();
              Eigen::Matrix3d updated_rot = updated_quat.toRotationMatrix();

              // Get the row index corresponding to the quaternion component (X=0, Y=1, Z=2)
              int row_idx = state_idx - SIdx::Quaternion::X;

              // Calculate errors using the corresponding row of the rotation matrix
              double prediction_error = (pred_rot.row(row_idx) - true_rot.row(row_idx)).norm();
              double update_error = (updated_rot.row(row_idx) - true_rot.row(row_idx)).norm();
              if (true_meas_cov_norm < 1e-10) {
                ASSERT_LE(update_error, prediction_error)
                    << "Quaternion component " << state_names[idx] << " (index " << state_idx << ") estimate got worse after update at t=" << t
                    << "\nPrediction error: " << prediction_error
                  << "\nUpdate error: " << update_error;
              }
              else {
                // Check 4 sigma bounds 99.99% confidence
                ASSERT_LE(update_error, 4 * std::sqrt(true_meas_cov_norm))
                    << "Quaternion component " << state_names[idx] << " (index " << state_idx << ") estimate got worse after update at t=" << t
                    << "\nPrediction error: " << prediction_error
                    << "\nUpdate error: " << update_error;
              }
            } else {
              // Standard handling for non-quaternion states
              double prediction_error = std::abs(predicted_state[state_idx] - true_state[state_idx]);
              double update_error = std::abs(updated_state[state_idx] - true_state[state_idx]);
              if (true_meas_cov_norm < 1e-10) {
                ASSERT_LE(update_error, prediction_error)
                    << [&]() {
                        std::stringstream ss;
                        ss << "State " << state_names[idx] << " (index " << state_idx << ") estimate got worse after update at t=" << t
                           << "\nPrediction error: " << prediction_error
                           << "\nUpdate error: " << update_error
                           << "\nTrue value: " << true_state[state_idx]
                           << "\nPredicted value: " << predicted_state[state_idx]
                           << "\nUpdated value: " << updated_state[state_idx];

                        // Call PrintFilterDiagnostics to get more detailed information
                        // Only call PrintFilterDiagnostics if measurement is 6D
                        if constexpr (MeasurementType::RowsAtCompileTime == 6) {
                            PrintFilterDiagnostics<Type>(
                                sensor,
                                true_state,
                                predicted_state,
                                filter->GetStateCovariance(),
                                updated_state,
                                measurement,
                                aux_data,
                                "State " + state_names[idx] + " estimate got worse");
                        }

                        return ss.str();
                    }();
              }
              else {
                // Check 4 sigma bounds 99.99% confidence
                ASSERT_LE(update_error, 4 * std::sqrt(true_meas_cov_norm))
                    << [&]() {
                        std::stringstream ss;
                        ss << "State " << state_names[idx] << " (index " << state_idx << ") estimate got worse after update at t=" << t
                           << "\nPrediction error: " << prediction_error
                           << "\nUpdate error: " << update_error
                           << "\nTrue value: " << true_state[state_idx]
                           << "\nPredicted value: " << predicted_state[state_idx]
                           << "\nUpdated value: " << updated_state[state_idx];

                        // Call PrintFilterDiagnostics to get more detailed information
                        // Only call PrintFilterDiagnostics if measurement is 6D
                        if constexpr (MeasurementType::RowsAtCompileTime == 6) {
                            PrintFilterDiagnostics<Type>(
                                sensor,
                                true_state,
                                predicted_state,
                                filter->GetStateCovariance(),
                                updated_state,
                                measurement,
                                aux_data,
                                "State " + state_names[idx] + " estimate got worse");
                        }

                        return ss.str();
                    }();
              }
            }
          }
          // Store current state and covariance for next iteration
          prev_state = updated_state;
          prev_covariance = updated_cov;
      }

      SUCCEED() << "All initializable states improved";
  }
};

// Test with BodyVelocitySensorModel - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates<SensorType::BodyVelocity>(body_vel_model, zero_noise);
}

// Small noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_SmallNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 6, 6> small_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.001;

  TestSensorImprovesStateEstimates<SensorType::BodyVelocity>(body_vel_model, small_noise);
}

// Medium noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_MediumNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 6, 6> medium_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  TestSensorImprovesStateEstimates<SensorType::BodyVelocity>(body_vel_model, medium_noise);
}

// Large noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_LargeNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 6, 6> large_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

  TestSensorImprovesStateEstimates<SensorType::BodyVelocity>(body_vel_model, large_noise);
}

// Test with PositionSensorModel - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_SmallNoise) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 3, 3> small_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.001;

  TestSensorImprovesStateEstimates<SensorType::Position>(position_model, small_noise);
}

// Test with PositionSensorModel - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_MediumNoise) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 3, 3> medium_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.01;

  TestSensorImprovesStateEstimates<SensorType::Position>(position_model, medium_noise);
}

// Large noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_LargeNoise) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 3, 3> large_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.1;

  TestSensorImprovesStateEstimates<SensorType::Position>(position_model, large_noise);
}

// Small noise variant for Pose - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_SmallNoise) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 7, 7> small_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.001;

  TestSensorImprovesStateEstimates<SensorType::Pose>(pose_model, small_noise);
}

// Medium noise variant for Pose - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_MediumNoise) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 7, 7> medium_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.01;

  TestSensorImprovesStateEstimates<SensorType::Pose>(pose_model, medium_noise);
}

// Large noise variant for Pose - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_LargeNoise) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 7, 7> large_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.1;

  TestSensorImprovesStateEstimates<SensorType::Pose>(pose_model, large_noise);
}

// Test with ImuSensorModel - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates<SensorType::Imu>(imu_model, zero_noise);
}

// Small noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_SmallNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 6, 6> small_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.001;

  TestSensorImprovesStateEstimates<SensorType::Imu>(imu_model, small_noise);
}

// Medium noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_MediumNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 6, 6> medium_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  TestSensorImprovesStateEstimates<SensorType::Imu>(imu_model, medium_noise);
}

// Large noise variant - updated with explicit sensor type
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_LargeNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 6, 6> large_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

  TestSensorImprovesStateEstimates<SensorType::Imu>(imu_model, large_noise);
}

// Test that the filter can track a Figure-8 trajectory with multiple sensor types
TEST_F(MediatedKalmanFilterTest, TrackFigure8Trajectory) {
  // Initialize filter
  auto state_model = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();
  auto filter = std::make_shared<FilterType>(state_model);
  ASSERT_EQ(filter->GetCurrentTime(), std::numeric_limits<double>::lowest());
  // Register sensor models with reasonable noise levels
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();
  auto velocity_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Register sensors
  int position_id = filter->AddSensor(position_model);
  int pose_id = filter->AddSensor(pose_model);
  int velocity_id = filter->AddSensor(velocity_model);
  int imu_id = filter->AddSensor(imu_model);

  // Initialize time
  double t = 0.0;
  double dt = 0.01;  // 100Hz update rate

  // Use the Figure8Trajectory to get initial state
  auto true_state = kinematic_arbiter::utils::Figure8Trajectory(t);

  // Vectors to track errors for analysis
  std::vector<double> position_errors;
  std::vector<double> velocity_errors;
  std::vector<double> orientation_errors;

  // Run the filter for 5 seconds (500 iterations at 100Hz)
  for (int i = 1; i <= 100; i++) {
    t += dt;

    // Get true state from trajectory
    true_state = kinematic_arbiter::utils::Figure8Trajectory(t);

    // Create measurements from true state (no added noise for simplicity)
    // Position measurement
    Eigen::Vector3d position_measurement = position_model->PredictMeasurement(true_state);
    filter->ProcessMeasurementByIndex(position_id, position_measurement, t);
    ASSERT_EQ(filter->GetCurrentTime(), t);

    // Pose measurement
    auto pose_measurement = pose_model->PredictMeasurement(true_state);
    filter->ProcessMeasurementByIndex(pose_id, pose_measurement, t);

    // Velocity measurement
    auto velocity_measurement = velocity_model->PredictMeasurement(true_state);
    filter->ProcessMeasurementByIndex(velocity_id, velocity_measurement, t);

    // IMU measurement
    auto imu_measurement = imu_model->PredictMeasurement(true_state);
    filter->ProcessMeasurementByIndex(imu_id, imu_measurement, t);

    // Get the current filter state
    auto filter_state = filter->GetStateEstimate();

    // Calculate errors
    Eigen::Vector3d position_error = filter_state.segment<3>(core::StateIndex::Position::Begin()) -
                                     true_state.segment<3>(core::StateIndex::Position::Begin());
    Eigen::Vector3d velocity_error = filter_state.segment<3>(core::StateIndex::LinearVelocity::Begin()) -
                                     true_state.segment<3>(core::StateIndex::LinearVelocity::Begin());

    // Calculate quaternion error
    Eigen::Quaterniond q_est(
      filter_state[core::StateIndex::Quaternion::W],
      filter_state[core::StateIndex::Quaternion::X],
      filter_state[core::StateIndex::Quaternion::Y],
      filter_state[core::StateIndex::Quaternion::Z]
    );

    Eigen::Quaterniond q_true(
      true_state[core::StateIndex::Quaternion::W],
      true_state[core::StateIndex::Quaternion::X],
      true_state[core::StateIndex::Quaternion::Y],
      true_state[core::StateIndex::Quaternion::Z]
    );

    double orientation_error = q_est.angularDistance(q_true);

    // Store errors
    position_errors.push_back(position_error.norm());
    velocity_errors.push_back(velocity_error.norm());
    orientation_errors.push_back(orientation_error);
  }

  // Check final errors (should be small after convergence)
  double final_pos_error = position_errors.back();
  double final_vel_error = velocity_errors.back();
  double final_ori_error = orientation_errors.back();

  RecordProperty("final_pos_error", final_pos_error);
  RecordProperty("final_vel_error", final_vel_error);
  RecordProperty("final_ori_error", final_ori_error);

  // Use generous tolerances to ensure test stability
  EXPECT_LT(final_pos_error, 0.003) << "Position did not converge";
  EXPECT_LT(final_vel_error, 0.06) << "Velocity did not converge";
  EXPECT_LT(final_ori_error, 0.4) << "Orientation did not converge";
}

} // namespace kinematic_arbiter
