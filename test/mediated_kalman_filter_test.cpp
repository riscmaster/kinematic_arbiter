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

// Use the correct state index type
using SIdx = kinematic_arbiter::core::StateIndex;

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

  void PrintFilterDiagnostics(
      const std::shared_ptr<core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>>& sensor,
      const StateVector& true_state,
      const StateVector& predicted_state,
      const StateMatrix& predicted_cov,
      const StateVector& updated_state,
      const Eigen::Matrix<double, 6, 1>& measurement,
      const core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>::MeasurementAuxData& aux_data,
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

  template<typename SensorModel>
  void TestSensorImprovesStateEstimates(
      std::shared_ptr<SensorModel> sensor,
      const Eigen::MatrixXd& noise_covariance) {
      using namespace core;
      using MeasurementType = typename SensorModel::MeasurementVector;

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

      // Get initializable states from the sensor
      [[maybe_unused]] StateFlags initializable_flags = sensor->GetInitializableStates();
      [[maybe_unused]] StateVector last_good_state = prev_state;
      [[maybe_unused]] StateMatrix last_good_cov = prev_covariance;

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

      // Random number generator and normal distribution setup
      std::mt19937 gen(std::random_device{}());
      std::normal_distribution<double> normal_dist(0.0, 1.0);

      // Precompute the transformation matrix for noise if needed
      Eigen::MatrixXd transform;
      if (add_noise) {
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(noise_covariance);
          transform = eigenSolver.operatorSqrt();
      }

      // Lambda function to add noise to a measurement
      auto addNoise = [&gen, &normal_dist, &transform](const MeasurementType& measurement) -> MeasurementType {
          // Generate standard normal samples
          Eigen::VectorXd z = Eigen::VectorXd::NullaryExpr(measurement.size(), [&]() { return normal_dist(gen); });

          // Transform to desired covariance and add to measurement
          return measurement + transform * z;
      };


    auto initializable_states = sensor->GetInitializableStates();
    std::vector<std::string> state_names = kinematic_arbiter::core::GetInitializableStateNames(initializable_states);

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

        // Get measurement and add noise if needed
        auto measurement = sensor->PredictMeasurement(true_state);
        if (add_noise) {
            measurement = addNoise(measurement);
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

        // Process measurement
        bool assumptions_held = filter->ProcessMeasurementByIndex(sensor_idx, t, measurement);
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
                          PrintFilterDiagnostics(
                              std::static_pointer_cast<core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>>(sensor),
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
                          PrintFilterDiagnostics(
                              std::static_pointer_cast<core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>>(sensor),
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
// Test with BodyVelocitySensorModel
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates(body_vel_model, zero_noise);
}

// Small noise variant
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_SmallNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 6, 6> small_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.001;

  TestSensorImprovesStateEstimates(body_vel_model, small_noise);
}

// Medium noise variant
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_MediumNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 6, 6> medium_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  TestSensorImprovesStateEstimates(body_vel_model, medium_noise);
}

// Large noise variant
TEST_F(MediatedKalmanFilterTest, BodyVelocitySensorImprovesEstimates_LargeNoise) {
  auto body_vel_model = std::make_shared<kinematic_arbiter::sensors::BodyVelocitySensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 6, 6> large_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

  TestSensorImprovesStateEstimates(body_vel_model, large_noise);
}

// Test with PositionSensorModel
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 3, 3> zero_noise = Eigen::Matrix<double, 3, 3>::Zero();

  TestSensorImprovesStateEstimates(position_model, zero_noise);
}

// // Small noise variant Refine checks on covariance
// TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_SmallNoise) {
//   auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

//   // Small noise (0.001)
//   Eigen::Matrix<double, 3, 3> small_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.001;

//   TestSensorImprovesStateEstimates(position_model, small_noise);
// }

// Medium noise variant
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_MediumNoise) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 3, 3> medium_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.01;

  TestSensorImprovesStateEstimates(position_model, medium_noise);
}

// Large noise variant
TEST_F(MediatedKalmanFilterTest, PositionSensorImprovesEstimates_LargeNoise) {
  auto position_model = std::make_shared<kinematic_arbiter::sensors::PositionSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 3, 3> large_noise = Eigen::Matrix<double, 3, 3>::Identity() * 0.1;

  TestSensorImprovesStateEstimates(position_model, large_noise);
}

// Test with PoseSensorModel
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 7, 7> zero_noise = Eigen::Matrix<double, 7, 7>::Zero();

  TestSensorImprovesStateEstimates(pose_model, zero_noise);
}

// // Small noise variant Refine checks on covariance
// TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_SmallNoise) {
//   auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

//   // Small noise (0.001)
//   Eigen::Matrix<double, 7, 7> small_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.001;

//   TestSensorImprovesStateEstimates(pose_model, small_noise);
// }

// Medium noise variant
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_MediumNoise) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 7, 7> medium_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.01;

  TestSensorImprovesStateEstimates(pose_model, medium_noise);
}

// Large noise variant
TEST_F(MediatedKalmanFilterTest, PoseSensorImprovesEstimates_LargeNoise) {
  auto pose_model = std::make_shared<kinematic_arbiter::sensors::PoseSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 7, 7> large_noise = Eigen::Matrix<double, 7, 7>::Identity() * 0.1;

  TestSensorImprovesStateEstimates(pose_model, large_noise);
}

// Test with ImuSensorModel
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Zero noise for perfect measurements
  Eigen::Matrix<double, 6, 6> zero_noise = Eigen::Matrix<double, 6, 6>::Zero();

  TestSensorImprovesStateEstimates(imu_model, zero_noise);
}

// Small noise variant
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_SmallNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Small noise (0.001)
  Eigen::Matrix<double, 6, 6> small_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.001;

  TestSensorImprovesStateEstimates(imu_model, small_noise);
}

// Medium noise variant
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_MediumNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Medium noise (0.01)
  Eigen::Matrix<double, 6, 6> medium_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;

  TestSensorImprovesStateEstimates(imu_model, medium_noise);
}

// Large noise variant
TEST_F(MediatedKalmanFilterTest, ImuSensorImprovesEstimates_LargeNoise) {
  auto imu_model = std::make_shared<kinematic_arbiter::sensors::ImuSensorModel>();

  // Large noise (0.1)
  Eigen::Matrix<double, 6, 6> large_noise = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

  TestSensorImprovesStateEstimates(imu_model, large_noise);
}

} // namespace kinematic_arbiter
