#pragma once

#include <stdexcept>
#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <optional>
#include <type_traits>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/mediation_types.hpp"

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Exception class for filter-related errors
 */
class FilterException : public std::runtime_error {
public:
  explicit FilterException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Generic measurement container with timestamp
 *
 * @tparam SensorType The sensor model type
 */
template<typename SensorType>
struct Measurement {
  using ValueType = typename SensorType::MeasurementVector;

  double timestamp;                                      // Measurement timestamp
  ValueType value;                                       // Measurement value
  std::optional<typename SensorType::MeasurementCovariance> noise_override; // Optional override for noise

  Measurement(double t, const ValueType& v)
    : timestamp(t), value(v), noise_override(std::nullopt) {}

  Measurement(double t, const ValueType& v,
              const typename SensorType::MeasurementCovariance& cov)
    : timestamp(t), value(v), noise_override(cov) {}
};

/**
 * @brief MediatedKalmanFilter implements a Kalman filter with chi-squared validation
 *
 * Leverages StateModelInterface for prediction and ProcessNoise management
 * and MeasurementModelInterface for measurement processing, validation and mediation.
 *
 * @tparam StateSize Dimension of state vector
 * @tparam ProcessModel Type implementing StateModelInterface
 * @tparam SensorModels Types implementing MeasurementModelInterface
 */
template<int StateSize, typename ProcessModel, typename... SensorModels>
class MediatedKalmanFilter {
public:
  // Type aliases
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using StateFlags = Eigen::Array<bool, StateSize, 1>;

  /**
   * @brief Constructor with process and measurement models
   */
  MediatedKalmanFilter(std::shared_ptr<ProcessModel> process_model,
                      std::shared_ptr<SensorModels>... sensor_models)
    : process_model_(process_model),
      sensor_models_(std::make_tuple(sensor_models...)),
      reference_time_(0.0),
      max_delay_window_(10.0) {}

  /**
   * @brief Process measurement from specified sensor
   */
  template<size_t SensorIndex, typename MeasurementType>
  bool ProcessMeasurement(double timestamp, const MeasurementType& measurement) {
    auto sensor_model = std::get<SensorIndex>(sensor_models_);

    // Try to initialize any states this sensor can provide
    StateFlags initializable = sensor_model->GetInitializableStates();

    // Create "not initialized" mask using select (avoids bitwise NOT)
    StateFlags not_initialized = initialized_states_.select(StateFlags::Zero(), StateFlags::Ones());

    // Use cwiseProduct for element-wise AND (states that can be initialized and aren't yet)
    StateFlags uninit_states = initializable.cwiseProduct(not_initialized);

    // Only attempt initialization for states we haven't initialized yet
    if (uninit_states.any()) {
      StateFlags new_states = sensor_model->InitializeState(
          measurement, initialized_states_, reference_state_, reference_covariance_);

      // Use cwiseMax for element-wise OR (states that were initialized before OR are newly initialized)
      initialized_states_ = initialized_states_.cwiseMax(new_states);
    }

    // Reject too-old measurements
    if (timestamp < reference_time_ - max_delay_window_) {
      return false;
    }

    double dt = timestamp - reference_time_;

    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateVector state_at_sensor_time = process_model_->PredictState(reference_state_, dt);
    StateMatrix Q_at_sensor_time = process_model_->GetProcessNoiseCovariance(dt);
    StateMatrix covariance_at_sensor_time = A * reference_covariance_ * A.transpose() + Q_at_sensor_time;

    // Apply measurement at sensor time
    if (!applyMeasurement<SensorIndex>(state_at_sensor_time, covariance_at_sensor_time, measurement, dt)) {
      return false;  // Measurement rejected
    }

    if (dt > 0.0) {
      // Update reference state and time
      reference_state_ = state_at_sensor_time;
      reference_covariance_ = covariance_at_sensor_time;
      reference_time_ = timestamp;  // Time advances for newer measurements
      return true;
    }

    // Reset reference state to reference time
    reference_state_ = process_model_->PredictState(reference_state_, -dt);
    reference_covariance_ = process_model_->GetProcessNoiseCovariance(-dt);
    return true;
  }

  /**
   * @brief Get state estimate at given time
   */
  StateVector GetStateEstimate(double timestamp = -1.0) const {
    if (timestamp < 0.0 || timestamp == reference_time_) {
      return reference_state_;
    }

    // Predict state to requested time without changing reference
    double dt = timestamp - reference_time_;
    return process_model_->PredictState(reference_state_, dt);
  }

  /**
   * @brief Get state covariance at given time
   */
  StateMatrix GetStateCovariance(double timestamp = -1.0) const {
    if (timestamp < 0.0 || timestamp == reference_time_) {
      return reference_covariance_;
    }

    // Predict covariance to requested time without changing reference
    double dt = timestamp - reference_time_;
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    return A * reference_covariance_ * A.transpose() + Q;
  }

  double GetCurrentTime() const { return reference_time_; }
  void SetMaxDelayWindow(double window) { max_delay_window_ = window; }

  /**
   * @brief Get process model
   */
  std::shared_ptr<ProcessModel> GetProcessModel() const {
    return process_model_;
  }

  // Update IsInitialized to check if any states are initialized
  bool IsInitialized() const { return reference_time_ > 0.0; }

private:
  /**
   * @brief Apply measurement update at a given state
   */
  template<size_t SensorIndex, typename MeasurementType>
  bool applyMeasurement(StateVector& state, StateMatrix& covariance, const MeasurementType& measurement, double dt = 0.0) {
    auto sensor_model = std::get<SensorIndex>(sensor_models_);

    // Store a copy of the prior state for process noise update
    StateVector prior_state = state;

    // Compute auxiliary data (innovation, Jacobian, innovation covariance)
    auto aux_data = sensor_model->ComputeAuxiliaryData(state, covariance, measurement);

    // Validate measurement using the simplified ValidateAndMediate interface
    // Note: This now handles covariance updates internally based on validation result
    bool assumptions_valid = sensor_model->ValidateAndMediate(
        state,                      // Current state
        covariance,                 // Current covariance
        measurement                // Measurement to validate
    );

    if (!assumptions_valid && sensor_model->GetValidationParams().mediation_action == MediationAction::Reject) {
      return false;  // Measurement rejected
    }

    // Compute Kalman gain using Cholesky decomposition for efficiency
    // K = P*H'*S^-1 can be computed as the solution to: S*K' = H*P'
    auto PHt = covariance * aux_data.jacobian.transpose();

    // Use Cholesky decomposition to solve for Kalman gain
    Eigen::LLT<typename std::decay_t<decltype(aux_data.innovation_covariance)>> llt_of_S(aux_data.innovation_covariance);

    // Check if decomposition succeeded (matrix is SPD)
    if (llt_of_S.info() != Eigen::Success) {
      // Add small regularization if needed
      auto regularized_S = aux_data.innovation_covariance;
      regularized_S.diagonal().array() += 1e-8;
      llt_of_S.compute(regularized_S);
    }

    // Solve for Kalman gain efficiently
    auto kalman_gain = llt_of_S.solve(PHt.transpose()).transpose();

    // Update state estimate (x = x + K*innovation)
    state += kalman_gain * aux_data.innovation;

    covariance = (StateMatrix::Identity() - kalman_gain * aux_data.jacobian) * covariance;

    // Update process noise using the proper UpdateProcessNoise interface
    process_model_->UpdateProcessNoise(
        prior_state,   // a priori state (before update)
        state,         // a posteriori state (after update)
        sensor_model->GetValidationParams().process_to_measurement_noise_ratio,  // process to measurement noise ratio (tunable)
        dt           // dt - small representative time step for this update
    );

    return true;
  }

  // Core data
  std::shared_ptr<ProcessModel> process_model_;
  std::tuple<std::shared_ptr<SensorModels>...> sensor_models_;

  // Reference state (at most recent measurement time)
  StateVector reference_state_;
  StateMatrix reference_covariance_;
  double reference_time_;
  double max_delay_window_;
  StateFlags initialized_states_;
};

} // namespace core
} // namespace kinematic_arbiter
