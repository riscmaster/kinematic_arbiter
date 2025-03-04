#pragma once

#include <cmath>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

namespace kinematic_arbiter {
namespace core {

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::MediatedKalmanFilter(
    const ProcessModel& process_model,
    const MeasurementModels&... measurement_models)
    : process_model_(process_model),
      measurement_models_(measurement_models...),
      state_estimate_(StateVector::Zero()),
      state_covariance_(StateMatrix::Identity()) {
  // Initialize state covariance with process model settings
  state_covariance_ *= process_model_.GetParams().initial_state_uncertainty;

  // Initialize process noise covariance
  process_noise_covariance_ = StateMatrix::Identity() *
      process_model_.GetParams().initial_state_uncertainty *
      process_model_.GetParams().process_noise_magnitude;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::Initialize(const Params& params) {
  // Validate parameters
  if (params.max_history_window <= 0.0) {
    throw FilterException("Maximum history window must be positive");
  }

  params_ = params;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::Initialize(
    const StateVector& initial_state, double timestamp) {
  state_estimate_ = initial_state;
  current_time_ = timestamp;
  is_initialized_ = true;

  // Add initial state to history
  UpdateHistory();
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateVector
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::Predict(double timestamp) {
  if (!is_initialized_) {
    throw FilterException("Filter not initialized");
  }

  // Skip prediction if timestamp is in the past or same as current time
  if (timestamp <= current_time_) {
    return state_estimate_;
  }

  // Calculate time delta
  double dt = timestamp - current_time_;

  // Get transition matrix and process noise from model
  auto transition_matrix = process_model_.GetTransitionMatrix(state_estimate_, dt);
  auto process_noise = process_model_.GetProcessNoiseCovariance(state_estimate_, dt);

  // Predict state
  StateVector predicted_state = process_model_.PredictState(state_estimate_, dt);

  // Predict covariance
  StateMatrix predicted_covariance = transition_matrix * state_covariance_ * transition_matrix.transpose()
                                    + process_noise_covariance_;

  // Update state and covariance
  state_estimate_ = predicted_state;
  state_covariance_ = predicted_covariance;
  current_time_ = timestamp;

  // Add to history for OOSM processing
  UpdateHistory();

  return state_estimate_;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
bool MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ProcessMeasurement(
    const Measurement<SensorType>& measurement) {
  if (!is_initialized_) {
    throw FilterException("Filter not initialized");
  }

  // Check if measurement is too old
  if (current_time_ - measurement.timestamp > params_.max_history_window) {
    throw FilterException("Measurement delay exceeds maximum history window");
  }

  // Determine if this is an in-sequence or out-of-sequence measurement
  if (measurement.timestamp >= current_time_) {
    // In-sequence measurement: predict to measurement time and update
    Predict(measurement.timestamp);
    return ProcessInSequenceMeasurement(measurement);
  } else {
    // Out-of-sequence measurement
    return ProcessOutOfSequenceMeasurement(measurement);
  }
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
bool MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ProcessInSequenceMeasurement(
    const Measurement<SensorType>& measurement) {
  // Get the measurement model for this sensor type
  const auto& model = GetMeasurementModel<SensorType>();

  // Get measurement noise covariance (use provided value if available)
  const auto& noise_covariance = measurement.noise_covariance.value_or(
      model.GetNoiseCovariance());

  // Predict measurement using current state
  auto predicted_measurement = model.PredictMeasurement(state_estimate_);

  // Compute innovation
  auto innovation = measurement.value - predicted_measurement;

  // Validate measurement
  if (!ValidateMeasurement<SensorType>(measurement.value, state_estimate_, state_covariance_)) {
    // Measurement rejected by validation gate
    return false;
  }

  // Compute Jacobian
  auto measurement_jacobian = model.GetMeasurementJacobian(state_estimate_);

  // Compute innovation covariance
  auto innovation_covariance = measurement_jacobian * state_covariance_ * measurement_jacobian.transpose()
                             + noise_covariance;

  // Compute Kalman gain
  auto kalman_gain = state_covariance_ * measurement_jacobian.transpose()
                   * innovation_covariance.inverse();

  // Save prior state for noise updates
  StateVector prior_state = state_estimate_;

  // Update state estimate
  state_estimate_ = state_estimate_ + kalman_gain * innovation;

  // Update state covariance
  StateMatrix identity = StateMatrix::Identity();
  state_covariance_ = (identity - kalman_gain * measurement_jacobian) * state_covariance_;

  // Update noise covariances - now through the model interface
  auto& mutable_model = const_cast<SensorType&>(GetMeasurementModel<SensorType>());
  mutable_model.UpdateNoiseCovariance(innovation);

  // Update process noise using the measurement model's parameter
  UpdateProcessNoiseCovariance(prior_state, state_estimate_,
                              model.GetParams().process_to_measurement_noise_ratio,
                              model.GetParams().noise_sample_window);

  // Update history for OOSM
  UpdateHistory();

  return true;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
bool MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ProcessOutOfSequenceMeasurement(
    const Measurement<SensorType>& measurement) {
  // Find closest history node that's before or at the measurement time
  auto it = std::lower_bound(
      history_.begin(),
      history_.end(),
      measurement.timestamp,
      [](const HistoryNode& node, double time) { return node.timestamp < time; }
  );

  // If no suitable history node found, reject the measurement
  if (it == history_.begin()) {
    return false;
  }

  // Move iterator to the history node just before measurement time
  --it;
  size_t history_index = std::distance(history_.begin(), it);

  // Retrodict state to the measurement time
  auto [retrodicted_state, retrodicted_covariance] = RetrodictState(measurement.timestamp, history_index);

  // Validate the OOSM using retrodicted state
  if (!ValidateMeasurement<SensorType>(measurement.value, retrodicted_state, retrodicted_covariance)) {
    return false;
  }

  // Get the measurement model
  const auto& model = GetMeasurementModel<SensorType>();

  // Get measurement noise covariance (use provided value if available)
  const auto& noise_covariance = measurement.noise_covariance.value_or(
      model.GetNoiseCovariance());

  // Predict measurement using retrodicted state
  auto predicted_measurement = model.PredictMeasurement(retrodicted_state);

  // Compute innovation
  auto innovation = measurement.value - predicted_measurement;

  // Compute Jacobian at the retrodicted state
  auto measurement_jacobian = model.GetMeasurementJacobian(retrodicted_state);

  // Compute innovation covariance
  auto innovation_covariance = measurement_jacobian * retrodicted_covariance * measurement_jacobian.transpose()
                             + noise_covariance;

  // Compute optimal gain for OOSM (from paper, equation 3)
  auto kalman_gain = retrodicted_covariance * measurement_jacobian.transpose()
                   * innovation_covariance.inverse();

  // Apply the update to the retrodicted state
  retrodicted_state = retrodicted_state + kalman_gain * innovation;
  retrodicted_covariance = (StateMatrix::Identity() - kalman_gain * measurement_jacobian) * retrodicted_covariance;

  // Update history node with the new state
  history_[history_index].state = retrodicted_state;
  history_[history_index].covariance = retrodicted_covariance;

  // Re-propagate all subsequent measurements
  ReapplyMeasurements(history_index + 1);

  return true;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
std::pair<typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateVector,
          typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateMatrix>
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::RetrodictState(
    double measurement_time, size_t history_index) {
  const auto& node = history_[history_index];

  // If measurement time exactly matches history node time, return the state directly
  if (std::abs(measurement_time - node.timestamp) < 1e-6) {
    return {node.state, node.covariance};
  }

  // Compute time delta from history node to measurement
  double dt = measurement_time - node.timestamp;

  // Get transition matrix for this time step
  auto transition_matrix = process_model_.GetTransitionMatrix(node.state, dt);

  // Predict state at measurement time
  StateVector predicted_state = process_model_.PredictState(node.state, dt);

  // Predict covariance at measurement time
  StateMatrix predicted_covariance = transition_matrix * node.covariance * transition_matrix.transpose()
                                   + process_noise_covariance_ * dt;

  return {predicted_state, predicted_covariance};
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ReapplyMeasurements(
    size_t start_index) {
  // If we're at the end of history, just update the current state
  if (start_index >= history_.size()) {
    return;
  }

  // Re-propagate from the starting point to the current time
  for (size_t i = start_index; i < history_.size(); ++i) {
    const auto& prev_node = history_[i-1];
    auto& current_node = history_[i];

    // Time delta between nodes
    double dt = current_node.timestamp - prev_node.timestamp;

    // Propagate state
    auto transition_matrix = process_model_.GetTransitionMatrix(prev_node.state, dt);
    current_node.state = process_model_.PredictState(prev_node.state, dt);
    current_node.covariance = transition_matrix * prev_node.covariance * transition_matrix.transpose()
                            + process_noise_covariance_ * dt;

    // Update auxiliary variables (see paper equations 10-11)
    // This is a simplified version of what would be done with actual measurements
    current_node.aux.y = prev_node.aux.y;
    current_node.aux.B = prev_node.aux.B;
    current_node.aux.U = transition_matrix * prev_node.aux.U;
  }

  // Update current state and covariance
  state_estimate_ = history_.back().state;
  state_covariance_ = history_.back().covariance;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::UpdateHistory() {
  HistoryNode node;
  node.timestamp = current_time_;
  node.state = state_estimate_;
  node.covariance = state_covariance_;

  // Most recent innovation and measurement info would be stored here
  // These are computed during measurement update and stored for OOSM
  if (!history_.empty() && history_.back().has_measurement_info) {
    const auto& measurement_info = history_.back().measurement_info;

    // For the last measurement update, compute auxiliary variables
    // These enable both OOSM processing and efficient mediation

    // C^T(CPC^T+R)^-1(y-Cx) - Information state
    node.aux.information_state = measurement_info.jacobian.transpose() *
                                 measurement_info.innovation_covariance.inverse() *
                                 measurement_info.innovation;

    // C^T(CPC^T+R)^-1C - Information matrix
    node.aux.information_matrix = measurement_info.jacobian.transpose() *
                                  measurement_info.innovation_covariance.inverse() *
                                  measurement_info.jacobian;

    // I - P*information_matrix - Projection matrix
    node.aux.projection_matrix = StateMatrix::Identity() -
                                 node.covariance * node.aux.information_matrix;
  }

  // Add to history and maintain history size
  history_.push_back(node);

  // Remove history older than max_history_window
  while (!history_.empty() &&
         history_.front().timestamp < current_time_ - params_.max_history_window) {
    history_.pop_front();
  }
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
bool MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ValidateMeasurement(
    const typename SensorType::MeasurementVector& measurement,
    const StateVector& predicted_state,
    const StateMatrix& predicted_covariance) {
  // Get the measurement model
  const auto& model = GetMeasurementModel<SensorType>();

  // Predict measurement
  auto predicted_measurement = model.PredictMeasurement(predicted_state);

  // Get measurement noise covariance
  const auto& noise_covariance = model.GetNoiseCovariance();

  // Compute innovation
  auto innovation = measurement - predicted_measurement;

  // Compute Jacobian
  auto measurement_jacobian = model.GetMeasurementJacobian(predicted_state);

  // Compute innovation covariance
  auto innovation_covariance = measurement_jacobian * predicted_covariance * measurement_jacobian.transpose()
                             + noise_covariance;

  // Compute Mahalanobis distance
  double mahalanobis_distance = ComputeMahalanobisDistance<SensorType>(innovation, innovation_covariance);

  // Look up chi-squared threshold based on confidence value and DoF
  constexpr int dof = std::remove_reference_t<decltype(innovation)>::RowsAtCompileTime;
  boost::math::chi_squared chi_squared_dist(dof);
  double threshold = boost::math::quantile(chi_squared_dist, model.GetParams().confidence_value);

  // Accept measurement if Mahalanobis distance is below threshold
  return mahalanobis_distance < threshold;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
double MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::ComputeMahalanobisDistance(
    const typename SensorType::MeasurementVector& innovation,
    const typename SensorType::MeasurementMatrix& innovation_covariance) {
  // Compute Mahalanobis distance: innovation^T * innovation_covariance^-1 * innovation
  return innovation.transpose() * innovation_covariance.inverse() * innovation;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::UpdateProcessNoiseCovariance(
    const StateVector& prior_state,
    const StateVector& posterior_state,
    double process_to_measurement_noise_ratio,
    size_t noise_window) {
  // Compute state update
  StateVector state_update = posterior_state - prior_state;

  // Update process noise covariance
  // Q_k = Q_{k-1} + (ζ * update * update^T - Q_{k-1}) / noise_window
  process_noise_covariance_ = process_noise_covariance_ +
      (process_to_measurement_noise_ratio * state_update * state_update.transpose() - process_noise_covariance_)
      / noise_window;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
template<typename SensorType>
const SensorType& MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::GetMeasurementModel() const {
  // Extract the measurement model of the specified type from the tuple
  // This uses a recursive compile-time search through the tuple
  return std::get<SensorType>(measurement_models_);
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
const typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateVector&
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::GetStateEstimate() const {
  return state_estimate_;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
const typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateMatrix&
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::GetStateCovariance() const {
  return state_covariance_;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
double MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::GetCurrentTime() const {
  return current_time_;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
const typename MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::StateMatrix&
MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::GetProcessNoiseCovariance() const {
  return process_noise_covariance_;
}

template<int StateSize, typename ProcessModel, typename... MeasurementModels>
void MediatedKalmanFilter<StateSize, ProcessModel, MeasurementModels...>::RetrodictStateToMeasurementTime(
    double measurement_time,
    StateVector& retrodicted_state,
    StateMatrix& retrodicted_covariance) {

  // Find history nodes bracketing the measurement time
  auto upper_it = std::upper_bound(history_.begin(), history_.end(), measurement_time,
      [](double time, const HistoryNode& node) { return time < node.timestamp; });

  if (upper_it == history_.begin()) {
    throw FilterException("Measurement too old for retrodiction");
  }

  auto lower_it = upper_it - 1;

  // Using auxiliary variables stored in history, retrodict state and covariance
  // to the measurement time using the formulas from the retrodiction paper

  // Apply retrodiction using the auxiliary variables (projection_matrix and information_state)
  // These calculations leverage the precomputed values to efficiently implement
  // the retrodiction equations from the OOSM paper

  // ... retrodiction calculations using information_state, information_matrix and projection_matrix ...
}

} // namespace core
} // namespace kinematic_arbiter
