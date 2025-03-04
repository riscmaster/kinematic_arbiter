#pragma once

#include <tuple>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"

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
 * @brief Timestamped state representation
 */
struct TimestampedState {
  double timestamp;  // Time in seconds
  Eigen::Matrix<double, kStateSize, 1> state;
  Eigen::Matrix<double, kStateSize, kStateSize> covariance;
};

/**
 * @brief Multi-sensor Mediated Kalman Filter with retrodiction
 *
 * This filter handles:
 * 1. Multiple sensor types with different update rates
 * 2. Asynchronous measurements with timestamps
 * 3. Out-of-sequence measurements using retrodiction
 * 4. Stale measurement detection and handling
 *
 * @tparam MeasurementModels Variable number of measurement model types
 */
template<typename... MeasurementModels>
class MultiSensorMediatedKalmanFilter {
public:
  using StateVector = Eigen::Matrix<double, kStateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;

  /**
   * @brief Parameters for the multi-sensor filter
   */
  struct Params {
    // Base filter parameters
    size_t noise_time_window = 100;
    double process_to_measurement_noise_ratio = 1.0;
    double confidence_value = 0.95;
    double initial_state_uncertainty = 1.0;

    // Multi-sensor specific parameters

    // Maximum age of measurements to consider (in seconds)
    double max_measurement_age = 1.0;

    // Number of recent states to store for retrodiction
    size_t retrodiction_buffer_size = 10;
  };

  /**
   * @brief State history entry for retrodiction
   */
  struct StateHistoryEntry {
    double timestamp;
    StateVector state;
    StateMatrix covariance;
    StateMatrix transition_matrix;  // State transition matrix from previous state
  };

  /**
   * @brief Constructor for multi-sensor filter
   *
   * @param state_model Model for state prediction
   * @param measurement_models Models for each measurement type
   */
  MultiSensorMediatedKalmanFilter(
      const StateModelInterface& state_model,
      const MeasurementModels&... measurement_models);

  /**
   * @brief Initialize the filter with parameters
   *
   * @param params Filter parameters
   * @throws FilterException if parameters are invalid
   */
  void Initialize(const Params& params);

  /**
   * @brief Initialize the filter with a starting state and timestamp
   *
   * @param initial_state Initial state vector
   * @param timestamp Initial timestamp (in seconds)
   */
  void Initialize(const StateVector& initial_state, double timestamp);

  /**
   * @brief Process a measurement from a specific sensor
   *
   * This method:
   * 1. Checks if the measurement is too old (stale)
   * 2. For in-sequence measurements: performs standard Kalman update
   * 3. For out-of-sequence measurements: applies retrodiction
   *
   * @tparam SensorIndex Index of the sensor in the measurement models tuple
   * @tparam MeasurementType Type of measurement for this sensor (must derive from MeasurementBase)
   * @param measurement The measurement data with timestamp
   * @return Updated state estimate
   * @throws FilterException if measurement is too old or filter is not initialized
   */
  template<size_t SensorIndex, typename MeasurementType>
  StateVector Update(const MeasurementType& measurement);

  /**
   * @brief Get the current state estimate with timestamp
   *
   * @return Current timestamped state
   */
  TimestampedState GetCurrentState() const;

private:
  // Core filter parameters
  bool is_initialized_ = false;

  // Current state information
  TimestampedState current_state_;
  StateMatrix process_covariance_;

  // Filter configuration
  Params params_;

  // Models
  StateModelInterface state_model_;
  std::tuple<MeasurementModels...> measurement_models_;

  // Measurement noise covariances (one per sensor)
  std::tuple<typename MeasurementModels::MeasurementMatrix...> measurement_covariances_;

  // Circular buffer of recent states for retrodiction
  std::vector<StateHistoryEntry> state_history_;
  size_t history_head_ = 0;

  /**
   * @brief Predict state from one timestamp to another
   *
   * @param state_at_t1 State at starting time t1
   * @param t2 Target timestamp
   * @return Predicted state at time t2
   */
  TimestampedState PredictStateToTime(
      const TimestampedState& state_at_t1,
      double t2);

  /**
   * @brief Add current state to history buffer
   *
   * @param state_entry State entry to add to history
   */
  void AddToHistory(const StateHistoryEntry& state_entry);

  /**
   * @brief Find the closest state history entry before a given timestamp
   *
   * @param timestamp Timestamp to find history entry for
   * @return Pointer to the history entry, or nullptr if none found
   */
  const StateHistoryEntry* FindClosestHistoryEntryBefore(double timestamp) const;

  /**
   * @brief Apply retrodiction for out-of-sequence measurement
   *
   * @tparam SensorIndex Index of the sensor
   * @tparam MeasurementType Type of measurement
   * @param measurement Measurement data with timestamp
   * @return true if retrodiction was applied, false if measurement should be discarded
   */
  template<size_t SensorIndex, typename MeasurementType>
  bool ApplyRetrodiction(const MeasurementType& measurement);

  /**
   * @brief Check if a measurement is too old to process
   *
   * @param measurement_timestamp Measurement timestamp
   * @return true if measurement is stale, false otherwise
   */
  bool IsMeasurementStale(double measurement_timestamp) const;

  /**
   * @brief Compute the retrodiction gain matrix
   *
   * @param state_at_current_time Current state estimate
   * @param state_at_measurement_time State estimate at measurement time
   * @param transition_matrices Sequence of state transition matrices from measurement time to current
   * @return Retrodiction gain matrix
   */
  StateMatrix ComputeRetrodictionGain(
      const TimestampedState& state_at_current_time,
      const TimestampedState& state_at_measurement_time,
      const std::vector<StateMatrix>& transition_matrices);

  /**
   * @brief Standard Kalman update for in-sequence measurements
   *
   * @tparam SensorIndex Index of the sensor
   * @tparam MeasurementType Type of measurement
   * @param measurement Measurement data with timestamp
   * @param predicted_state State predicted to measurement time
   * @return Updated state at measurement time
   */
  template<size_t SensorIndex, typename MeasurementType>
  TimestampedState ApplyMeasurementUpdate(
      const MeasurementType& measurement,
      const TimestampedState& predicted_state);
};

} // namespace core
} // namespace kinematic_arbiter
