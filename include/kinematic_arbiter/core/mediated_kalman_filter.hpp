#pragma once

#include <stdexcept>
#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <optional>
#include <type_traits>
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
 * @brief Template class for a Mediated Kalman Filter with OOSM support
 *
 * Implements a unified algorithm that combines:
 * 1. Mediated Kalman filtering - For robust handling of potentially invalid assumptions
 * 2. OOSM processing - For optimal updates with out-of-sequence measurements
 *
 * This filter continuously validates the fundamental Kalman filter assumptions through
 * mediation and handles measurements arriving in arbitrary order through retrodiction.
 *
 * Practical applications include:
 * - Multi-sensor fusion where sensors have different reliabilities
 * - Systems with communication delays (networked robots, distributed sensing)
 * - Applications requiring fault detection and recovery
 * - Edge computing with asynchronous sensor data
 *
 * @tparam StateSize Dimension of the state vector
 * @tparam ProcessModel Model that defines state transition and jacobian
 * @tparam MeasurementModels... Variable number of measurement model types
 */
template<
  int StateSize,
  typename ProcessModel,
  typename... MeasurementModels
>
class MediatedKalmanFilter {
public:
  // Type definitions for matrices
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;

  // Forward declaration of the Measurement struct
  template<typename SensorType>
  struct Measurement;

  /**
   * @brief Parameters for OOSM handling in the filter
   */
  struct Params {
    // Maximum history length for OOSM handling in seconds
    // This is also the maximum allowed measurement delay
    double max_history_window = 1.0;
    int max_history_size = 100; // Maximum number of history nodes to store
  };

  /**
   * @brief History node for OOSM processing and mediation
   *
   * Stores all necessary information to perform retrodiction and mediation
   */
  struct HistoryNode {
    double timestamp;               // When this state was recorded
    StateVector state;              // State estimate at this time
    StateMatrix covariance;         // State covariance at this time

    // Unified auxiliary variables for both OOSM and mediation
    struct AuxiliaryVariables {
      // For OOSM retrodiction and efficient Kalman updates
      StateVector information_state;     // Related to state innovation contribution (y in OOSM paper)
                                         // Computed as: C^T(CPC^T+R)^-1(y-Cx)

      StateMatrix information_matrix;    // Information contribution from measurement (B in OOSM paper)
                                         // Computed as: C^T(CPC^T+R)^-1C

      StateMatrix projection_matrix;     // State projection through measurement update (U in OOSM paper)
                                         // Computed as: I - P*information_matrix
                                         // Enables efficient retrodiction for OOSM processing
    };
    AuxiliaryVariables aux;
  };

  /**
   * @brief Measurement data with timestamp for a specific sensor type
   *
   * @tparam SensorType The measurement model type
   */
  template<typename SensorType>
  struct Measurement {
    double timestamp;               // When the measurement was taken

    // The actual measurement vector
    typename SensorType::MeasurementVector value;

    // Optional override for the measurement noise covariance
    std::optional<typename SensorType::MeasurementMatrix> noise_covariance;
  };

  /**
   * @brief Create a new Mediated Kalman Filter
   *
   * @param process_model Model that implements state transition
   * @param measurement_models... Models that implement measurement predictions
   */
  MediatedKalmanFilter(
      const ProcessModel& process_model,
      const MeasurementModels&... measurement_models);

  /**
   * @brief Initialize the filter with OOSM parameters
   *
   * @param params Filter parameters
   * @throws FilterException if parameters are invalid
   */
  void Initialize(const Params& params);

  /**
   * @brief Initialize the filter with a starting state
   *
   * @param initial_state Initial state vector
   * @param timestamp Initial timestamp
   */
  void Initialize(const StateVector& initial_state, double timestamp = 0.0);

  /**
   * @brief Predict state forward to the specified time
   *
   * Implements the prediction step of the Kalman filter:
   * x̌_k = A_{k-1} * x̂_{k-1} + v_k
   * P̌_k = A_{k-1} * P̂_{k-1} * A_{k-1}^T + Q_k
   *
   * @param timestamp Time to predict to
   * @return Updated state estimate
   * @throws FilterException if filter is not initialized
   */
  StateVector Predict(double timestamp);

  /**
   * @brief Process a measurement from any registered sensor
   *
   * This method handles both in-sequence and out-of-sequence measurements:
   * 1. For in-sequence: Standard mediated Kalman update
   * 2. For out-of-sequence: Uses retrodiction algorithm for optimal updates
   *
   * @tparam SensorType Type of sensor providing the measurement
   * @param measurement Measurement data with timestamp
   * @return true if measurement was used, false if rejected by validation
   * @throws FilterException if measurement delay exceeds max_measurement_delay
   */
  template<typename SensorType>
  bool ProcessMeasurement(const Measurement<SensorType>& measurement);

  /**
   * @brief Get the current state estimate
   *
   * @return Current state estimate
   */
  const StateVector& GetStateEstimate() const;

  /**
   * @brief Get the current state covariance
   *
   * @return Current state covariance
   */
  const StateMatrix& GetStateCovariance() const;

  /**
   * @brief Get the timestamp of the current state estimate
   *
   * @return Current timestamp
   */
  double GetCurrentTime() const;

  /**
   * @brief Get the process noise covariance matrix
   *
   * @return Process noise covariance
   */
  const StateMatrix& GetProcessNoiseCovariance() const;

  /**
   * @brief Get the measurement noise covariance for a specific sensor
   *
   * @tparam SensorType Type of sensor
   * @return Measurement noise covariance for the sensor
   */
  template<typename SensorType>
  const typename SensorType::MeasurementMatrix& GetMeasurementNoiseCovariance() const;

private:
  // Filter state
  double current_time_ = 0.0;
  bool is_initialized_ = false;
  StateVector state_estimate_;
  StateMatrix state_covariance_;
  StateMatrix process_noise_covariance_;

  // Filter parameters
  Params params_;

  // Models
  ProcessModel process_model_;
  std::tuple<MeasurementModels...> measurement_models_;

  // History for OOSM processing
  std::deque<HistoryNode> history_;

  /**
   * @brief Get the measurement model for a specific sensor type
   *
   * Helper method to extract the right measurement model from the tuple
   *
   * @tparam SensorType The sensor type to get the model for
   * @return Reference to the measurement model
   */
  template<typename SensorType>
  const SensorType& GetMeasurementModel() const;

  /**
   * @brief Add current state to history for OOSM processing
   *
   * Updates the auxiliary variables needed for retrodiction algorithm
   * and maintains history window based on configuration.
   */
  void UpdateHistory();

  /**
   * @brief Prune history nodes older than max_history_window
   */
  void PruneHistory();

  /**
   * @brief Process an in-sequence measurement (latest in time)
   *
   * Standard mediated Kalman filter update:
   * 1. Validate measurement using chi-squared test
   * 2. If valid, update state and covariance
   * 3. Update noise estimates
   *
   * @tparam SensorType Type of sensor providing the measurement
   * @param measurement Measurement data
   * @return true if measurement was used, false if rejected
   */
  template<typename SensorType>
  bool ProcessInSequenceMeasurement(const Measurement<SensorType>& measurement);

  /**
   * @brief Process an out-of-sequence measurement
   *
   * Implements Algorithm I, Case II from the retrodiction paper:
   * 1. Find appropriate history node
   * 2. Retrodict state to measurement time
   * 3. Compute optimal update
   * 4. Re-process all subsequent measurements
   *
   * @tparam SensorType Type of sensor providing the measurement
   * @param measurement Out-of-sequence measurement
   * @return true if measurement was used, false if rejected
   */
  template<typename SensorType>
  bool ProcessOutOfSequenceMeasurement(const Measurement<SensorType>& measurement);

  /**
   * @brief Retrodict state to the time of a measurement
   *
   * Computes the optimal state estimate at a past time point using
   * stored auxiliary variables (y, B, U) as defined in the retrodiction paper.
   *
   * @param measurement_time Time to retrodict to
   * @param history_index Index of nearest history node
   * @return Pair of retrodicted state and covariance
   */
  std::pair<StateVector, StateMatrix> RetrodictState(double measurement_time, size_t history_index);

  /**
   * @brief Validate a measurement using chi-squared test
   *
   * Tests if innovation is within expected bounds based on confidence_value.
   * This is a key part of the mediation algorithm to detect invalid measurements.
   *
   * @tparam SensorType Type of sensor
   * @param measurement The measurement to validate
   * @param predicted_state State prediction at measurement time
   * @param predicted_covariance Covariance prediction at measurement time
   * @return true if measurement passes validation, false otherwise
   */
  template<typename SensorType>
  bool ValidateMeasurement(
      const typename SensorType::MeasurementVector& measurement,
      const StateVector& predicted_state,
      const StateMatrix& predicted_covariance);

  /**
   * @brief Compute Mahalanobis distance for innovation validation
   *
   * Used to quantify how "surprising" a measurement is relative to the prediction.
   *
   * @tparam SensorType Type of sensor
   * @param innovation Innovation vector (z - h(x))
   * @param innovation_covariance Covariance of the innovation
   * @return Mahalanobis distance
   */
  template<typename SensorType>
  double ComputeMahalanobisDistance(
      const typename SensorType::MeasurementVector& innovation,
      const typename SensorType::MeasurementMatrix& innovation_covariance);

  /**
   * @brief Update measurement noise covariance based on innovations
   *
   * Adaptive estimation of measurement noise to improve filter robustness.
   *
   * @tparam SensorType Type of sensor
   * @param innovation Innovation vector (z - h(x))
   */
  template<typename SensorType>
  void UpdateMeasurementNoiseCovariance(
      const typename SensorType::MeasurementVector& innovation);

  /**
   * @brief Update process noise covariance based on state updates
   *
   * Adaptive estimation of process noise to improve filter robustness.
   *
   * @param prior_state State estimate before update
   * @param posterior_state State estimate after update
   */
  void UpdateProcessNoiseCovariance(
      const StateVector& prior_state,
      const StateVector& posterior_state);

  /**
   * @brief Re-apply all measurements after an OOSM update
   *
   * After processing an OOSM, we need to re-process all subsequent
   * measurements to maintain consistency.
   *
   * @param start_index Index in history to start reprocessing from
   */
  void ReapplyMeasurements(size_t start_index);
};

} // namespace core
} // namespace kinematic_arbiter

// Include the implementation
#include "kinematic_arbiter/core/mediated_kalman_filter.tpp"
