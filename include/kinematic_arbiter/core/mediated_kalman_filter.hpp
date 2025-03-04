#pragma once

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
 * @brief Template class for a Mediated Kalman Filter
 *
 * Implements the algorithm described in "Mediated Kalman Filter" by Spencer Maughan.
 * The filter maintains confidence in Kalman filter assumptions and provides simplified tuning.
 *
 * The mediation provides a rigorous framework for encapsulating and actively maintaining
 * the fundamental assumptions of the Kalman filter. By continuously validating these
 * assumptions, mediation enables the detection of both recoverable and non-recoverable
 * failures, ensuring appropriate failure mode handling.
 *
 * @tparam StateSize Dimension of the state vector
 * @tparam MeasurementSize Dimension of the measurement vector
 * @tparam StateModel Model that defines state transition and jacobian
 * @tparam MeasurementModel Model that defines measurement prediction and jacobian
 */
template<
  int StateSize,
  int MeasurementSize,
  typename StateModel,
  typename MeasurementModel
>
class MediatedKalmanFilter {
public:
  // Type definitions for matrices
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using MeasurementVector = Eigen::Matrix<double, MeasurementSize, 1>;
  using MeasurementMatrix = Eigen::Matrix<double, MeasurementSize, MeasurementSize>;
  using ObservationMatrix = Eigen::Matrix<double, MeasurementSize, StateSize>;

  /**
   * @brief Parameters for the mediated Kalman filter
   */
  struct Params {
    // Time window over which measurement/process noise will be estimated
    size_t noise_time_window = 100;

    // Process noise modifier (ζ). Lower increases confidence in the process model
    // and higher decreases confidence in process model. A value of 1.0 is neutral.
    // Range: [0, inf]
    double process_to_measurement_noise_ratio = 1.0;

    // Confidence level that measurement and estimate share the same mean
    // Used to calculate the chi-squared critical value
    // Range: (0, 1)
    double confidence_value = 0.95;

    // Uncertainty bound used to initialize the state estimate
    double initial_state_uncertainty = 1.0;
  };

  /**
   * @brief Construct a new Mediated Kalman Filter
   *
   * @param time_step Time step between filter updates in seconds
   * @param state_model Model that implements state transition
   * @param measurement_model Model that implements measurement prediction
   */
  MediatedKalmanFilter(
      double time_step,
      const StateModel& state_model,
      const MeasurementModel& measurement_model);

  /**
   * @brief Initialize the filter with parameters
   *
   * @param params Filter parameters
   * @throws FilterException if parameters are invalid
   */
  void Initialize(const Params& params);

  /**
   * @brief Initialize the filter with a starting state
   *
   * @param initial_state Initial state vector
   */
  void Initialize(const StateVector& initial_state);

  /**
   * @brief Process a new measurement and update the state estimate
   *
   * Implements the complete algorithm:
   * 1. Prediction step
   * 2. Mediation (measurement validation)
   * 3. Update step
   * 4. Measurement noise update
   * 5. Process noise update
   *
   * @param measurement New measurement (y_k)
   * @return Updated state estimate (x̂_k)
   */
  StateVector Update(const MeasurementVector& measurement);

  /**
   * @brief Get the current state estimate
   *
   * @return Current state estimate (x̂_k)
   */
  const StateVector& GetStateEstimate() const;

  /**
   * @brief Get the current state covariance
   *
   * @return Current state covariance (P̂_k)
   */
  const StateMatrix& GetStateCovariance() const;

private:
  // Core filter parameters
  const double time_step_;
  bool is_initialized_ = false;

  // Models
  StateModel state_model_;
  MeasurementModel measurement_model_;

  // Filter configuration
  size_t sample_window_;  // n: sample window for noise estimation
  double process_to_measurement_noise_ratio_;  // ζ: process noise modifier
  double critical_value_;  // χ_c: chi-squared critical value
  double initial_process_variance_;

  // State estimation
  StateVector state_estimate_;  // x̂_k: state estimate
  StateMatrix covariance_estimate_;  // P̂_k: state covariance
  StateMatrix process_covariance_;  // Q_k: process noise covariance
  MeasurementMatrix measurement_covariance_;  // R_k: measurement noise covariance

  // Internal methods for filter operation

  /**
   * @brief Prediction step of the Kalman filter
   *
   * Implements:
   * x̌_k = A_{k-1} * x̂_{k-1} + v_k
   * P̌_k = A_{k-1} * P̂_{k-1} * A_{k-1}^T + Q_k
   */
  void PredictStep();

  /**
   * @brief Update step of the Kalman filter
   *
   * Implements:
   * K_k = P̌_k * C_k^T * (C_k * P̌_k * C_k^T + R_k)^{-1}
   * x̂_k = x̌_k + K_k * (y_k - C_k * x̌_k)
   * P̂_k = (I - K_k * C_k) * P̌_k
   *
   * @param measurement Measurement vector (y_k)
   */
  void UpdateStep(const MeasurementVector& measurement);

  /**
   * @brief Validate measurement using chi-squared test
   *
   * Tests the assumption that measurement and predicted measurement share the same mean:
   * (y_k - C_k * x̌_k)^T * (C_k * P̌_k * C_k^T + R_k)^{-1} * (y_k - C_k * x̌_k) < χ_c
   *
   * @param measurement Measurement vector (y_k)
   * @param predicted_measurement Predicted measurement (C_k * x̌_k)
   * @param innovation_covariance Innovation covariance (C_k * P̌_k * C_k^T + R_k)
   * @return true if measurement is valid, false otherwise
   */
  bool ValidateMeasurement(
      const MeasurementVector& measurement,
      const MeasurementVector& predicted_measurement,
      const MeasurementMatrix& innovation_covariance) const;

  /**
   * @brief Compute Mahalanobis distance for innovation validation
   *
   * @param innovation Innovation vector (y_k - C_k * x̌_k)
   * @param innovation_covariance Innovation covariance (C_k * P̌_k * C_k^T + R_k)
   * @return Mahalanobis distance
   */
  double ComputeMahalanobisDistance(
      const MeasurementVector& innovation,
      const MeasurementMatrix& innovation_covariance) const;

  /**
   * @brief Update measurement noise covariance
   *
   * Implements:
   * R̂_k = R̂_{k-1} + ((y_k - C_k * x̂_k) * (y_k - C_k * x̂_k)^T - R̂_{k-1}) / n
   *
   * @param innovation Innovation vector (y_k - C_k * x̂_k)
   */
  void UpdateMeasurementNoise(
      const MeasurementVector& innovation);

  /**
   * @brief Update process noise covariance
   *
   * Implements:
   * Q̂_k = Q̂_{k-1} + (ζ * (x̌_k - x̂_k) * (x̌_k - x̂_k)^T - Q̂_{k-1}) / n
   *
   * @param prior_state Prior state estimate (x̌_k)
   * @param posterior_state Posterior state estimate (x̂_k)
   */
  void UpdateProcessNoise(
      const StateVector& prior_state,
      const StateVector& posterior_state);
};

} // namespace core
} // namespace kinematic_arbiter
