#pragma once

#include <Eigen/Dense>

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Interface for measurement models used in Kalman filtering
 *
 * @tparam StateType Type of state vector
 * @tparam MeasurementType Type of measurement vector
 */
template<typename StateType, typename MeasurementType>
class MeasurementModelInterface {
public:
  // Type definitions
  using StateVector = StateType;
  using MeasurementVector = MeasurementType;
  using MeasurementMatrix = Eigen::Matrix<double, MeasurementType::RowsAtCompileTime, MeasurementType::RowsAtCompileTime>;
  using MeasurementJacobian = Eigen::Matrix<double, MeasurementType::RowsAtCompileTime, StateType::RowsAtCompileTime>;

  /**
   * @brief Parameters for measurement model mediation
   */
  struct Params {
    // Sample window size for adaptive measurement noise estimation
    size_t noise_sample_window = 100;

    // Process noise modifier (ζ). Lower values increase confidence in the process model
    // Range: [0, inf)
    double process_to_measurement_noise_ratio = 1.0;

    // Confidence level for measurement validation (used for chi-squared test)
    // Range: (0, 1)
    double confidence_value = 0.95;

    // Initial measurement noise covariance
    double initial_noise_uncertainty = 1.0;
  };

  /**
   * @brief Constructor with parameters
   *
   * @param params Parameters for this measurement model
   */
  explicit MeasurementModelInterface(const Params& params = Params())
    : params_(params),
      noise_covariance_(MeasurementMatrix::Identity() * params.initial_noise_uncertainty) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~MeasurementModelInterface() = default;

  /**
   * @brief Predict expected measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement
   */
  virtual MeasurementVector PredictMeasurement(const StateVector& state) const = 0;

  /**
   * @brief Compute the Jacobian of the measurement model
   *
   * @param state Current state estimate
   * @return Jacobian matrix (∂h/∂x)
   */
  virtual MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const = 0;

  /**
   * @brief Get the current measurement noise covariance
   *
   * @return Measurement noise covariance matrix
   */
  const MeasurementMatrix& GetNoiseCovariance() const { return noise_covariance_; }

  /**
   * @brief Get the measurement model parameters
   *
   * @return Const reference to parameters
   */
  const Params& GetParams() const { return params_; }

  /**
   * @brief Update the measurement noise covariance based on innovation
   *
   * @param innovation The measurement innovation
   */
  void UpdateNoiseCovariance(const MeasurementVector& innovation) {
    // R_k = R_{k-1} + (innovation * innovation^T - R_{k-1}) / noise_sample_window
    noise_covariance_ = noise_covariance_ +
        (innovation * innovation.transpose() - noise_covariance_) / params_.noise_sample_window;
  }

protected:
  Params params_;
  MeasurementMatrix noise_covariance_;
};

} // namespace core
} // namespace kinematic_arbiter
