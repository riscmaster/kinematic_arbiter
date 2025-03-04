#pragma once

#include <Eigen/Dense>
#include "kinematic_arbiter/core/state_model_interface.hpp"

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Base class for all measurement types
 */
struct MeasurementBase {
  double timestamp;  // Time in seconds

  explicit MeasurementBase(double ts) : timestamp(ts) {}
  virtual ~MeasurementBase() = default;
};

/**
 * @brief Interface for measurement models
 *
 * Defines the measurement model and its Jacobian.
 *
 * @tparam MeasurementType Type of measurement (must derive from MeasurementBase)
 * @tparam MeasurementSize Dimension of the measurement vector
 */
template<typename MeasurementType, int MeasurementSize>
class MeasurementModelInterface {
public:
  using StateVector = Eigen::Matrix<double, kStateSize, 1>;
  using MeasurementVector = Eigen::Matrix<double, MeasurementSize, 1>;
  using MeasurementMatrix = Eigen::Matrix<double, MeasurementSize, MeasurementSize>;
  using ObservationMatrix = Eigen::Matrix<double, MeasurementSize, kStateSize>;

  /**
   * @brief Predict measurement based on current state
   *
   * Implements the measurement model: y_k = C_k * x_k
   *
   * @param state Current state vector (x_k)
   * @return Predicted measurement (C_k * x_k)
   */
  virtual MeasurementVector PredictMeasurement(
      const StateVector& state) const = 0;

  /**
   * @brief Compute observation matrix (Jacobian of measurement function)
   *
   * Returns the matrix C_k in the measurement model
   *
   * @param state Current state vector (x_k)
   * @return Observation matrix (C_k)
   */
  virtual ObservationMatrix GetObservationMatrix(
      const StateVector& state) const = 0;

  /**
   * @brief Get the initial measurement noise covariance matrix
   *
   * @return Initial measurement noise covariance matrix (R_k)
   */
  virtual MeasurementMatrix GetInitialNoiseCovarianceMatrix() const = 0;

  virtual ~MeasurementModelInterface() = default;
};

} // namespace core
} // namespace kinematic_arbiter
