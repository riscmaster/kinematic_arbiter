#pragma once

#include <Eigen/Dense>

namespace kinematic_arbiter {
namespace core {

// Fixed state size for position, orientation, velocity, angular velocity, acceleration, angular acceleration
constexpr int kStateSize = 18;

/**
 * @brief Interface for state transition models
 *
 * Defines the process model for state transition and its Jacobian.
 * The state vector is fixed at 18 dimensions:
 * - Position (3D)
 * - Orientation (3D)
 * - Linear velocity (3D)
 * - Angular velocity (3D)
 * - Linear acceleration (3D)
 * - Angular acceleration (3D)
 */
class StateModelInterface {
public:
  using StateVector = Eigen::Matrix<double, kStateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;

  /**
   * @brief Predict next state based on current state and time step
   *
   * Implements the process model: x_k = A_{k-1} * x_{k-1} + v_k
   *
   * @param current_state Current state vector (x_{k-1})
   * @param time_step Time step in seconds
   * @return Predicted next state (x_k)
   */
  virtual StateVector PredictState(
      const StateVector& current_state,
      double time_step) const = 0;

  /**
   * @brief Compute state transition matrix (Jacobian of state transition)
   *
   * Returns the matrix A_{k-1} in the process model
   *
   * @param current_state Current state vector (x_{k-1})
   * @param time_step Time step in seconds
   * @return State transition matrix (A_{k-1})
   */
  virtual StateMatrix GetStateTransitionMatrix(
      const StateVector& current_state,
      double time_step) const = 0;

  virtual ~StateModelInterface() = default;
};

} // namespace core
} // namespace kinematic_arbiter
