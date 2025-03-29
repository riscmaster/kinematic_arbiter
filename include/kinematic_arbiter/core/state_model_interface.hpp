#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Interface for state transition models used in Kalman filtering
 */
class StateModelInterface {
public:
  // Type definitions
  static constexpr int StateSize = StateIndex::kFullStateSize;
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using Vector3d = Eigen::Vector3d;
  /**
   * @brief Parameters for state model
   */
  struct Params {
    // Sample window size for process noise estimation (n)
    size_t process_noise_window;

    // Default constructor with initialization
    Params() : process_noise_window(40) {}
  };

  /**
   * @brief Reset the state model to initial state
   */
  void reset() {
    process_noise_ = StateMatrix::Identity();
  }

  /**
   * @brief Constructor with parameters
   *
   * @param params Parameters for this state model
   */
  explicit StateModelInterface(const Params& params = Params())
    : params_(params),
      process_noise_(StateMatrix::Identity()) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~StateModelInterface() = default;

  /**
   * @brief Predict state forward in time: x̂_k^- = f(x̂_{k-1}^+, u_k)
   *
   * @param state Current state estimate x̂_{k-1}^+
   * @param dt Time step in seconds
   * @return Predicted next state x̂_k^-
   */
  virtual StateVector PredictState(const StateVector& state, double dt) const = 0;


  /**
   * @brief Get the state transition matrix: A_k
   *
   * The matrix A_k linearizes the state transition function:
   * A_k = ∂f/∂x evaluated at x̂_{k-1}^+ and u_k
   *
   * @param state Current state estimate x̂_{k-1}^+
   * @param dt Time step in seconds
   * @return State transition matrix A_k
   */
  virtual StateMatrix GetTransitionMatrix(const StateVector& state, double dt) const = 0;

/**
   * @brief Predict state forward in time: x̂_k^- = f(x̂_{k-1}^+, u_k)
   *
   * @param state Current state estimate x̂_{k-1}^+
   * @param dt Time step in seconds
   * @return Predicted next state x̂_k^-
   */
  virtual StateVector PredictStateWithInputAccelerations(const StateVector& state, double dt, const Vector3d& linear_acceleration=Vector3d::Zero(), const Vector3d& angular_acceleration=Vector3d::Zero()) const
  {
    StateVector new_state = state;
    new_state.segment<3>(StateIndex::LinearAcceleration::Begin()) = linear_acceleration;
    new_state.segment<3>(StateIndex::AngularAcceleration::Begin()) = angular_acceleration;
    return PredictState(new_state, dt);
  };

  /**
   * @brief Get the process noise covariance matrix: Q_k
   *
   * Process noise represents uncertainty in the state transition.
   *
   * @param dt Time step in seconds
   * @return Process noise covariance matrix Q_k
   */
  virtual StateMatrix GetProcessNoiseCovariance(double dt) const {
    return process_noise_ * fabs(dt);
  }

  /**
   * @brief Update process noise based on state correction
   *
   * Implements dynamic process noise estimation using the equation:
   * Q̂_k = Q̂_{k-1} + (ζ·dt·(x̂_k^- - x̂_k^+)(x̂_k^- - x̂_k^+)^T - Q̂_{k-1})/n
   *
   * Where:
   * - Q̂_k is the new process noise estimate
   * - ζ is the process to measurement noise ratio
   * - dt is the time step
   * - x̂_k^- is the a priori state estimate (before update)
   * - x̂_k^+ is the a posteriori state estimate (after update)
   * - n is the sample window size
   *
   * @param a_priori_state The a priori state estimate x̂_k^-
   * @param a_posteriori_state The a posteriori state estimate x̂_k^+
   * @param process_to_measurement_ratio The process to measurement noise ratio (ζ)
   * @param dt Time step in seconds
   */
  void UpdateProcessNoise(
      const StateVector& a_priori_state,
      const StateVector& a_posteriori_state,
      double process_to_measurement_ratio,
      double dt) {

    // Compute state correction
    StateVector state_diff = a_priori_state - a_posteriori_state;
       // Apply maximum bound to state differences
    StateVector bounded_diff =
        state_diff.array().max(-kMaxStateDiff).min(kMaxStateDiff);

    // Zero out terms that are below the minimum threshold
    StateVector final_diff = bounded_diff.array() * (bounded_diff.array().abs() >= kMinStateDiff).cast<double>();

    StateMatrix update_contribution = final_diff * final_diff.transpose();


    // Apply update only to masked elements
    for (int i = 0; i < StateSize; ++i) {
      for (int j = 0; j < StateSize; ++j) {
        if (update_contribution(i, j) != 0.0) {  // Either element had significant change
          process_noise_(i, j) += process_to_measurement_ratio * dt *
                              (update_contribution(i, j) - process_noise_(i, j)) /
                              params_.process_noise_window;
        }
      }
    }
  }

  /**
   * @brief Get the current process noise matrix
   *
   * @return Current process noise matrix Q̂
   */
  const StateMatrix& GetProcessNoise() const {
    return process_noise_;
  }

  /**
   * @brief Get the state model parameters
   *
   * @return Const reference to parameters
   */
  const Params& GetParams() const { return params_; }

protected:
  // Threshold for numerical comparisons
  static constexpr double kMinStateDiff = 1e-6;
  static constexpr double kMaxStateDiff = 1e6;

  // State model parameters
  Params params_;

  // Process noise covariance
  StateMatrix process_noise_ = StateMatrix::Identity();
};

} // namespace core
} // namespace kinematic_arbiter
