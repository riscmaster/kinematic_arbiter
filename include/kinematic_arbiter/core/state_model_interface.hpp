#pragma once

#include <Eigen/Dense>

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Interface for state transition models used in Kalman filtering
 *
 * @tparam StateType Type of state vector
 */
template<typename StateType>
class StateModelInterface {
public:
  // Type definitions
  using StateVector = StateType;
  using StateMatrix = Eigen::Matrix<double, StateType::RowsAtCompileTime, StateType::RowsAtCompileTime>;

  /**
   * @brief Parameters for state model
   */
  struct Params {
    // Initial state covariance uncertainty
    double initial_state_uncertainty = 1.0;

    // Base process noise magnitude (Q matrix scaling)
    double process_noise_magnitude = 0.01;
  };

  /**
   * @brief Constructor with parameters
   *
   * @param params Parameters for this state model
   */
  explicit StateModelInterface(const Params& params = Params())
    : params_(params) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~StateModelInterface() = default;

  /**
   * @brief Predict state forward in time
   *
   * @param state Current state estimate
   * @param dt Time step in seconds
   * @return Predicted next state
   */
  virtual StateVector PredictState(const StateVector& state, double dt) const = 0;

  /**
   * @brief Get the state transition matrix
   *
   * @param state Current state estimate
   * @param dt Time step in seconds
   * @return State transition matrix (F)
   */
  virtual StateMatrix GetTransitionMatrix(const StateVector& state, double dt) const = 0;

  /**
   * @brief Get the process noise covariance
   *
   * @param state Current state estimate
   * @param dt Time step in seconds
   * @return Process noise covariance matrix (Q)
   */
  virtual StateMatrix GetProcessNoiseCovariance(const StateVector& state, double dt) const = 0;

  /**
   * @brief Get the state model parameters
   *
   * @return Const reference to parameters
   */
  const Params& GetParams() const { return params_; }

protected:
  Params params_;
};

} // namespace core
} // namespace kinematic_arbiter
