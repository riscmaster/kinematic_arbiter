#pragma once

#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>

namespace kinematic_arbiter {
namespace models {


/**
 * @brief State model for 3D rigid body with quaternion orientation
 *
 * Implements a state transition model for a rigid body in 3D space with:
 * - 3D position
 * - Quaternion orientation
 * - 3D linear velocity
 * - 3D angular velocity
 * - 3D linear acceleration
 * - 3D angular acceleration
 *
 * The model uses quaternion kinematics for rotation representation to avoid gimbal lock.
 */
class RigidBodyStateModel : public core::StateModelInterface {
public:
  // Use convenient matrix/vector aliases for 3D/4D operations
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Vector3d = Eigen::Vector3d;
  using Vector4d = Eigen::Vector4d;
  using Matrix4d = Eigen::Matrix<double, 4, 4>;

  // Type alias for cleaner access to state indices
  using SIdx = core::StateIndex;

  /**
   * @brief Construct a new RigidBodyStateModel
   * @param params State model parameters
   */
  explicit RigidBodyStateModel(const Params& params = Params())
    : core::StateModelInterface(params) {}

  /**
   * @brief Predict state forward in time: x̂_k^- = f(x̂_{k-1}^+, u_k)
   *
   * @param current_state Current state estimate x̂_{k-1}^+
   * @param time_step Time step in seconds
   * @return Predicted next state x̂_k^-
   */
  StateVector PredictState(const StateVector& current_state, double time_step) const override {
    StateVector new_states = current_state;

    // Extract quaternion and normalize
    Eigen::Quaterniond orientation(
      current_state[SIdx::Quaternion::W],
      current_state[SIdx::Quaternion::X],
      current_state[SIdx::Quaternion::Y],
      current_state[SIdx::Quaternion::Z]
    );
    orientation.normalize();

    // Get rotation matrix
    Matrix3d rotation_matrix_b_to_w = orientation.toRotationMatrix();

    // Extract velocity and acceleration components
    const Vector3d linear_velocity =
      current_state.segment<3>(SIdx::LinearVelocity::Begin());

    const Vector3d angular_velocity =
      current_state.segment<3>(SIdx::AngularVelocity::Begin());

    const Vector3d linear_acceleration =
      current_state.segment<3>(SIdx::LinearAcceleration::Begin());

    const Vector3d angular_acceleration =
      current_state.segment<3>(SIdx::AngularAcceleration::Begin());

    // Linear XYZ Position Prediction Model
    new_states.segment<3>(SIdx::Position::Begin()) +=
      rotation_matrix_b_to_w * ((linear_velocity + time_step * 0.5 * linear_acceleration) * time_step);

    // Angular Position (Quaternion) Prediction Model
    // Use quaternion kinematics: q(t+dt) = exp(0.5*ω*dt) * q(t)
    Eigen::Quaterniond delta_q;
    double angular_velocity_norm = angular_velocity.norm();

    if (angular_velocity_norm > std::numeric_limits<double>::min()) {
      Vector3d axis = angular_velocity / angular_velocity_norm;
      double angle = angular_velocity_norm * time_step;
      delta_q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    } else {
      delta_q = Eigen::Quaterniond::Identity();
    }

    Eigen::Quaterniond new_quaternion = delta_q * orientation;
    new_quaternion.normalize();

    new_states[SIdx::Quaternion::W] = new_quaternion.w();
    new_states[SIdx::Quaternion::X] = new_quaternion.x();
    new_states[SIdx::Quaternion::Y] = new_quaternion.y();
    new_states[SIdx::Quaternion::Z] = new_quaternion.z();

    // Velocity prediction (integrate acceleration)
    new_states.segment<3>(SIdx::LinearVelocity::Begin()) += time_step * linear_acceleration;
    new_states.segment<3>(SIdx::AngularVelocity::Begin()) += time_step * angular_acceleration;

    // Acceleration is modelled as constant zero (system dynamics)
    // This assumes no external forces are applied between updates
    new_states.segment<3>(SIdx::LinearAcceleration::Begin()) = Vector3d::Zero();
    new_states.segment<3>(SIdx::AngularAcceleration::Begin()) = Vector3d::Zero();

    return new_states;
  }

  /**
   * @brief Get the state transition matrix: A_k
   *
   * The matrix A_k linearizes the state transition function:
   * A_k = ∂f/∂x evaluated at x̂_{k-1}^+ and u_k
   *
   * @param current_state Current state estimate x̂_{k-1}^+
   * @param time_step Time step in seconds
   * @return State transition matrix A_k
   */
  StateMatrix GetTransitionMatrix(const StateVector& current_state, double time_step) const override {
    StateMatrix jacobian = StateMatrix::Zero();

    // Extract quaternion and normalize
    Eigen::Quaterniond orientation(
      current_state[SIdx::Quaternion::W],
      current_state[SIdx::Quaternion::X],
      current_state[SIdx::Quaternion::Y],
      current_state[SIdx::Quaternion::Z]
    );
    orientation.normalize();

    // Get rotation matrix (world to body frame)
    Matrix3d rotation_matrix_w_to_b = orientation.toRotationMatrix().transpose();

    // Position block - identity for position elements
    jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::Position::Begin()) =
      Matrix3d::Identity();

    // Position wrt linear velocity
    jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::LinearVelocity::Begin()) =
      rotation_matrix_w_to_b * (time_step * Matrix3d::Identity());

    // Position wrt linear acceleration
    jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::LinearAcceleration::Begin()) =
      rotation_matrix_w_to_b * (time_step * time_step * 0.5 * Matrix3d::Identity());

    // Remove coupling between linear and angular position in the jacobian by
    // setting to zero. Derivation of partial of position with respect to the
    // rotation in quaternion form can be found here:
    // http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    jacobian.block<3,4>(SIdx::Position::Begin(), SIdx::Quaternion::Begin()) =
        Eigen::Matrix<double, 3, 4>::Zero();

    // Quaternion self-propagation
    jacobian.block<4,4>(SIdx::Quaternion::Begin(), SIdx::Quaternion::Begin()) =
        Matrix4d::Identity();

    // Extract angular velocity components
    const Vector3d angular_velocity =
      current_state.segment<3>(SIdx::AngularVelocity::Begin());

    // Calculate angular velocity magnitude
    double angular_velocity_norm = angular_velocity.norm();

    // Compute the derivative matrices using the derived formula
    // d/dw[(w/||w||) * sin(||w|| * t/2)] ≈ (t*I)/2 - (t^3*||w||^2)/16 * (w*w^T/||w||^2)

    // First term: (t*I)/2
    double t = time_step;
    Matrix3d identity_term = (t/2.0) * Matrix3d::Identity();

    // Second term: (t^3*||w||^2)/16 * (w*w^T/||w||^2)
    Matrix3d outer_product_term;
    if (angular_velocity_norm > std::numeric_limits<double>::min()) {
      // Use full formula when angular velocity is non-zero
      outer_product_term = (t*t*t * angular_velocity_norm*angular_velocity_norm / 16.0) *
                          (angular_velocity * angular_velocity.transpose()) /
                          (angular_velocity_norm * angular_velocity_norm);
    } else {
      outer_product_term = Matrix3d::Zero();
    }

    // Combined derivative matrix for the vector part of quaternion
    Matrix3d vector_derivative = identity_term - outer_product_term;

    // Create quaternion derivatives for angular velocity components
    Vector4d dq_dx = Vector4d(
        -0.25 * time_step*time_step*angular_velocity_norm * angular_velocity[0],
        vector_derivative(0,0), vector_derivative(0,1), vector_derivative(0,2)
    );
    Vector4d dq_dy = Vector4d(
        -0.25 * time_step*time_step*angular_velocity_norm * angular_velocity[1],
        vector_derivative(1,0), vector_derivative(1,1), vector_derivative(1,2)
    );
    Vector4d dq_dz = Vector4d(
        -0.25 * time_step*time_step*angular_velocity_norm * angular_velocity[2],
        vector_derivative(2,0), vector_derivative(2,1), vector_derivative(2,2)
    );

    // Set the Jacobian block for quaternion wrt angular velocity
    jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::X) = dq_dx;
    jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::Y) = dq_dy;
    jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::Z) = dq_dz;

    // Velocity blocks - identity for velocity elements
    jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearVelocity::Begin()) =
      Matrix3d::Identity();

    // Linear velocity wrt linear acceleration
    jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearAcceleration::Begin()) =
      time_step * Matrix3d::Identity();

    // Angular velocity blocks - identity for angular velocity elements
    jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularVelocity::Begin()) =
      Matrix3d::Identity();

    // Angular velocity wrt angular acceleration
    jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularAcceleration::Begin()) =
      time_step * Matrix3d::Identity();

    // Acceleration blocks are modelled as zero (no change)
    // Corresponding jacobian elements remain zero

    return jacobian;
  }
};

} // namespace models
} // namespace kinematic_arbiter
