#pragma once

#include <Eigen/Geometry>
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief Heading velocity sensor model
 *
 * Models a sensor that measures the projection of velocity onto the heading direction.
 * The heading direction is defined as the x-axis of the body frame rotated by the body orientation.
 * Measurement vector is a scalar [v_h] representing the velocity along the heading direction.
 */
class HeadingVelocitySensorModel : public core::MeasurementModelInterface<Eigen::Matrix<double, 1, 1>> {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface<Eigen::Matrix<double, 1, 1>>;
  using StateVector = typename Base::StateVector;
  using MeasurementVector = typename Base::MeasurementVector;
  using MeasurementJacobian = typename Base::MeasurementJacobian;
  using StateFlags = typename Base::StateFlags;
  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit HeadingVelocitySensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : Base(sensor_pose_in_body_frame, params) {}

  /**
   * @brief Predict measurement based on current state
   *
   * @param state Current state estimate
   * @return Predicted measurement
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override {
    MeasurementVector predicted_measurement = MeasurementVector::Zero();

    // Extract velocity from state
    Eigen::Vector3d velocity = state.segment<3>(core::StateIndex::LinearVelocity::Begin());

    // Extract orientation quaternion from state
    Eigen::Quaterniond orientation(
        state(core::StateIndex::Quaternion::W),
        state(core::StateIndex::Quaternion::X),
        state(core::StateIndex::Quaternion::Y),
        state(core::StateIndex::Quaternion::Z)
    );

    // Compute heading vector h(q) = R(q)[1,0,0]^T
    Eigen::Vector3d heading_vector = ComputeHeadingVector(orientation);

    // Compute velocity projection onto heading: z_v = v^T * h(q)
    predicted_measurement(0) = velocity.dot(heading_vector);

    return predicted_measurement;
  }

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate
   * @return Jacobian of measurement with respect to state
   */
  MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const override {
    MeasurementJacobian jacobian = MeasurementJacobian::Zero();

    // Extract velocity from state
    Eigen::Vector3d velocity = state.segment<3>(core::StateIndex::LinearVelocity::Begin());

    // Extract orientation quaternion from state
    double qw = state(core::StateIndex::Quaternion::W);
    double qx = state(core::StateIndex::Quaternion::X);
    double qy = state(core::StateIndex::Quaternion::Y);
    double qz = state(core::StateIndex::Quaternion::Z);

    // Compute heading vector h(q)
    Eigen::Vector3d heading_vector = ComputeHeadingVector(qw, qx, qy, qz);

    // Jacobian with respect to velocity: ∂z_v/∂v = h(q)^T
    jacobian.block<1, 3>(0, core::StateIndex::LinearVelocity::Begin()) = heading_vector.transpose();

    // Jacobian with respect to quaternion: ∂z_v/∂q = v^T * ∂h/∂q
    double vx = velocity.x();
    double vy = velocity.y();
    double vz = velocity.z();

    // Quaternion derivatives as per the mathematical model
    jacobian(0, core::StateIndex::Quaternion::W) = 2.0 * (vy * qz - vz * qy);
    jacobian(0, core::StateIndex::Quaternion::X) = 2.0 * (vy * qy + vz * qz);
    jacobian(0, core::StateIndex::Quaternion::Y) = 2.0 * (-2.0 * vx * qy + vy * qx - vz * qw);
    jacobian(0, core::StateIndex::Quaternion::Z) = 2.0 * (-2.0 * vx * qz + vy * qw + vz * qx);

    return jacobian;
  }

  /**
   * @brief Get states that this sensor can directly initialize
   *
   * Heading velocity sensor can partially initialize:
   * - Linear velocity components along heading direction
   * - Only yaw component of quaternion (affecting W and Z components)
   *
   * @return Flags for initializable states
   */
  StateFlags GetInitializableStates() const override {
    StateFlags flags = StateFlags::Zero();

    // Heading velocity sensor can partially initialize linear velocity
    flags[core::StateIndex::LinearVelocity::X] = true;
    flags[core::StateIndex::LinearVelocity::Y] = true;
    flags[core::StateIndex::LinearVelocity::Z] = true;

    return flags;
  }

  /**
   * @brief Initialize state from heading velocity measurement
   *
   * Assumes heading velocity represents velocity magnitude along the vehicle's heading.
   * Requires a valid quaternion to determine the heading direction.
   *
   * @param measurement Heading velocity measurement
   * @param valid_states Flags indicating which states are valid
   * @param state State vector to update
   * @param covariance State covariance to update
   * @return Flags indicating which states were initialized
   */
  StateFlags InitializeState(
      const MeasurementVector& measurement,
      const StateFlags& valid_states,
      StateVector& state,
      StateCovariance& covariance) const override {

    StateFlags initialized_states = StateFlags::Zero();

    // Extract heading velocity measurement
    double velocity_magnitude = measurement(0);

    // Check if magnitude is sufficient for meaningful initialization
    const double MIN_VELOCITY = 0.5;  // m/s
    if (std::abs(velocity_magnitude) < MIN_VELOCITY) {
      return initialized_states;  // Too small to provide reliable information
    }

    // Check quaternion validity - we need full orientation to determine heading
    bool quaternion_valid =
        valid_states[core::StateIndex::Quaternion::W] &&
        valid_states[core::StateIndex::Quaternion::X] &&
        valid_states[core::StateIndex::Quaternion::Y] &&
        valid_states[core::StateIndex::Quaternion::Z];

    // We can only initialize if we have a valid quaternion
    if (quaternion_valid) {
      // Extract orientation quaternion
      Eigen::Quaterniond q(
          state(core::StateIndex::Quaternion::W),
          state(core::StateIndex::Quaternion::X),
          state(core::StateIndex::Quaternion::Y),
          state(core::StateIndex::Quaternion::Z)
      );

      // Compute heading vector
      Eigen::Vector3d heading_vector = ComputeHeadingVector(q);

      // Initialize velocity along heading
      state.segment<3>(core::StateIndex::LinearVelocity::Begin()) =
          velocity_magnitude * heading_vector;

      // Set appropriate covariance
      double velocity_variance = measurement_covariance_(0, 0);

      // Heading projection matrix (outer product)
      Eigen::Matrix3d heading_proj = heading_vector * heading_vector.transpose();

      // Perpendicular projection matrix
      Eigen::Matrix3d perp_proj = Eigen::Matrix3d::Identity() - heading_proj;

      // Set low uncertainty along heading, high perpendicular
      double perp_variance = 10.0 * velocity_variance;
      Eigen::Matrix3d vel_cov = velocity_variance * heading_proj + perp_variance * perp_proj;

      covariance.block<3, 3>(
          core::StateIndex::LinearVelocity::Begin(),
          core::StateIndex::LinearVelocity::Begin()) = vel_cov;

      // Mark linear velocity as initialized
      initialized_states[core::StateIndex::LinearVelocity::X] = true;
      initialized_states[core::StateIndex::LinearVelocity::Y] = true;
      initialized_states[core::StateIndex::LinearVelocity::Z] = true;
    }
    return initialized_states;
  }

private:
  /**
   * @brief Compute heading vector from quaternion
   *
   * @param orientation Quaternion representing orientation
   * @return Heading vector
   */
  Eigen::Vector3d ComputeHeadingVector(const Eigen::Quaterniond& orientation) const {
    // Heading vector is the x-axis of the body frame rotated by the orientation
    return orientation * Eigen::Vector3d::UnitX();
  }

  /**
   * @brief Compute heading vector from quaternion components
   *
   * @param qw W component of quaternion
   * @param qx X component of quaternion
   * @param qy Y component of quaternion
   * @param qz Z component of quaternion
   * @return Heading vector
   */
  Eigen::Vector3d ComputeHeadingVector(double qw, double qx, double qy, double qz) const {
    // Compute heading vector directly from quaternion components
    // h(q) = [1-2(qy²+qz²), 2(qx*qy+qw*qz), 2(qx*qz-qw*qy)]^T
    Eigen::Vector3d heading;
    heading.x() = 1.0 - 2.0 * (qy * qy + qz * qz);
    heading.y() = 2.0 * (qx * qy + qw * qz);
    heading.z() = 2.0 * (qx * qz - qw * qy);
    return heading;
  }
};

} // namespace sensors
} // namespace kinematic_arbiter
