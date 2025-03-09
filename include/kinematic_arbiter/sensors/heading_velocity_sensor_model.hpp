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
