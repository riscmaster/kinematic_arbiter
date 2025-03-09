#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief 3DOF position measurement model
 *
 * Models a sensor that measures position in 3D space.
 * Measurement vector is [x, y, z]' representing position in the global frame.
 */
class PositionSensorModel : public core::MeasurementModelInterface<Eigen::Matrix<double, 3, 1>> {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface<Eigen::Matrix<double, 3, 1>>;
  using StateVector = typename Base::StateVector;
  using MeasurementVector = typename Base::MeasurementVector;
  using MeasurementJacobian = typename Base::MeasurementJacobian;

  /**
   * @brief Indices for accessing position measurement components
   */
  struct MeasurementIndex {
    static constexpr int X = 0;
    static constexpr int Y = 1;
    static constexpr int Z = 2;
  };

  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit PositionSensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : Base(sensor_pose_in_body_frame, params) {}

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [x, y, z]'
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override {
    MeasurementVector predicted_measurement = MeasurementVector::Zero();

    // Extract position from state
    Eigen::Vector3d position = state.segment<3>(core::StateIndex::Position::X);

    // Extract orientation quaternion from state
    Eigen::Quaterniond orientation(
        state(core::StateIndex::Quaternion::W),
        state(core::StateIndex::Quaternion::X),
        state(core::StateIndex::Quaternion::Y),
        state(core::StateIndex::Quaternion::Z)
    );

    // Extract the sensor-to-body transform components
    Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();

    // Compute predicted position in global frame
    // p_sensor_global = p_body + R_body_to_global * p_sensor_in_body
    Eigen::Vector3d predicted_position = position + orientation * trans_b_s;

    // Fill measurement vector
    predicted_measurement = predicted_position;

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

    // Position part of the Jacobian - derivative with respect to position is identity
    jacobian.block<3, 3>(0, core::StateIndex::Position::X) = Eigen::Matrix3d::Identity();

    // If the sensor is mounted with an offset, we need to compute the Jacobian
    // with respect to orientation
    if (!sensor_pose_in_body_frame_.translation().isZero()) {
      // Normalize quaternion (critical for numerical stability)
      double qw = state(core::StateIndex::Quaternion::W);
      double qx = state(core::StateIndex::Quaternion::X);
      double qy = state(core::StateIndex::Quaternion::Y);
      double qz = state(core::StateIndex::Quaternion::Z);
      double norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
      qw /= norm; qx /= norm; qy /= norm; qz /= norm;

      // Extract sensor position in body frame
      Eigen::Vector3d p = sensor_pose_in_body_frame_.translation();
      double px = p.x();
      double py = p.y();
      double pz = p.z();

      // Derivative with respect to qw
      jacobian(0, core::StateIndex::Quaternion::W) = 2 * (qw*px - qz*py + qy*pz);
      jacobian(1, core::StateIndex::Quaternion::W) = 2 * (qz*px + qw*py - qx*pz);
      jacobian(2, core::StateIndex::Quaternion::W) = 2 * (-qy*px + qx*py + qw*pz);

      // Derivative with respect to qx
      jacobian(0, core::StateIndex::Quaternion::X) = 2 * (qx*px + qy*py + qz*pz);
      jacobian(1, core::StateIndex::Quaternion::X) = 2 * (qy*qx - qw*pz);
      jacobian(2, core::StateIndex::Quaternion::X) = 2 * (qz*qx + qw*py);

      // Derivative with respect to qy
      jacobian(0, core::StateIndex::Quaternion::Y) = 2 * (qx*qy + qw*pz);
      jacobian(1, core::StateIndex::Quaternion::Y) = 2 * (qy*py + qx*px + qz*pz);
      jacobian(2, core::StateIndex::Quaternion::Y) = 2 * (qz*qy - qw*px);

      // Derivative with respect to qz
      jacobian(0, core::StateIndex::Quaternion::Z) = 2 * (qx*qz - qw*py);
      jacobian(1, core::StateIndex::Quaternion::Z) = 2 * (qy*qz + qw*px);
      jacobian(2, core::StateIndex::Quaternion::Z) = 2 * (qz*pz + qx*px + qy*py);
    }

    return jacobian;
  }
};

} // namespace sensors
} // namespace kinematic_arbiter
