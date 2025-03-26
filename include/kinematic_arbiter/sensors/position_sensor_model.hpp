#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace {
  constexpr int kMeasurementDimension = core::MeasurementDimension<core::SensorType::Position>::value;
}

/**
 * @brief 3DOF position measurement model
 *
 * Models a sensor that measures position in 3D space.
 * Measurement vector is [x, y, z]' representing position in the global frame.
 */
class PositionSensorModel : public core::MeasurementModelInterface {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface;
  using StateVector = typename Base::StateVector;
  using Vector = Eigen::Matrix<double, kMeasurementDimension, 1>;
  using Jacobian = Eigen::Matrix<double, kMeasurementDimension, core::StateIndex::kFullStateSize>;
  using Covariance = Eigen::Matrix<double, kMeasurementDimension, kMeasurementDimension>;
  using StateFlags = typename Base::StateFlags;

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
    : Base(core::SensorType::Position, sensor_pose_in_body_frame, params, Covariance::Identity()) {}

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [x, y, z]'
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override {
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

    return position + orientation * trans_b_s;
  }

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate
   * @return Jacobian of measurement with respect to state
   */
  MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const override {
    Jacobian jacobian = Jacobian::Zero();

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

  /**
   * @brief Get states that this sensor can directly initialize
   *
   * Position sensor provides position measurements in global frame,
   * so it can initialize position state components.
   *
   * @return Flags for initializable states
   */
  StateFlags GetInitializableStates() const override {
    StateFlags flags = StateFlags::Zero();

    // Position sensor can initialize position
    flags[core::StateIndex::Position::X] = true;
    flags[core::StateIndex::Position::Y] = true;
    flags[core::StateIndex::Position::Z] = true;

    return flags;
  }

  /**
   * @brief Initialize state from position measurement
   *
   * Initializes position states from sensor measurement,
   * accounting for sensor-to-body frame offset if quaternion is valid.
   *
   * @param measurement Position measurement [x, y, z]
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

    // Extract position from measurement
    Eigen::Vector3d sensor_position = measurement;

    // Extract the sensor-to-body translation
    Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();

    // Check if quaternion is valid for lever arm compensation
    bool quaternion_valid =
        valid_states[core::StateIndex::Quaternion::W] &&
        valid_states[core::StateIndex::Quaternion::X] &&
        valid_states[core::StateIndex::Quaternion::Y] &&
        valid_states[core::StateIndex::Quaternion::Z];

    Eigen::Vector3d body_position;

    if (quaternion_valid && trans_b_s.norm() > 1e-6) {
      // With valid quaternion, we can properly account for the lever arm
      Eigen::Quaterniond orientation(
          state(core::StateIndex::Quaternion::W),
          state(core::StateIndex::Quaternion::X),
          state(core::StateIndex::Quaternion::Y),
          state(core::StateIndex::Quaternion::Z)
      );

      // Compensate for lever arm: p_body = p_sensor - R_body_to_global * p_sensor_in_body
      body_position = sensor_position - orientation * trans_b_s;
    } else {
      // Without valid quaternion, we ignore the lever arm
      // This will introduce error if the lever arm is significant
      body_position = sensor_position;
    }

    // Update state with initialized position
    state.segment<3>(core::StateIndex::Position::Begin()) = body_position;

    // Update covariance in state
    covariance.block<3, 3>(
        core::StateIndex::Position::Begin(),
        core::StateIndex::Position::Begin()) = measurement_covariance_;

    // If we ignored the lever arm, increase position uncertainty
    if (!quaternion_valid && trans_b_s.norm() > 1e-6) {
      // Add additional uncertainty based on lever arm length
      double lever_arm_length = trans_b_s.norm();
      double additional_variance = lever_arm_length * lever_arm_length;

      // Use Eigen diagonal matrix addition instead of loops
      covariance.block<3, 3>(
          core::StateIndex::Position::Begin(),
          core::StateIndex::Position::Begin()) +=
          additional_variance * Eigen::Matrix3d::Identity();
    }

    // Mark position states as initialized
    initialized_states[core::StateIndex::Position::X] = true;
    initialized_states[core::StateIndex::Position::Y] = true;
    initialized_states[core::StateIndex::Position::Z] = true;

    return initialized_states;
  }
};

} // namespace sensors
} // namespace kinematic_arbiter
