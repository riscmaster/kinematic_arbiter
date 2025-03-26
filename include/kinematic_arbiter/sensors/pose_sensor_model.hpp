#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace {
  constexpr int kMeasurementDimension = core::MeasurementDimension<core::SensorType::Pose>::value;
}

/**
 * @brief 7DOF pose measurement model (position + orientation)
 *
 * Models a sensor that measures position and orientation in 3D space.
 * Measurement vector is [x, y, z, qw, qx, qy, qz]' where
 * [qw, qx, qy, qz] represents orientation as a quaternion.
 */
class PoseSensorModel : public core::MeasurementModelInterface {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface;
  using StateVector = typename Base::StateVector;
  using Vector = Eigen::Matrix<double, kMeasurementDimension, 1>;
  using Jacobian = Eigen::Matrix<double, kMeasurementDimension, core::StateIndex::kFullStateSize>;
  using Covariance = Eigen::Matrix<double, kMeasurementDimension, kMeasurementDimension>;
  using StateFlags = typename Base::StateFlags;

  /**
   * @brief Indices for accessing pose measurement components
   */
  struct MeasurementIndex {
    // Position indices
    static constexpr int X = 0;
    static constexpr int Y = 1;
    static constexpr int Z = 2;

    // Quaternion indices
    static constexpr int QW = 3;
    static constexpr int QX = 4;
    static constexpr int QY = 5;
    static constexpr int QZ = 6;
  };

  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit PoseSensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : Base(core::SensorType::Pose, sensor_pose_in_body_frame, params, Covariance::Identity()) {}

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [x, y, z, qw, qx, qy, qz]'
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override {
    Vector predicted_measurement = Vector::Zero();

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
    Eigen::Quaterniond rot_b_s(sensor_pose_in_body_frame_.rotation());

    // Compute predicted position in global frame
    // p_sensor_global = p_body + R_body_to_global * p_sensor_in_body
    Eigen::Vector3d predicted_position = position + orientation * trans_b_s;

    // Compute predicted orientation in global frame
    // q_sensor_global = q_body * q_sensor_in_body
    Eigen::Quaterniond predicted_orientation = orientation * rot_b_s;
    predicted_orientation.normalize(); // Ensure unit quaternion

    // Fill measurement vector
    predicted_measurement.segment<3>(MeasurementIndex::X) = predicted_position;
    predicted_measurement(MeasurementIndex::QW) = predicted_orientation.w();
    predicted_measurement(MeasurementIndex::QX) = predicted_orientation.x();
    predicted_measurement(MeasurementIndex::QY) = predicted_orientation.y();
    predicted_measurement(MeasurementIndex::QZ) = predicted_orientation.z();

    return predicted_measurement;
  }

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate (unused in this implementation as Jacobian is constant)
   * @return Jacobian of measurement with respect to state
   */
  MeasurementJacobian GetMeasurementJacobian(const StateVector& /* state */) const override {
    Jacobian jacobian = Jacobian::Zero();

    // Position part of the Jacobian - derivative with respect to position is identity
    jacobian.block<3, 3>(MeasurementIndex::X, core::StateIndex::Position::X) =
        Eigen::Matrix3d::Identity();

    // Remove coupling between linear and angular position in the jacobian by
    // setting to zero. Derivation of partial of position with respect to the
    // rotation in quaternion form can be found here:
    // http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    jacobian.block<3, 4>(MeasurementIndex::X, core::StateIndex::Quaternion::W) =
        Eigen::Matrix<double, 3, 4>::Zero();

    // Extract sensor-in-body quaternion components (q2)
    const Eigen::Quaterniond q2(sensor_pose_in_body_frame_.rotation());
    const double a2 = q2.w(), b2 = q2.x(), c2 = q2.y(), d2 = q2.z();

    // Quaternion Jacobian matrix - exactly matching the first provided formulation
    Eigen::Matrix4d quaternion_product_jacobian;
    quaternion_product_jacobian <<
        // clang-format off
         a2,  b2,  c2,  d2,
        -b2,  a2,  d2, -c2,
        -c2, -d2,  a2,  b2,
        -d2,  c2, -b2,  a2;
        // clang-format on

    // Fill the quaternion part of the measurement Jacobian
    jacobian.block<4, 4>(MeasurementIndex::QW, core::StateIndex::Quaternion::W) =
        quaternion_product_jacobian;

    return jacobian;
  }

  /**
   * @brief Get states that this sensor can directly initialize
   *
   * Pose sensor provides both position and orientation measurements,
   * so it can directly initialize these state components.
   *
   * @return Flags for initializable states
   */
  StateFlags GetInitializableStates() const override {
    StateFlags flags = StateFlags::Zero();

    // Pose sensor can initialize position
    flags[core::StateIndex::Position::X] = true;
    flags[core::StateIndex::Position::Y] = true;
    flags[core::StateIndex::Position::Z] = true;

    // Pose sensor can initialize quaternion
    flags[core::StateIndex::Quaternion::W] = true;
    flags[core::StateIndex::Quaternion::X] = true;
    flags[core::StateIndex::Quaternion::Y] = true;
    flags[core::StateIndex::Quaternion::Z] = true;

    return flags;
  }

  /**
   * @brief Initialize state from pose measurement
   *
   * Initializes position and orientation states from sensor measurement,
   * accounting for sensor-to-body frame transformation.
   *
   * @param measurement Pose measurement [x, y, z, qw, qx, qy, qz]
   * @param valid_states Flags indicating which states are valid
   * @param state State vector to update
   * @param covariance State covariance to update
   * @return Flags indicating which states were initialized
   */
  StateFlags InitializeState(
      const MeasurementVector& measurement,
      const StateFlags&,
      StateVector& state,
      StateCovariance& covariance) const override {

    StateFlags initialized_states = StateFlags::Zero();

    // Extract position and orientation from measurement
    Eigen::Vector3d sensor_position = measurement.segment<3>(MeasurementIndex::X);
    Eigen::Quaterniond sensor_orientation(
        measurement(MeasurementIndex::QW),
        measurement(MeasurementIndex::QX),
        measurement(MeasurementIndex::QY),
        measurement(MeasurementIndex::QZ)
    );

    // Extract the sensor-to-body transform components
    Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();
    Eigen::Quaterniond rot_b_s(sensor_pose_in_body_frame_.rotation());

    // Transform from sensor to body frame
    // For orientation: q_body = q_sensor * (q_sensor_in_body)^-1
    Eigen::Quaterniond body_orientation = sensor_orientation * rot_b_s.inverse();
    body_orientation.normalize();  // Ensure unit quaternion

    // For position: p_body = p_sensor - R_body_to_global * p_sensor_in_body
    Eigen::Vector3d body_position = sensor_position - body_orientation * trans_b_s;

    // Update state with initialized values
    state.segment<3>(core::StateIndex::Position::Begin()) = body_position;
    state(core::StateIndex::Quaternion::W) = body_orientation.w();
    state(core::StateIndex::Quaternion::X) = body_orientation.x();
    state(core::StateIndex::Quaternion::Y) = body_orientation.y();
    state(core::StateIndex::Quaternion::Z) = body_orientation.z();

    // Transform measurement covariance to state covariance
    // Note: This is a simplified approach; a more rigorous implementation would
    // apply the proper uncertainty transformation through the Jacobian

    // For position covariance (direct mapping with lever arm effects)
    Eigen::Matrix3d pos_cov = measurement_covariance_.block<3, 3>(0, 0);

    // For quaternion covariance (simplified transfer from measurement)
    Eigen::Matrix4d quat_cov = measurement_covariance_.block<4, 4>(3, 3);

    // Update covariance blocks
    covariance.block<3, 3>(
        core::StateIndex::Position::Begin(),
        core::StateIndex::Position::Begin()) = pos_cov;

    covariance.block<4, 4>(
        core::StateIndex::Quaternion::Begin(),
        core::StateIndex::Quaternion::Begin()) = quat_cov;

    // Mark states as initialized
    initialized_states[core::StateIndex::Position::X] = true;
    initialized_states[core::StateIndex::Position::Y] = true;
    initialized_states[core::StateIndex::Position::Z] = true;
    initialized_states[core::StateIndex::Quaternion::W] = true;
    initialized_states[core::StateIndex::Quaternion::X] = true;
    initialized_states[core::StateIndex::Quaternion::Y] = true;
    initialized_states[core::StateIndex::Quaternion::Z] = true;

    return initialized_states;
  }
};

} // namespace sensors
} // namespace kinematic_arbiter
