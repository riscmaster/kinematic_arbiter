#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief Body velocity measurement model (linear + angular velocity)
 *
 * Models a sensor that measures linear and angular velocities in the body frame.
 * Measurement vector is [vx, vy, vz, wx, wy, wz]' where
 * [vx, vy, vz] represents linear velocity and [wx, wy, wz] represents angular velocity.
 */
class BodyVelocitySensorModel : public core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>> {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface<Eigen::Matrix<double, 6, 1>>;
  using StateVector = typename Base::StateVector;
  using MeasurementVector = typename Base::MeasurementVector;
  using MeasurementJacobian = typename Base::MeasurementJacobian;

  /**
   * @brief Indices for accessing body velocity measurement components
   */
  struct MeasurementIndex {
    // Linear velocity indices
    static constexpr int VX = 0;
    static constexpr int VY = 1;
    static constexpr int VZ = 2;

    // Angular velocity indices
    static constexpr int WX = 3;
    static constexpr int WY = 4;
    static constexpr int WZ = 5;
  };

  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit BodyVelocitySensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : Base(sensor_pose_in_body_frame, params) {}

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [vx, vy, vz, wx, wy, wz]'
   */
  MeasurementVector PredictMeasurement(const StateVector& state) const override {
    MeasurementVector predicted_measurement = MeasurementVector::Zero();

    // Extract linear velocity from state
    Eigen::Vector3d body_lin_vel = state.segment<3>(core::StateIndex::LinearVelocity::X);

    // Extract angular velocity from state
    Eigen::Vector3d body_ang_vel = state.segment<3>(core::StateIndex::AngularVelocity::X);

    // Extract the sensor-to-body transform components
    Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();
    Eigen::Matrix3d rot_b_s = sensor_pose_in_body_frame_.rotation();

    // Calculate the skew-symmetric matrix for the cross product: ω × r
    auto skew_matrix = [](const Eigen::Vector3d& v) -> Eigen::Matrix3d {
      Eigen::Matrix3d skew;
      skew << 0, -v(2), v(1),
              v(2), 0, -v(0),
              -v(1), v(0), 0;
      return skew;
    };

    // Calculate linear velocity at the sensor location in body frame
    // v_sensor = v_body + ω × r, where r is the position of the sensor in the body frame
    Eigen::Vector3d body_sensor_lin_vel = body_lin_vel + skew_matrix(body_ang_vel) * trans_b_s;

    // Transform velocities to sensor frame
    Eigen::Vector3d sensor_lin_vel = rot_b_s.transpose() * body_sensor_lin_vel;
    Eigen::Vector3d sensor_ang_vel = rot_b_s.transpose() * body_ang_vel;

    // Fill measurement vector
    predicted_measurement.segment<3>(MeasurementIndex::VX) = sensor_lin_vel;
    predicted_measurement.segment<3>(MeasurementIndex::WX) = sensor_ang_vel;

    return predicted_measurement;
  }

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate
   * @return Jacobian of measurement with respect to state
   */
  MeasurementJacobian GetMeasurementJacobian(const StateVector&) const override {
    MeasurementJacobian jacobian = MeasurementJacobian::Zero();

    // Extract transform parameters
    Eigen::Vector3d sensor_offset = sensor_pose_in_body_frame_.translation();
    Eigen::Matrix3d body_to_sensor_rotation = sensor_pose_in_body_frame_.rotation().transpose();

    // ===== LINEAR VELOCITY JACOBIAN =====

    // Linear velocity w.r.t. linear velocity - simple rotation from body to sensor frame
    jacobian.block<3, 3>(MeasurementIndex::VX, core::StateIndex::LinearVelocity::X) = body_to_sensor_rotation;

    // Linear velocity w.r.t. angular velocity - lever arm effect
    // The lever arm effect can be represented by partial derivatives of the cross product
    // These derivatives have a standard form for cross products

    // Define cross product derivatives in a more compact way
    // Following standard notation where [r×] is the skew-symmetric matrix of vector r
    // For a sensor at position r, the derivatives of ω×r with respect to each ω component
    // correspond to the columns of a skew matrix of r

    // Create skew-symmetric matrix of the sensor offset
    Eigen::Matrix3d skew_offset = Eigen::Matrix3d::Zero();
    skew_offset <<
        0,              -sensor_offset.z(),  sensor_offset.y(),
        sensor_offset.z(),  0,              -sensor_offset.x(),
       -sensor_offset.y(),  sensor_offset.x(),  0;

    // Each column of this matrix represents the lever arm effect for one angular velocity component
    // Extract columns for better readability
    Eigen::Vector3d lever_arm_effect_wx = -skew_offset.col(0);  // Effect of ω_x
    Eigen::Vector3d lever_arm_effect_wy = -skew_offset.col(1);  // Effect of ω_y
    Eigen::Vector3d lever_arm_effect_wz = -skew_offset.col(2);  // Effect of ω_z

    // Apply body-to-sensor rotation and fill Jacobian
    jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::X) =
        body_to_sensor_rotation * lever_arm_effect_wx;
    jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::Y) =
        body_to_sensor_rotation * lever_arm_effect_wy;
    jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::Z) =
        body_to_sensor_rotation * lever_arm_effect_wz;

    // ===== ANGULAR VELOCITY JACOBIAN =====

    // Angular velocity w.r.t. angular velocity - simple rotation from body to sensor frame
    jacobian.block<3, 3>(MeasurementIndex::WX, core::StateIndex::AngularVelocity::X) = body_to_sensor_rotation;

    return jacobian;
  }
};

} // namespace sensors
} // namespace kinematic_arbiter
