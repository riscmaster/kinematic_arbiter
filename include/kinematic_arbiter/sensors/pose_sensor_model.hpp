#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief 7DOF pose measurement model (position + orientation)
 *
 * Models a sensor that measures position and orientation in 3D space.
 * Measurement vector is [x, y, z, qw, qx, qy, qz]' where
 * [qw, qx, qy, qz] represents orientation as a quaternion.
 */
class PoseSensorModel : public core::MeasurementModelInterface<Eigen::Matrix<double, 7, 1>> {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface<Eigen::Matrix<double, 7, 1>>;
  using StateVector = typename Base::StateVector;
  using MeasurementVector = typename Base::MeasurementVector;
  using MeasurementJacobian = typename Base::MeasurementJacobian;

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
    : Base(sensor_pose_in_body_frame, params) {}

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [x, y, z, qw, qx, qy, qz]'
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
    MeasurementJacobian jacobian = MeasurementJacobian::Zero();

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
};

} // namespace sensors
} // namespace kinematic_arbiter
