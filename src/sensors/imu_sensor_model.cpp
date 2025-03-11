#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/statistical_utils.hpp"

namespace kinematic_arbiter {
namespace sensors {

using core::StateIndex;

ImuSensorModel::MeasurementVector ImuSensorModel::PredictMeasurement(
    const StateVector& state) const {
  MeasurementVector predicted_measurement;

  // Extract state components
  const Eigen::Quaterniond q(
      state(StateIndex::Quaternion::W),
      state(StateIndex::Quaternion::X),
      state(StateIndex::Quaternion::Y),
      state(StateIndex::Quaternion::Z)
  );
  const Eigen::Vector3d omega = state.segment<3>(StateIndex::AngularVelocity::X);
  const Eigen::Vector3d a_linear = state.segment<3>(StateIndex::LinearAcceleration::X);
  const Eigen::Vector3d alpha = state.segment<3>(StateIndex::AngularAcceleration::X);

  // Extract sensor configuration
  const Eigen::Vector3d& r = sensor_pose_in_body_frame_.translation();
  const Eigen::Matrix3d& R_BS = sensor_pose_in_body_frame_.rotation().transpose();

  // Gravity vector in world frame
  const Eigen::Vector3d g_W(0.0, 0.0, kGravity);

  // Base gyroscope measurement without bias
  Eigen::Vector3d gyro = R_BS * omega;

  // Base accelerometer measurement without bias
  Eigen::Vector3d accel = R_BS * (
      a_linear +
      q.inverse() * g_W +
      alpha.cross(r) +
      omega.cross(omega.cross(r))
  );

  // Apply biases only if calibration is enabled
  if (config_.calibration_enabled) {
    gyro += bias_estimator_.GetGyroBias();
    accel += bias_estimator_.GetAccelBias();
  }

  // Set the measurement components
  predicted_measurement.segment<3>(MeasurementIndex::GX) = gyro;
  predicted_measurement.segment<3>(MeasurementIndex::AX) = accel;

  return predicted_measurement;
}

bool ImuSensorModel::UpdateBiasEstimates(
    const StateVector& state,
    const Eigen::MatrixXd& state_covariance,
    const MeasurementVector& measurement) {

  if (!config_.calibration_enabled || !IsStationary(state, state_covariance, measurement)) {
    return false;
  }

  // Get predicted measurement
  MeasurementVector predicted_measurement = PredictMeasurement(state);


  // Update bias estimates
  bias_estimator_.EstimateBiases(
      measurement.segment<3>(MeasurementIndex::GX),
      measurement.segment<3>(MeasurementIndex::AX),
      predicted_measurement.segment<3>(MeasurementIndex::GX),
      predicted_measurement.segment<3>(MeasurementIndex::AX));

  return true;
}

ImuSensorModel::MeasurementJacobian ImuSensorModel::GetMeasurementJacobian(
    const StateVector& state) const {
  MeasurementJacobian jacobian = MeasurementJacobian::Zero();

  // Extract state components and sensor configuration
  const Eigen::Vector3d& r = sensor_pose_in_body_frame_.translation();
  const Eigen::Matrix3d& R_BS = sensor_pose_in_body_frame_.rotation().transpose();

  const Eigen::Vector3d omega = state.segment<3>(StateIndex::AngularVelocity::X);

  // Extract and normalize quaternion orientation
  Eigen::Quaterniond q(
      state(StateIndex::Quaternion::W),
      state(StateIndex::Quaternion::X),
      state(StateIndex::Quaternion::Y),
      state(StateIndex::Quaternion::Z)
  );
  q.normalize();

  // Helper function to create a skew-symmetric matrix for cross products
  auto skew = [](const Eigen::Vector3d& v) -> Eigen::Matrix3d {
    return (Eigen::Matrix3d() <<
            0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0).finished();
  };

  // ==== LOWER BLOCK: GYROSCOPE JACOBIAN ====
  // Gyro measurement is simply body angular velocity rotated to sensor frame
  jacobian.block<3, 3>(MeasurementIndex::GX, StateIndex::AngularVelocity::X) = R_BS;

  // ==== UPPER BLOCK: ACCELEROMETER JACOBIAN ====

  // 1. Linear acceleration direct mapping
  jacobian.block<3, 3>(MeasurementIndex::AX, StateIndex::LinearAcceleration::X) = R_BS;

  // 2. Angular acceleration effect (tangential acceleration): -R_{BS}[r]_\times
  jacobian.block<3, 3>(MeasurementIndex::AX, StateIndex::AngularAcceleration::X) =
      -R_BS * skew(r);

  // 3. Angular velocity effect on acceleration
  // To fix the issues with the angular velocity Jacobian, we'll use the expanded cross product formula
  Eigen::Matrix3d omega_cross_r_jacobian = Eigen::Matrix3d::Zero();

  // Manually construct the Jacobian for d(ω×(ω×r))/dω
  // This accounts for both terms in the derivative
  for (int i = 0; i < 3; i++) {
    Eigen::Vector3d e_i = Eigen::Vector3d::Zero();
    e_i(i) = 1.0;

    // First term: e_i × (ω × r)
    Eigen::Vector3d term1 = e_i.cross(omega.cross(r));

    // Second term: ω × (e_i × r)
    Eigen::Vector3d term2 = omega.cross(e_i.cross(r));

    // Combined effect
    omega_cross_r_jacobian.col(i) = term1 + term2;
  }

  jacobian.block<3, 3>(MeasurementIndex::AX, StateIndex::AngularVelocity::X) =
      R_BS * omega_cross_r_jacobian;

  // 4. Quaternion effect on gravity
  // Derive Jacobian for rotation of gravity vector by quaternion
  // For gravity vector g_W = [0, 0, g]^T in world frame
  // The rotated gravity in body frame is: g_B = q.inverse() * g_W * q
  const double qw = q.w(), qx = q.x(), qy = q.y(), qz = q.z();
  const double g = kGravity;

  // Using the derivation from section 4.3 of the documentation:
  // For g_W = [0, 0, g]^T, the rotation q.inverse() * g_W gives:
  // g_B = g * [
  //   2(q_x*q_z - q_w*q_y)
  //   2(q_y*q_z + q_w*q_x)
  //   1 - 2(q_x^2 + q_y^2)
  // ]
  //
  // The Jacobian ∂g_B/∂q is therefore:
  Eigen::Matrix<double, 3, 4> quaternion_gravity_jacobian;
  quaternion_gravity_jacobian <<
      /* ∂/∂q_w */      /* ∂/∂q_x */      /* ∂/∂q_y */      /* ∂/∂q_z */
      -2*g*qy,          2*g*qz,           -2*g*qw,          2*g*qx,          // ∂g_x/∂q
      2*g*qx,           2*g*qw,           2*g*qz,           2*g*qy,          // ∂g_y/∂q
      0,                -4*g*qx,          -4*g*qy,          0;               // ∂g_z/∂q

  // Apply sensor-to-body rotation to transform the Jacobian to sensor frame
  jacobian.block<3, 4>(MeasurementIndex::AX, StateIndex::Quaternion::W) =
      R_BS * quaternion_gravity_jacobian;

  return jacobian;
}

bool ImuSensorModel::IsStationary(
    const StateVector& state,
    const StateCovariance& state_covariance,
    const MeasurementVector& measurement) const {

  // Extract measurements directly using segment
  Eigen::Vector3d measured_accel = measurement.segment<3>(MeasurementIndex::AX);

  // Compute acceleration norm (should be close to gravity when stationary)
  double accel_norm = measured_accel.norm();

  // Pre-compute critical values for different degrees of freedom
  double critical_value6dof = core::CalculateChiSquareCriticalValueNDof(5, config_.stationary_confidence_threshold);
  double critical_value1dof = core::CalculateChiSquareCriticalValueNDof(0, config_.stationary_confidence_threshold);

  // 1. Test state velocities (6-DOF)
  int state_vel_idx = StateIndex::LinearVelocity::X;
  Eigen::Matrix<double, 6, 1> state_vel = state.segment<6>(state_vel_idx);
  Eigen::Matrix<double, 6, 6> state_vel_cov = state_covariance.block<6, 6>(state_vel_idx, state_vel_idx);
  bool is_state_vel0 = state_vel.dot(state_vel_cov.llt().solve(state_vel)) < critical_value6dof;

  // 2. Test state accelerations (6-DOF)
  int state_acc_idx = StateIndex::LinearAcceleration::X;
  Eigen::Matrix<double, 6, 1> state_acc = state.segment<6>(state_acc_idx);
  Eigen::Matrix<double, 6, 6> state_acc_cov = state_covariance.block<6, 6>(state_acc_idx, state_acc_idx);
  bool is_state_acc0 = state_acc.dot(state_acc_cov.llt().solve(state_acc)) < critical_value6dof;

  // 3. Test IMU acceleration norm against gravity (1-DOF)
  double gravity_diff = accel_norm - kGravity;
  double accel_norm_var = this->measurement_covariance_.block<3, 3>(MeasurementIndex::AX, MeasurementIndex::AX).trace();
  double gravity_mahalanobis = (gravity_diff * gravity_diff) / (accel_norm_var + kGravityVariance);
  bool is_imu_accel_gravity = gravity_mahalanobis < critical_value1dof;

  // Ignoring angular velocity from IMU for now due to confounding bias effects

  return is_state_vel0 && is_state_acc0 && is_imu_accel_gravity;
}

/**
 * @brief Get states that this sensor can directly initialize
 *
 * IMU provides measurements of angular velocity and linear acceleration.
 * When stationary, it can also help initialize roll and pitch components
 * of orientation from gravity direction.
 *
 * @return Flags for initializable states
 */
typename ImuSensorModel::StateFlags ImuSensorModel::GetInitializableStates() const {
  StateFlags flags = StateFlags::Zero();

  // IMU can initialize angular velocity
  flags[StateIndex::AngularVelocity::X] = true;
  flags[StateIndex::AngularVelocity::Y] = true;
  flags[StateIndex::AngularVelocity::Z] = true;

  // IMU can initialize linear acceleration
  flags[StateIndex::LinearAcceleration::X] = true;
  flags[StateIndex::LinearAcceleration::Y] = true;
  flags[StateIndex::LinearAcceleration::Z] = true;

  // IMU can initialize roll and pitch (but not yaw) from gravity
  flags[StateIndex::Quaternion::W] = true;
  flags[StateIndex::Quaternion::X] = true;
  flags[StateIndex::Quaternion::Y] = true;

  // When stationary, can also initialize angular acceleration (to zero)
  flags[StateIndex::AngularAcceleration::X] = true;
  flags[StateIndex::AngularAcceleration::Y] = true;
  flags[StateIndex::AngularAcceleration::Z] = true;

  return flags;
}

/**
 * @brief Initialize state from IMU measurement
 *
 * Initializes angular velocity and linear acceleration directly.
 * When stationary, initializes orientation (roll/pitch) from gravity
 * and sets angular acceleration to zero.
 *
 * @param measurement IMU measurement [gx, gy, gz, ax, ay, az]
 * @param valid_states Flags indicating which states are valid
 * @param state State vector to update
 * @param covariance State covariance to update
 * @return Flags indicating which states were initialized
 */
typename ImuSensorModel::StateFlags ImuSensorModel::InitializeState(
    const MeasurementVector& measurement,
    const StateFlags& valid_states,
    StateVector& state,
    StateCovariance& covariance) const {

  StateFlags initialized_states = StateFlags::Zero();

  // Extract IMU measurements
  Eigen::Vector3d gyro = measurement.segment<3>(MeasurementIndex::GX);
  Eigen::Vector3d accel = measurement.segment<3>(MeasurementIndex::AX);

  // Transform to body frame
  const Eigen::Matrix3d R_SB = sensor_pose_in_body_frame_.rotation();

  // Apply bias correction if calibration is enabled
  if (config_.calibration_enabled) {
    gyro -= bias_estimator_.GetGyroBias();
    accel -= bias_estimator_.GetAccelBias();
  }

  // Transform to body frame
  Eigen::Vector3d omega_body = R_SB * gyro;
  Eigen::Vector3d accel_body = R_SB * accel;

  // Initialize angular velocity
  state.segment<3>(StateIndex::AngularVelocity::Begin()) = omega_body;

  // Set angular velocity covariance
  Eigen::Matrix3d gyro_cov = measurement_covariance_.block<3, 3>(0, 0);
  covariance.block<3, 3>(
      StateIndex::AngularVelocity::Begin(),
      StateIndex::AngularVelocity::Begin()) = R_SB * gyro_cov * R_SB.transpose();

  // Mark angular velocity as initialized
  initialized_states[StateIndex::AngularVelocity::X] = true;
  initialized_states[StateIndex::AngularVelocity::Y] = true;
  initialized_states[StateIndex::AngularVelocity::Z] = true;

  // Check if stationary
  bool is_stationary = IsStationary(state, covariance, measurement);

  // If stationary, we can initialize orientation (roll/pitch) from gravity
  if (is_stationary) {
    // Extract sensor-to-body transform components
    const Eigen::Vector3d& r = sensor_pose_in_body_frame_.translation();

    // Correct for centripetal acceleration if angular velocity is significant
    if (omega_body.norm() > 0.1) {  // rad/s threshold
      accel_body -= omega_body.cross(omega_body.cross(r));
    }

    // Estimate gravity in body frame (negative of measured acceleration when stationary)
    Eigen::Vector3d g_body = -accel_body;
    g_body.normalize();

    // Create orientation from gravity direction
    // This only defines roll and pitch, not yaw
    Eigen::Vector3d z_world(0, 0, 1);  // Up direction in world frame
    Eigen::Quaterniond q_gravity = Eigen::Quaterniond::FromTwoVectors(g_body, z_world);

    // If we already have valid yaw, preserve it
    bool yaw_valid = valid_states[StateIndex::Quaternion::Z];
    if (yaw_valid) {
      // Extract current quaternion
      Eigen::Quaterniond current_q(
          state(StateIndex::Quaternion::W),
          state(StateIndex::Quaternion::X),
          state(StateIndex::Quaternion::Y),
          state(StateIndex::Quaternion::Z)
      );

      // Convert to rotation matrix and extract Euler angles
      Eigen::Matrix3d rot_matrix = current_q.toRotationMatrix();

      // Get yaw from current quaternion (rotation around Z)
      double yaw = std::atan2(rot_matrix(1, 0), rot_matrix(0, 0));

      // Create rotation matrix with gravity-derived roll/pitch and preserved yaw
      Eigen::Matrix3d gravity_rot = q_gravity.toRotationMatrix();
      Eigen::Matrix3d yaw_rot = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      // Apply yaw rotation to gravity-derived orientation
      Eigen::Matrix3d combined_rot = gravity_rot * yaw_rot;
      Eigen::Quaterniond combined_q(combined_rot);

      // Update state with combined quaternion
      state(StateIndex::Quaternion::W) = combined_q.w();
      state(StateIndex::Quaternion::X) = combined_q.x();
      state(StateIndex::Quaternion::Y) = combined_q.y();
      state(StateIndex::Quaternion::Z) = combined_q.z();
    } else {
      // No valid yaw, just use gravity-derived orientation
      state(StateIndex::Quaternion::W) = q_gravity.w();
      state(StateIndex::Quaternion::X) = q_gravity.x();
      state(StateIndex::Quaternion::Y) = q_gravity.y();
      state(StateIndex::Quaternion::Z) = q_gravity.z();
    }

    // Set appropriate covariance for quaternion
    // Lower uncertainty for roll/pitch, high for yaw
    Eigen::Matrix3d accel_cov = measurement_covariance_.block<3, 3>(3, 3);

    // Simple mapping of accelerometer noise to orientation uncertainty
    // More rigorous approach would use proper uncertainty propagation
    double roll_pitch_variance = 0.01;  // rad^2
    double yaw_variance = M_PI * M_PI / 3.0;  // Very high uncertainty for yaw

    // Simplified diagonal covariance for quaternion
    for (int i = 0; i < 3; i++) {
      covariance(StateIndex::Quaternion::Begin() + i,
                 StateIndex::Quaternion::Begin() + i) = roll_pitch_variance;
    }
    covariance(StateIndex::Quaternion::Begin() + 3,
               StateIndex::Quaternion::Begin() + 3) = yaw_variance;

    // Mark quaternion components as initialized
    initialized_states[StateIndex::Quaternion::W] = true;
    initialized_states[StateIndex::Quaternion::X] = true;
    initialized_states[StateIndex::Quaternion::Y] = true;

    // Don't mark Z (yaw) as initialized unless we preserved it from a valid state
    if (yaw_valid) {
      initialized_states[StateIndex::Quaternion::Z] = true;
    }

    // When stationary, set angular acceleration to zero
    state.segment<3>(StateIndex::AngularAcceleration::Begin()).setZero();

    // Set small covariance for angular acceleration
    double ang_accel_variance = 0.01;  // (rad/s^2)^2
    for (int i = 0; i < 3; i++) {
      covariance(StateIndex::AngularAcceleration::Begin() + i,
                 StateIndex::AngularAcceleration::Begin() + i) = ang_accel_variance;
    }

    // Mark angular acceleration as initialized
    initialized_states[StateIndex::AngularAcceleration::X] = true;
    initialized_states[StateIndex::AngularAcceleration::Y] = true;
    initialized_states[StateIndex::AngularAcceleration::Z] = true;
  }

  // Initialize linear acceleration
  // First, we need to compensate for gravity if orientation is valid
  bool orientation_valid =
      valid_states[StateIndex::Quaternion::W] &&
      valid_states[StateIndex::Quaternion::X] &&
      valid_states[StateIndex::Quaternion::Y] &&
      valid_states[StateIndex::Quaternion::Z];

  if (orientation_valid) {
    // Extract quaternion
    Eigen::Quaterniond q(
        state(StateIndex::Quaternion::W),
        state(StateIndex::Quaternion::X),
        state(StateIndex::Quaternion::Y),
        state(StateIndex::Quaternion::Z)
    );

    // Gravity vector in world frame
    const Eigen::Vector3d g_W(0.0, 0.0, kGravity);

    // Compute linear acceleration by removing gravity and centripetal effects
    Eigen::Vector3d a_linear = accel_body - q.inverse() * g_W;

    // If we have angular velocity and lever arm, remove centripetal effects
    if (initialized_states[StateIndex::AngularVelocity::X]) {
      const Eigen::Vector3d& r = sensor_pose_in_body_frame_.translation();
      a_linear -= omega_body.cross(omega_body.cross(r));

      // If we also have angular acceleration, remove that effect too
      if (valid_states[StateIndex::AngularAcceleration::X]) {
        Eigen::Vector3d alpha = state.segment<3>(StateIndex::AngularAcceleration::Begin());
        a_linear -= alpha.cross(r);
      }
    }

    // Update state with linear acceleration
    state.segment<3>(StateIndex::LinearAcceleration::Begin()) = a_linear;
  } else {
    // Without valid orientation, we can't properly remove gravity
    // Just use the raw acceleration with a warning flag
    if (is_stationary) {
      // If stationary, we know acceleration should be zero in body frame
      state.segment<3>(StateIndex::LinearAcceleration::Begin()).setZero();
    } else {
      // Otherwise, use measured value but with high uncertainty
      state.segment<3>(StateIndex::LinearAcceleration::Begin()) = accel_body;
    }
  }

  // Set linear acceleration covariance
  Eigen::Matrix3d accel_cov = measurement_covariance_.block<3, 3>(3, 3);
  Eigen::Matrix3d lin_accel_cov = R_SB * accel_cov * R_SB.transpose();

  // If orientation isn't valid, increase uncertainty to account for gravity
  if (!orientation_valid && !is_stationary) {
    lin_accel_cov.diagonal().array() += kGravity * kGravity;
  }

  covariance.block<3, 3>(
      StateIndex::LinearAcceleration::Begin(),
      StateIndex::LinearAcceleration::Begin()) = lin_accel_cov;

  // Mark linear acceleration as initialized
  initialized_states[StateIndex::LinearAcceleration::X] = true;
  initialized_states[StateIndex::LinearAcceleration::Y] = true;
  initialized_states[StateIndex::LinearAcceleration::Z] = true;

  return initialized_states;
}

} // namespace sensors
} // namespace kinematic_arbiter
