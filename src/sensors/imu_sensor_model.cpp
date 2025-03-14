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

Eigen::Matrix<double, 6, 1> ImuSensorModel::GetPredictionModelInputs(
    const StateVector& state_before_prediction,
    const StateCovariance& state_covariance_before_prediction,
    const MeasurementVector& measurement_after_prediction,
    double dt) const {
  if (fabs(dt) < 1e-10) {
    return state_before_prediction.segment<6>(StateIndex::LinearAcceleration::Begin());
  }

  // Use the proper type with namespace/scope
  Eigen::Matrix<double, 6, 1> inputs;

  // Extract current state components (before prediction)
  const Eigen::Quaterniond current_q(
      state_before_prediction(StateIndex::Quaternion::W),
      state_before_prediction(StateIndex::Quaternion::X),
      state_before_prediction(StateIndex::Quaternion::Y),
      state_before_prediction(StateIndex::Quaternion::Z)
  );
  const Eigen::Vector3d current_ang_vel = state_before_prediction.segment<3>(StateIndex::AngularVelocity::Begin());

  // Get current accelerations from state
  const Eigen::Vector3d current_lin_acc = state_before_prediction.segment<3>(StateIndex::LinearAcceleration::Begin());
  const Eigen::Vector3d current_ang_acc = state_before_prediction.segment<3>(StateIndex::AngularAcceleration::Begin());

  // Extract gyro and accelerometer measurements (after prediction)
  const Eigen::Vector3d measured_gyro = measurement_after_prediction.segment<3>(MeasurementIndex::GX);
  const Eigen::Vector3d measured_accel = measurement_after_prediction.segment<3>(MeasurementIndex::AX);

  // Extract sensor configuration
  const Eigen::Vector3d& r = sensor_pose_in_body_frame_.translation();
  const Eigen::Matrix3d& R_SB = sensor_pose_in_body_frame_.rotation();

  // Gravity vector in world frame
  const Eigen::Vector3d g_W(0.0, 0.0, kGravity);

  // Remove bias if calibration is enabled
  Eigen::Vector3d corrected_gyro = measured_gyro;
  Eigen::Vector3d corrected_accel = measured_accel;
  if (config_.calibration_enabled) {
    corrected_gyro -= bias_estimator_.GetGyroBias();
    corrected_accel -= bias_estimator_.GetAccelBias();
  }

  // Convert gyro measurement to body frame angular velocity after prediction
  Eigen::Vector3d omega_after = R_SB.transpose() * corrected_gyro;

  // Solve for angular acceleration using the relationship between current and after-prediction angular velocity
  // ω_after = ω_before + α*dt
  // Rearranging: α = (ω_after - ω_before) / dt
  Eigen::Vector3d a_angular_from_meas = (omega_after - current_ang_vel) / dt;

  // Extract state covariance for accelerations
  Eigen::Matrix3d lin_acc_cov = state_covariance_before_prediction.block<3, 3>(
      StateIndex::LinearAcceleration::Begin(), StateIndex::LinearAcceleration::Begin());
  Eigen::Matrix3d ang_acc_cov = state_covariance_before_prediction.block<3, 3>(
      StateIndex::AngularAcceleration::Begin(), StateIndex::AngularAcceleration::Begin());

  // Extract measurement covariance for accelerometer and gyroscope
  Eigen::Matrix3d accel_meas_cov = measurement_covariance_.block<3, 3>(MeasurementIndex::AX, MeasurementIndex::AX);
  Eigen::Matrix3d gyro_meas_cov = measurement_covariance_.block<3, 3>(MeasurementIndex::GX, MeasurementIndex::GX);

  // Apply process_to_measurement_noise_ratio to scale the state covariance
  double ratio = validation_params_.process_to_measurement_noise_ratio;
  Eigen::Matrix3d scaled_lin_acc_cov = lin_acc_cov * ratio;
  Eigen::Matrix3d scaled_ang_acc_cov = ang_acc_cov * ratio;

  // Compute measurement-derived acceleration covariance (simplified)
  Eigen::Matrix3d lin_acc_meas_cov = R_SB.transpose() * accel_meas_cov * R_SB;
  Eigen::Matrix3d ang_acc_meas_cov = gyro_meas_cov / (dt * dt);  // Simplified transformation

  // Ensure covariance matrices are positive definite
  const double min_eigenvalue = 1e-6;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_lin(scaled_lin_acc_cov);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_ang(scaled_ang_acc_cov);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_lin_meas(lin_acc_meas_cov);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_ang_meas(ang_acc_meas_cov);

  for (int i = 0; i < 3; ++i) {
    if (es_lin.eigenvalues()(i) < min_eigenvalue)
      scaled_lin_acc_cov += Eigen::Matrix3d::Identity() * (min_eigenvalue - es_lin.eigenvalues()(i));
    if (es_ang.eigenvalues()(i) < min_eigenvalue)
      scaled_ang_acc_cov += Eigen::Matrix3d::Identity() * (min_eigenvalue - es_ang.eigenvalues()(i));
    if (es_lin_meas.eigenvalues()(i) < min_eigenvalue)
      lin_acc_meas_cov += Eigen::Matrix3d::Identity() * (min_eigenvalue - es_lin_meas.eigenvalues()(i));
    if (es_ang_meas.eigenvalues()(i) < min_eigenvalue)
      ang_acc_meas_cov += Eigen::Matrix3d::Identity() * (min_eigenvalue - es_ang_meas.eigenvalues()(i));
  }

  // Optimal fusion of angular acceleration using LDLT decomposition
  Eigen::LDLT<Eigen::Matrix3d> solver_ang_state(scaled_ang_acc_cov);
  Eigen::LDLT<Eigen::Matrix3d> solver_ang_meas(ang_acc_meas_cov);

  // Calculate (C1⁻¹ + C2⁻¹) implicitly
  Eigen::Matrix3d ang_Cinv_sum = solver_ang_state.solve(Eigen::Matrix3d::Identity()) +
                                solver_ang_meas.solve(Eigen::Matrix3d::Identity());

  // Compute weighted sum: C1⁻¹x1 + C2⁻¹x2
  Eigen::Vector3d ang_weighted_sum = solver_ang_state.solve(current_ang_acc) +
                                   solver_ang_meas.solve(a_angular_from_meas);

  // Solve (C1⁻¹ + C2⁻¹)x̄ = weighted_sum
  Eigen::LDLT<Eigen::Matrix3d> ang_final_solver(ang_Cinv_sum);
  Eigen::Vector3d a_angular = ang_final_solver.solve(ang_weighted_sum);

  // For more accurate quaternion prediction, use the optimally fused angular acceleration
  Eigen::Vector3d axis_angle = current_ang_vel * dt + 0.5 * a_angular * dt * dt;
  double angle = axis_angle.norm();

  Eigen::Quaterniond delta_q;
  if (angle > std::numeric_limits<double>::epsilon()) {
    delta_q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis_angle.normalized()));
  } else {
    delta_q = Eigen::Quaterniond::Identity();
  }

  Eigen::Quaterniond predicted_q = delta_q * current_q;
  predicted_q.normalize();

  // Compute centripetal and tangential accelerations at the sensor location
  Eigen::Vector3d centripetal_accel = omega_after.cross(omega_after.cross(r));
  Eigen::Vector3d tangential_accel = a_angular.cross(r);

  // Compute gravity in body frame after prediction
  Eigen::Vector3d gravity_body = predicted_q.inverse() * g_W;

  // Solve for linear acceleration from accelerometer measurement
  // a_meas = R_SB^T * (a_linear + gravity_body + tangential_accel + centripetal_accel)
  // Rearranging: a_linear = R_SB * a_meas - gravity_body - tangential_accel - centripetal_accel
  Eigen::Vector3d a_linear_from_meas = R_SB.transpose() * corrected_accel -
                                      gravity_body -
                                      tangential_accel -
                                      centripetal_accel;

  // Optimal fusion of linear acceleration using LDLT decomposition
  Eigen::LDLT<Eigen::Matrix3d> solver_lin_state(scaled_lin_acc_cov);
  Eigen::LDLT<Eigen::Matrix3d> solver_lin_meas(lin_acc_meas_cov);

  // Calculate (C1⁻¹ + C2⁻¹) implicitly
  Eigen::Matrix3d lin_Cinv_sum = solver_lin_state.solve(Eigen::Matrix3d::Identity()) +
                                solver_lin_meas.solve(Eigen::Matrix3d::Identity());

  // Compute weighted sum: C1⁻¹x1 + C2⁻¹x2
  Eigen::Vector3d lin_weighted_sum = solver_lin_state.solve(current_lin_acc) +
                                   solver_lin_meas.solve(a_linear_from_meas);

  // Solve (C1⁻¹ + C2⁻¹)x̄ = weighted_sum
  Eigen::LDLT<Eigen::Matrix3d> lin_final_solver(lin_Cinv_sum);
  Eigen::Vector3d a_linear = lin_final_solver.solve(lin_weighted_sum);

  // Set the output accelerations
  inputs.segment<3>(0) = a_linear;
  inputs.segment<3>(3) = a_angular;

  return inputs;
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
    const StateFlags& ,
    StateVector& state,
    StateCovariance& covariance) const {

  StateFlags initialized_states = StateFlags::Zero();

  // Extract gyro and accelerometer measurements
  Eigen::Vector3d gyro = measurement.segment<3>(0);
  Eigen::Vector3d accel = measurement.segment<3>(3);

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
  Eigen::Matrix3d omega_body_cov = R_SB * gyro_cov * R_SB.transpose();
  covariance.block<3, 3>(
      StateIndex::AngularVelocity::Begin(),
      StateIndex::AngularVelocity::Begin()) = omega_body_cov;

  // Mark angular velocity as initialized
  initialized_states[StateIndex::AngularVelocity::X] = true;
  initialized_states[StateIndex::AngularVelocity::Y] = true;
  initialized_states[StateIndex::AngularVelocity::Z] = true;

  // Calculate accelerometer covariance in body frame
  Eigen::Matrix3d accel_cov = measurement_covariance_.block<3, 3>(3, 3);
  Eigen::Matrix3d accel_body_cov = R_SB * accel_cov * R_SB.transpose();

  // Check if IMU is stationary for orientation initialization
  bool is_stationary = IsStationary(state, covariance, measurement);

  // Always extract current yaw regardless of validity
  // Extract current quaternion
  Eigen::Quaterniond current_q(
      state(StateIndex::Quaternion::W),
      state(StateIndex::Quaternion::X),
      state(StateIndex::Quaternion::Y),
      state(StateIndex::Quaternion::Z)
  );
  current_q.normalize();

  // Convert to rotation matrix and extract yaw
  Eigen::Matrix3d rot_matrix = current_q.toRotationMatrix();
  double current_yaw = std::atan2(rot_matrix(1, 0), rot_matrix(0, 0));

  if (is_stationary) {
    // Calculate roll from accelerometer when stationary
    double ay = accel_body(1);
    double az = accel_body(2);
    double roll = std::atan2(ay, az);

    // Propagate uncertainty for roll
    double d_roll_d_ay = std::abs(az / (ay*ay + az*az));
    double d_roll_d_az = std::abs(-ay / (ay*ay + az*az));
    Eigen::Vector2d d_roll_d_ayz;
    d_roll_d_ayz << d_roll_d_ay, d_roll_d_az;
    double roll_var = d_roll_d_ayz.transpose() *
                     accel_body_cov.block<2,2>(1,1) *
                     d_roll_d_ayz;

    // Calculate pitch from accelerometer when stationary
    double ax = accel_body(0);
    double gravity = 9.80665; // Standard gravity in m/s^2
    double pitch = std::asin(-ax / gravity);

    // Propagate uncertainty for pitch
    double d_pitch_d_ax = std::abs(1.0 / (gravity * std::sqrt(1.0 - (ax*ax)/(gravity*gravity))));
    double pitch_var = d_pitch_d_ax * accel_body_cov(0,0) * d_pitch_d_ax;

    // Create rotation matrix from roll, pitch, and yaw
    Eigen::Matrix3d roll_rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d pitch_rot = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d yaw_rot = Eigen::AngleAxisd(current_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Combine rotations in XYZ order (roll-pitch-yaw)
    rot_matrix = roll_rot * pitch_rot * yaw_rot;
    // Convert to quaternion
    Eigen::Quaterniond q(rot_matrix);
    q.normalize();

    // Update state with quaternion
    state(StateIndex::Quaternion::W) = q.w();
    state(StateIndex::Quaternion::X) = q.x();
    state(StateIndex::Quaternion::Y) = q.y();
    state(StateIndex::Quaternion::Z) = q.z();

    // Set appropriate covariance for quaternion components
    double roll_pitch_variance = std::max(roll_var, pitch_var);

    // Create a mask matrix for yaw-related components (W and Z)
    Eigen::Matrix4d yaw_mask = Eigen::Matrix4d::Zero();
    yaw_mask(0, 0) = 1.0; // W component
    yaw_mask(3, 3) = 1.0; // Z component

    // Create a mask for roll/pitch components (W, X, Y)
    Eigen::Matrix4d roll_pitch_mask = Eigen::Matrix4d::Identity() - yaw_mask;

    // Combine the covariances:
    // - For roll/pitch components: use the newly calculated variance
    // - For yaw components: preserve the original covariance magnitude
    covariance.block<4, 4>(StateIndex::Quaternion::Begin(), StateIndex::Quaternion::Begin()) =
        (roll_pitch_mask * roll_pitch_variance) +
        (yaw_mask * covariance.block<4, 4>(StateIndex::Quaternion::Begin(), StateIndex::Quaternion::Begin()).norm());

    // Mark roll and pitch as initialized (quaternion components)
    initialized_states[StateIndex::Quaternion::W] = true;
    initialized_states[StateIndex::Quaternion::X] = true;
    initialized_states[StateIndex::Quaternion::Y] = true;

    // When stationary, linear acceleration should be zero in inertial frame
    // and angular acceleration should be zero
    state.segment<3>(StateIndex::LinearAcceleration::Begin()).setZero();
    state.segment<3>(StateIndex::AngularAcceleration::Begin()).setZero();

    // Set covariance for linear and angular acceleration
    covariance.block<3, 3>(
        StateIndex::LinearAcceleration::Begin(),
        StateIndex::LinearAcceleration::Begin()) = accel_body_cov;

    covariance.block<3, 3>(
        StateIndex::AngularAcceleration::Begin(),
        StateIndex::AngularAcceleration::Begin()) = omega_body_cov;

    // Mark linear and angular acceleration as initialized
    for (int i = 0; i < 3; i++) {
      initialized_states[StateIndex::LinearAcceleration::Begin() + i] = true;
      initialized_states[StateIndex::AngularAcceleration::Begin() + i] = true;
    }
  }

  return initialized_states;
}

} // namespace sensors
} // namespace kinematic_arbiter
