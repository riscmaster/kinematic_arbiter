#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include <drake/math/rotation_matrix.h>
#include <drake/math/roll_pitch_yaw.h>
#include <cmath>
#include <boost/math/distributions/chi_squared.hpp>

namespace kinematic_arbiter {
namespace core {

MediatedKalmanFilter::MediatedKalmanFilter(double time_step)
  : time_step_(time_step),
    is_initialized_(false),
    is_pose_initialized_(false) {

  // Initialize state and covariance matrices to zero
  state_estimate_.setZero();
  covariance_estimate_.setZero();
  process_covariance_.setZero();
  measurement_covariance_.setZero();
}

MediatedKalmanFilter::Result MediatedKalmanFilter::Validate(const Params& params) {
  if (params.noise_time_window == 0) {
    return Result::Failure("Noise time window must be greater than zero");
  }

  if (params.process_to_measurement_noise_ratio < 0) {
    return Result::Failure("Process to measurement noise ratio must be non-negative");
  }

  if (params.confidence_value <= 0 || params.confidence_value >= 1) {
    return Result::Failure("Confidence value must be in range (0, 1)");
  }

  if (params.initial_state_uncertainty <= 0) {
    return Result::Failure("Initial state uncertainty must be positive");
  }

  return Result::Success();
}

MediatedKalmanFilter::Result MediatedKalmanFilter::Initialize(const Params& params) {
  // Validate parameters
  Result validation_result = Validate(params);
  if (!validation_result.success) {
    return validation_result;
  }

  // Store parameters
  sample_window_ = params.noise_time_window;
  process_to_measurement_noise_ratio_ = params.process_to_measurement_noise_ratio;
  initial_process_variance_ = params.initial_state_uncertainty;

  // Calculate critical chi-squared value for the given confidence level
  // For pose measurements, we have 6 degrees of freedom (3D position + 3D orientation)
  boost::math::chi_squared chi_squared_dist(kPoseSize);
  critical_value_ = boost::math::quantile(chi_squared_dist, params.confidence_value);

  // Initialize state vector to zero
  state_estimate_.setZero();

  // Initialize covariance matrix with initial uncertainty
  covariance_estimate_.setIdentity();
  covariance_estimate_ *= initial_process_variance_;

  // Initialize process noise covariance
  process_covariance_.setIdentity();
  process_covariance_ *= initial_process_variance_;

  // Initialize measurement noise covariance
  measurement_covariance_.setIdentity();
  measurement_covariance_ *= initial_process_variance_ / process_to_measurement_noise_ratio_;

  is_initialized_ = true;
  return Result::Success();
}

void MediatedKalmanFilter::Initialize(const drake::math::RigidTransform<double>& initial_pose) {
  if (!is_initialized_) {
    // Set default parameters if not already initialized
    Params default_params;
    Initialize(default_params);
  }

  // Set the initial pose
  SetPoseEstimate(initial_pose);
}

void MediatedKalmanFilter::SetPoseEstimate(const drake::math::RigidTransform<double>& pose) {
  // Extract position
  Eigen::Vector3d position = pose.translation();

  // Extract orientation as roll-pitch-yaw
  drake::math::RotationMatrix<double> rotation(pose.rotation());
  drake::math::RollPitchYaw<double> rpy(rotation);
  Eigen::Vector3d orientation = rpy.vector();

  // Set position and orientation in state vector
  state_estimate_.segment<3>(0) = position;
  state_estimate_.segment<3>(3) = orientation;

  // Zero out velocities and accelerations
  state_estimate_.segment<12>(6).setZero();

  // Reset covariance to initial value
  covariance_estimate_.setIdentity();
  covariance_estimate_ *= initial_process_variance_;

  is_pose_initialized_ = true;
}

void MediatedKalmanFilter::SetStateEstimate(const StateVector& state_estimate) {
  state_estimate_ = state_estimate;

  // Reset covariance to initial value
  covariance_estimate_.setIdentity();
  covariance_estimate_ *= initial_process_variance_;

  is_pose_initialized_ = true;
}

StateVector MediatedKalmanFilter::Update(const drake::math::RigidTransform<double>& pose_measurement) {
  if (!is_initialized_ || !is_pose_initialized_) {
    // Initialize with the measurement if not already initialized
    Initialize(pose_measurement);
    return state_estimate_;
  }

  // Prediction step
  PredictStep();

  // Update step
  UpdateStep(pose_measurement);

  return state_estimate_;
}

void MediatedKalmanFilter::PredictStep() {
  // State transition matrix (constant velocity + constant acceleration model)
  StateMatrix F = StateMatrix::Identity();

  // Update positions with velocities and accelerations
  // Position += velocity * dt + 0.5 * acceleration * dt^2
  F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * time_step_;
  F.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * 0.5 * time_step_ * time_step_;

  // Update orientations with angular velocities and accelerations
  F.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity() * time_step_;
  F.block<3, 3>(3, 15) = Eigen::Matrix3d::Identity() * 0.5 * time_step_ * time_step_;

  // Update velocities with accelerations
  // Velocity += acceleration * dt
  F.block<3, 3>(6, 12) = Eigen::Matrix3d::Identity() * time_step_;
  F.block<3, 3>(9, 15) = Eigen::Matrix3d::Identity() * time_step_;

  // Predict state
  state_estimate_ = F * state_estimate_;

  // Predict covariance
  covariance_estimate_ = F * covariance_estimate_ * F.transpose() + process_covariance_;
}

void MediatedKalmanFilter::UpdateStep(const drake::math::RigidTransform<double>& measurement) {
  // Measurement matrix (we only measure position and orientation)
  Eigen::Matrix<double, kPoseSize, kStateSize> H = Eigen::Matrix<double, kPoseSize, kStateSize>::Zero();
  H.block<kPoseSize, kPoseSize>(0, 0) = Eigen::Matrix<double, kPoseSize, kPoseSize>::Identity();

  // Convert measurement to vector form
  PoseVector z = PoseToVector(measurement);

  // Calculate predicted measurement
  PoseVector z_pred = H * state_estimate_;

  // Calculate innovation (measurement residual)
  PoseVector innovation = z - z_pred;

  // Normalize angular components of innovation to [-π, π]
  for (int i = 3; i < 6; ++i) {
    innovation(i) = std::fmod(innovation(i) + M_PI, 2 * M_PI) - M_PI;
  }

  // Calculate innovation covariance
  PoseMatrix S = H * covariance_estimate_ * H.transpose() + measurement_covariance_;

  // Calculate Kalman gain
  Eigen::Matrix<double, kStateSize, kPoseSize> K =
      covariance_estimate_ * H.transpose() * S.inverse();

  // Calculate Mahalanobis distance for innovation validation
  double mahalanobis_distance = ComputeMahalanobisDistance(innovation, S);

  // Adapt noise covariances based on innovation statistics
  AdaptNoiseCovariances(innovation, mahalanobis_distance);

  // Update state estimate
  if (mahalanobis_distance <= critical_value_) {
    // Normal update
    state_estimate_ = state_estimate_ + K * innovation;
  } else {
    // Outlier detected - use robust update with scaled innovation
    double scaling_factor = std::sqrt(critical_value_ / mahalanobis_distance);
    state_estimate_ = state_estimate_ + K * (innovation * scaling_factor);
  }

  // Update covariance estimate
  covariance_estimate_ = (StateMatrix::Identity() - K * H) * covariance_estimate_;
}

MediatedKalmanFilter::PoseVector MediatedKalmanFilter::PoseToVector(
    const drake::math::RigidTransform<double>& pose) const {
  PoseVector pose_vector;

  // Extract position
  pose_vector.segment<3>(0) = pose.translation();

  // Extract orientation as roll-pitch-yaw
  drake::math::RotationMatrix<double> rotation(pose.rotation());
  drake::math::RollPitchYaw<double> rpy(rotation);
  pose_vector.segment<3>(3) = rpy.vector();

  return pose_vector;
}

double MediatedKalmanFilter::ComputeMahalanobisDistance(
    const PoseVector& innovation, const PoseMatrix& innovation_covariance) const {
  return innovation.transpose() * innovation_covariance.inverse() * innovation;
}

void MediatedKalmanFilter::AdaptNoiseCovariances(
    const PoseVector& innovation, double mahalanobis_distance) {
  // Simple adaptive scheme - increase process noise if innovation is large
  if (mahalanobis_distance > critical_value_) {
    // Outlier detected - increase process noise temporarily
    double scaling_factor = mahalanobis_distance / critical_value_;
    process_covariance_ *= std::min(scaling_factor, 2.0);  // Limit the increase
  } else {
    // Gradually return to nominal process noise
    process_covariance_ *= 0.99;
  }

  // Ensure process noise doesn't go below initial value
  for (int i = 0; i < kStateSize; ++i) {
    process_covariance_(i, i) = std::max(process_covariance_(i, i),
                                        initial_process_variance_ * 0.1);
  }
}

} // namespace core
} // namespace kinematic_arbiter
