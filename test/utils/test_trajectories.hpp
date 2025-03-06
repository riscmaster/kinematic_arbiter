#pragma once

#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace kinematic_arbiter {
namespace testing {

/**
 * @brief Generates a figure-8 trajectory for testing
 *
 * Creates a smooth figure-8 trajectory with position, velocity and acceleration.
 * This is useful for testing state prediction and measurement models.
 *
 * @param time Current time in seconds
 * @return State vector containing position, orientation, velocity and acceleration
 */
Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1> Figure8Trajectory(double time) {
  using SIdx = core::StateIndex;
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;

  // Dimensions of the figure-8
  constexpr double kMaxVelocity = 1.0;
  constexpr double kLength = 0.5;
  constexpr double kWidth = 0.25;
  constexpr double kWidthSlope = 0.1;  // radians
  constexpr double kAnglularScale = 1.0 / 10.0;

  // Corresponding variables to achieve the above dimensions
  const double kXAmplitude = kLength * 0.5;
  const double kYAmplitude = kWidth * 0.5;
  const double kZAmplitude = kWidth * std::tan(kWidthSlope);
  const double kPeriod =
      M_PI * std::sqrt(kXAmplitude * kXAmplitude +
                  4 * (kYAmplitude * kYAmplitude + kZAmplitude * kZAmplitude)) /
      kMaxVelocity;
  const double kXZFrequency = 2 * M_PI / kPeriod;
  const double kXFrequency = kXZFrequency * 0.5;

  // Create state vector
  StateVector states = StateVector::Zero();

  // Linear Position
  states[SIdx::Position::X] =
      kXAmplitude * std::cos(kXFrequency * time);
  states[SIdx::Position::Y] =
      kYAmplitude * std::sin(kXZFrequency * time);
  states[SIdx::Position::Z] =
      kZAmplitude * std::sin(kXZFrequency * time);

  // Angular Position (create quaternion from position-derived euler angles)
  Eigen::Vector3d position_vector = states.segment<3>(SIdx::Position::Begin());
  Eigen::Vector3d roll_pitch_yaw = position_vector * kAnglularScale;

  // Convert RPY to quaternion
  Eigen::AngleAxisd yawAngle(roll_pitch_yaw[2], Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd pitchAngle(roll_pitch_yaw[1], Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rollAngle(roll_pitch_yaw[0], Eigen::Vector3d::UnitX());
  Eigen::Quaterniond orientation = yawAngle * pitchAngle * rollAngle;

  // Set quaternion components
  states[SIdx::Quaternion::W] = orientation.w();
  states[SIdx::Quaternion::X] = orientation.x();
  states[SIdx::Quaternion::Y] = orientation.y();
  states[SIdx::Quaternion::Z] = orientation.z();

  // Get rotation matrix for body frame calculations
  Eigen::Matrix3d rotate_inertial_to_body = orientation.toRotationMatrix().transpose();

  // Velocity in inertial frame
  Eigen::Vector3d inertial_velocity(
      -kXAmplitude * kXFrequency * std::sin(kXFrequency * time),
      kYAmplitude * kXZFrequency * std::cos(kXZFrequency * time),
      kZAmplitude * kXZFrequency * std::cos(kXZFrequency * time));

  // Convert to body frame and set in state vector
  states.segment<3>(SIdx::LinearVelocity::Begin()) =
      rotate_inertial_to_body * inertial_velocity;

  // Angular velocity derived from position
  states.segment<3>(SIdx::AngularVelocity::Begin()) =
      inertial_velocity * kAnglularScale;

  // Acceleration in inertial frame
  Eigen::Vector3d inertial_acceleration(
      -kXAmplitude * kXFrequency * kXFrequency * std::cos(kXFrequency * time),
      -kYAmplitude * kXZFrequency * kXZFrequency * std::sin(kXZFrequency * time),
      -kZAmplitude * kXZFrequency * kXZFrequency * std::sin(kXZFrequency * time));

  // Convert to body frame and set in state vector
  states.segment<3>(SIdx::LinearAcceleration::Begin()) =
      rotate_inertial_to_body * inertial_acceleration;

  // Angular acceleration derived from linear acceleration
  states.segment<3>(SIdx::AngularAcceleration::Begin()) =
      inertial_acceleration * kAnglularScale;

  return states;
}

// Common test constants
constexpr double kTestTimeStep = 0.01;  // 100 Hz update rate
constexpr double kFuzzyMoreThanZero = 1e-10;

} // namespace testing
} // namespace kinematic_arbiter
