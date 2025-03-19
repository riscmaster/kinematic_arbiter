#pragma once

#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace kinematic_arbiter {
namespace utils {

/**
 * @brief Configuration for Figure-8 trajectory generation
 */
struct Figure8Config {
  double max_velocity = 1.0;    // Maximum velocity along the trajectory
  double length = 0.5;          // Length of the figure-8 in X dimension
  double width = 0.25;          // Width of the figure-8 in Y dimension
  double width_slope = 0.1;     // Z inclination in radians
  double angular_scale = 0.1;   // Scale factor for angular motion
};

/**
 * @brief Generates a figure-8 trajectory for testing
 *
 * Creates a smooth figure-8 trajectory with position, velocity and acceleration.
 * This is useful for testing state prediction and measurement models.
 *
 * @param time Current time in seconds
 * @param config Configuration parameters for the trajectory (optional)
 * @return State vector containing position, orientation, velocity and acceleration
 */
Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1> Figure8Trajectory(
    double time,
    const Figure8Config& config = Figure8Config()) {

  using SIdx = core::StateIndex;
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;

  // Corresponding variables to achieve the dimensions
  const double kXAmplitude = config.length * 0.5;
  const double kYAmplitude = config.width * 0.5;
  const double kZAmplitude = config.width * std::tan(config.width_slope);
  const double kPeriod =
      M_PI * std::sqrt(kXAmplitude * kXAmplitude +
                  4 * (kYAmplitude * kYAmplitude + kZAmplitude * kZAmplitude)) /
      config.max_velocity;
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
  Eigen::Vector3d roll_pitch_yaw = position_vector * config.angular_scale;

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
      inertial_velocity * config.angular_scale;

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
      inertial_acceleration * config.angular_scale;

  return states;
}

} // namespace utils
} // namespace kinematic_arbiter
