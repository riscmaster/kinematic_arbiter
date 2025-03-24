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
  double angular_scale = 0.01;   // Scale factor for angular motion
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

  // Trajectory parameters
  const double a = config.length * 0.5;        // x amplitude
  const double b = config.width * 0.5;         // y amplitude
  const double c = config.width * std::tan(config.width_slope); // z amplitude

  // Period and frequency calculations
  const double T = M_PI * std::sqrt(a*a + 4*(b*b + c*c)) / config.max_velocity;
  const double w1 = 2 * M_PI / T;    // y,z frequency
  const double w2 = w1 * 0.5;        // x frequency
  const double t = time;             // current time

  // Create state vector
  StateVector states = StateVector::Zero();

  // Position vector in world frame
  Eigen::Vector3d position(
      a * std::cos(w2 * t),
      b * std::sin(w1 * t),
      c * std::sin(w1 * t));

  // Store position
  states.segment<3>(SIdx::Position::Begin()) = position;

  // Linear Inertial (World Frame) Velocity
  Eigen::Vector3d i_v(
      -a * w2 * std::sin(w2 * t),
      b * w1 * std::cos(w1 * t),
      c * w1 * std::cos(w1 * t));

  // Linear Inertial (World Frame) Acceleration
  Eigen::Vector3d i_a(
      -a * w2 * w2 * std::cos(w2 * t),
      -b * w1 * w1 * std::sin(w1 * t),
      -c * w1 * w1 * std::sin(w1 * t));

  // Linear Inertial (World Frame) Jerk
  Eigen::Vector3d i_j(
      a * w2 * w2 * w2 * std::sin(w2 * t),
      -b * w1 * w1 * w1 * std::cos(w1 * t),
      -c * w1 * w1 * w1 * std::cos(w1 * t));

  // Current Orientation (body X aligned with velocity)
  Eigen::Quaterniond orientation;
  orientation.setFromTwoVectors(Eigen::Vector3d::UnitX(), i_v.normalized());
  orientation.normalize();

  // Store quaternion
  states[SIdx::Quaternion::W] = orientation.w();
  states[SIdx::Quaternion::X] = orientation.x();
  states[SIdx::Quaternion::Y] = orientation.y();
  states[SIdx::Quaternion::Z] = orientation.z();

  // Convert world velocity to body frame
  Eigen::Vector3d b_v = orientation.inverse() * i_v;
  states.segment<3>(SIdx::LinearVelocity::Begin()) = b_v;

  // 1. Compute angular velocity in world frame
  Eigen::Vector3d omega_world;
  if (i_v.squaredNorm() > 1e-9) {
      omega_world = i_v.cross(i_a) / i_v.squaredNorm();
  } else {
      omega_world.setZero();
  }

  // 2. Transform to body frame
  Eigen::Vector3d omega_body = orientation.inverse() * omega_world;
  states.segment<3>(SIdx::AngularVelocity::Begin()) = omega_body;

  // 3. Compute body-frame acceleration PROPERLY
  Eigen::Vector3d b_a = orientation.inverse() * i_a;
  b_a -= 2 * omega_body.cross(b_v); // Full Coriolis compensation
  states.segment<3>(SIdx::LinearAcceleration::Begin()) = b_a;

  // 4. Angular acceleration (derivative of omega_world)
  Eigen::Vector3d alpha_world;
  if (i_v.squaredNorm() > 1e-9) {
      alpha_world = (i_v.cross(i_j) * i_v.squaredNorm() -
                    2 * i_v.cross(i_a) * i_v.dot(i_a))
                    / (i_v.squaredNorm() * i_v.squaredNorm());
  } else {
      alpha_world.setZero();
  }

  // 5. Transform to body frame
  Eigen::Vector3d alpha_body = orientation.inverse() * alpha_world;
  states.segment<3>(SIdx::AngularAcceleration::Begin()) = alpha_body;

  return states;
}

// /**
//  * @brief Configuration for oscillating trajectory generation
//  */
// struct OscillatingConfig {
//   Eigen::Vector3d max_linear_accel = Eigen::Vector3d(10.0, 0.0, 0.0); // Max linear acceleration (m/s^2)
//   Eigen::Vector3d max_angular_accel = Eigen::Vector3d(0.0, 0.0, 0.0); // Max angular acceleration (rad/s^2)

//   double linear_time_period = 0.5;  // Time period for linear acceleration phases (seconds)
//   double angular_time_period = 0.1; // Time period for angular acceleration phases (seconds)
// };

// /**
//  * @brief Generates an oscillating trajectory for testing
//  *
//  * Creates a trajectory that follows an acceleration pattern: 0, max, -max, 0, -max, max.
//  * This is useful for testing state prediction, control models, and inertial compensation.
//  *
//  * @param time Current time in seconds
//  * @param start_time Starting time of the trajectory
//  * @param config Configuration parameters for the trajectory (optional)
//  * @return State vector containing position, orientation, velocity, and acceleration
//  */
// Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1> OscillatingTrajectory(
//     double time,
//     double start_time,
//     const OscillatingConfig& config = OscillatingConfig()) {

//   using SIdx = core::StateIndex;
//   using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;

//   // Create state vector
//   StateVector states = StateVector::Zero();

//   double adjusted_time = time - start_time;

//   // Linear Motion Calculation
//   const double phase_duration_lin = config.linear_time_period / 5.0;
//   double t_phase_lin = std::fmod(adjusted_time, 5 * phase_duration_lin);

//   for (int i = 0; i < 3; ++i) {
//     double pos = 0.0, vel = 0.0, acc = 0.0;

//     if (t_phase_lin < phase_duration_lin) {
//       acc = 0.0;
//       pos = 0.0;
//       vel = 0.0;
//     } else if (t_phase_lin < 2 * phase_duration_lin) {
//       acc = config.max_linear_accel[i];
//       pos = 0.5 * acc * (t_phase_lin - phase_duration_lin) * (t_phase_lin - phase_duration_lin);
//       vel = acc * (t_phase_lin - phase_duration_lin);
//     } else if (t_phase_lin < 3 * phase_duration_lin) {
//       acc = -config.max_linear_accel[i];
//       pos = acc * phase_duration_lin * (t_phase_lin - 2 * phase_duration_lin) -
//             0.5 * acc * (t_phase_lin - 2 * phase_duration_lin) * (t_phase_lin - 2 * phase_duration_lin);
//       vel = acc * phase_duration_lin - acc * (t_phase_lin - 2 * phase_duration_lin);
//     } else if (t_phase_lin < 4 * phase_duration_lin) {
//       acc = -config.max_linear_accel[i];
//       pos = -0.5 * acc * (t_phase_lin - 3 * phase_duration_lin) * (t_phase_lin - 3 * phase_duration_lin);
//       vel = -acc * (t_phase_lin - 3 * phase_duration_lin);
//     } else {
//       acc = config.max_linear_accel[i];
//       pos = -acc * phase_duration_lin * (t_phase_lin - 4 * phase_duration_lin) +
//             0.5 * acc * (t_phase_lin - 4 * phase_duration_lin) * (t_phase_lin - 4 * phase_duration_lin);
//       vel = -acc * phase_duration_lin + acc * (t_phase_lin - 4 * phase_duration_lin);
//     }

//     states[SIdx::Position::Begin() + i] = pos;
//     states[SIdx::LinearVelocity::Begin() + i] = vel;
//     states[SIdx::LinearAcceleration::Begin() + i] = acc;
//   }

//   // Angular Motion Calculation
//   const double phase_duration_ang = config.angular_time_period / 5.0;
//   double t_phase_ang = std::fmod(adjusted_time, 5 * phase_duration_ang);

//   for (int i = 0; i < 3; ++i) {
//     double ang_pos = 0.0, ang_vel = 0.0, ang_acc = 0.0;

//     if (t_phase_ang < phase_duration_ang) {
//       ang_acc = 0.0;
//       ang_pos = 0.0;
//       ang_vel = 0.0;
//     } else if (t_phase_ang < 2 * phase_duration_ang) {
//       ang_acc = config.max_angular_accel[i];
//       ang_pos = 0.5 * ang_acc * (t_phase_ang - phase_duration_ang) * (t_phase_ang - phase_duration_ang);
//       ang_vel = ang_acc * (t_phase_ang - phase_duration_ang);
//     } else if (t_phase_ang < 3 * phase_duration_ang) {
//       ang_acc = -config.max_angular_accel[i];
//       ang_pos = ang_acc * phase_duration_ang * (t_phase_ang - 2 * phase_duration_ang) -
//                  0.5 * ang_acc * (t_phase_ang - 2 * phase_duration_ang) * (t_phase_ang - 2 * phase_duration_ang);
//       ang_vel = ang_acc * phase_duration_ang - ang_acc * (t_phase_ang - 2 * phase_duration_ang);
//     } else if (t_phase_ang < 4 * phase_duration_ang) {
//       ang_acc = -config.max_angular_accel[i];
//       ang_pos = -0.5 * ang_acc * (t_phase_ang - 3 * phase_duration_ang) * (t_phase_ang - 3 * phase_duration_ang);
//       ang_vel = -ang_acc * (t_phase_ang - 3 * phase_duration_ang);
//     } else {
//       ang_acc = config.max_angular_accel[i];
//       ang_pos = -ang_acc * phase_duration_ang * (t_phase_ang - 4 * phase_duration_ang) +
//                  0.5 * ang_acc * (t_phase_ang - 4 * phase_duration_ang) * (t_phase_ang - 4 * phase_duration_ang);
//       ang_vel = -ang_acc * phase_duration_ang + ang_acc * (t_phase_ang - 4 * phase_duration_ang);
//     }

//     states[SIdx::AngularVelocity::Begin() + i] = ang_vel;
//     states[SIdx::AngularAcceleration::Begin() + i] = ang_acc;
//   }

//   return states;
// }


} // namespace utils
} // namespace kinematic_arbiter
