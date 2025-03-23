#include <gtest/gtest.h>
#include <kinematic_arbiter/core/trajectory_utils.hpp>

class KinematicConsistencyTest : public ::testing::Test {
protected:
  // Helper to extract state components
  struct StateComponents {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_acceleration;

    // Extract components from state vector
    StateComponents(const Eigen::VectorXd& state) {
      using SIdx = kinematic_arbiter::core::StateIndex;

      position = state.segment<3>(SIdx::Position::Begin());

      orientation = Eigen::Quaterniond(
          state[SIdx::Quaternion::W],
          state[SIdx::Quaternion::X],
          state[SIdx::Quaternion::Y],
          state[SIdx::Quaternion::Z]
      );
      orientation.normalize();

      linear_velocity = state.segment<3>(SIdx::LinearVelocity::Begin());
      angular_velocity = state.segment<3>(SIdx::AngularVelocity::Begin());
      linear_acceleration = state.segment<3>(SIdx::LinearAcceleration::Begin());
      angular_acceleration = state.segment<3>(SIdx::AngularAcceleration::Begin());
    }
  };
};

TEST_F(KinematicConsistencyTest, TemporalConsistency) {
  const double dt = 0.0001; // Very small time step for consistency checks

  for (double t = 0.0; t < 10.0; t += 0.5) {
    auto state_curr = StateComponents(kinematic_arbiter::utils::Figure8Trajectory(t));
    auto state_next = StateComponents(kinematic_arbiter::utils::Figure8Trajectory(t + dt));

    // Position should change according to velocity
    Eigen::Vector3d world_velocity = state_curr.orientation * state_curr.linear_velocity;
    Eigen::Vector3d expected_pos_change = world_velocity * dt;
    Eigen::Vector3d actual_pos_change = state_next.position - state_curr.position;

    EXPECT_NEAR((expected_pos_change - actual_pos_change).norm(), 0.0, dt * dt * 10)
        << "Position change inconsistent with velocity at t=" << t;

    // Orientation should change according to angular velocity
    if (state_curr.angular_velocity.norm() > 1e-6) {
      // Create quaternion representing rotation due to angular velocity
      Eigen::Vector3d axis = state_curr.angular_velocity.normalized();
      double angle = state_curr.angular_velocity.norm() * dt;
      Eigen::Quaterniond q_delta(Eigen::AngleAxisd(angle, axis));

      // Predict next orientation
      Eigen::Quaterniond expected_orientation = state_curr.orientation * q_delta;
      expected_orientation.normalize();

      // Angular distance between predicted and actual orientation
      double orientation_diff = 1.0 - std::abs(expected_orientation.dot(state_next.orientation));
      EXPECT_NEAR(orientation_diff, 0.0, dt * dt * 100)
          << "Orientation change inconsistent with angular velocity at t=" << t;
    }
  }
}

TEST_F(KinematicConsistencyTest, AccelerationConsistency) {
  const double dt = 0.0001; // Very small time step for consistency checks

  for (double t = 0.0; t < 10.0; t += 0.5) {
    auto state_curr = StateComponents(kinematic_arbiter::utils::Figure8Trajectory(t));
    auto state_next = StateComponents(kinematic_arbiter::utils::Figure8Trajectory(t + dt));

    // Velocity should change according to acceleration + Coriolis effect
    Eigen::Vector3d expected_vel_change = state_curr.linear_acceleration * dt;

    // Coriolis contribution: ω × v * dt
    expected_vel_change += state_curr.angular_velocity.cross(state_curr.linear_velocity) * dt;

    Eigen::Vector3d actual_vel_change = state_next.linear_velocity - state_curr.linear_velocity;

    // Allow relatively high tolerance due to higher-order effects
    EXPECT_NEAR((expected_vel_change - actual_vel_change).norm(), 0.0, dt * 10)
        << "Velocity change inconsistent with acceleration at t=" << t;

    // Angular velocity should change according to angular acceleration
    if (state_curr.angular_acceleration.norm() > 1e-6) {
      Eigen::Vector3d expected_ang_vel_change = state_curr.angular_acceleration * dt;
      Eigen::Vector3d actual_ang_vel_change = state_next.angular_velocity - state_curr.angular_velocity;

      // Verify direction is consistent (dot product > 0)
      if (actual_ang_vel_change.norm() > dt * dt) {
        double alignment = expected_ang_vel_change.dot(actual_ang_vel_change) /
                          (expected_ang_vel_change.norm() * actual_ang_vel_change.norm());
        EXPECT_GT(alignment, 0.0)
            << "Angular acceleration not aligned with angular velocity change at t=" << t;
      }
    }
  }
}
