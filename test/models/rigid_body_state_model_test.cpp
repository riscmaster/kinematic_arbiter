#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "test/utils/test_trajectories.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <string>

namespace kinematic_arbiter {
namespace models {
namespace test {

// Use the updated types consistently
using StateVector = Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1>;
using StateMatrix = Eigen::Matrix<double, core::StateIndex::kFullStateSize, core::StateIndex::kFullStateSize>;
using SIdx = core::StateIndex;

/**
 * @brief Test fixture for RigidBodyStateModel tests
 */
class RigidBodyStateModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create model with suitable parameters
    core::StateModelInterface::Params params;
    params.process_noise_window = 40;

    // Initialize model with parameters
    model_ = std::make_unique<RigidBodyStateModel>(params);
  }

  std::unique_ptr<RigidBodyStateModel> model_;
};

/**
 * @brief Parameterized test for basic motion scenarios
 */
struct MotionParams {
  std::string name;           // Test case name for identification

  // Initial state
  Eigen::Vector3d position;   // Initial position (x, y, z)
  Eigen::Vector4d quaternion; // Initial orientation (w, x, y, z)
  Eigen::Vector3d lin_vel;    // Initial linear velocity
  Eigen::Vector3d ang_vel;    // Initial angular velocity
  Eigen::Vector3d lin_acc;    // Linear acceleration
  Eigen::Vector3d ang_acc;    // Angular acceleration

  // Test parameters
  double dt;                  // Time step for prediction

  // Expected results
  Eigen::Vector3d exp_position;   // Expected position after dt
  Eigen::Vector4d exp_quaternion; // Expected orientation after dt
};

class BasicMotionTestP : public RigidBodyStateModelTest,
                         public ::testing::WithParamInterface<MotionParams> {};

/**
 * @brief Parameterized test for Figure8 trajectory with different durations and tolerances
 */
struct Figure8Params {
  double duration;          // Test duration in seconds
  double time_step;         // Prediction time step in seconds
  double error_per_second;  // Allowable position error growth rate
};

class Figure8TestP : public RigidBodyStateModelTest,
                     public ::testing::WithParamInterface<Figure8Params> {};

// Helper function to normalize a quaternion
Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& q) {
  return q.normalized();
}

// Helper function to create quaternion from axis-angle
Eigen::Vector4d axisAngleToQuaternion(const Eigen::Vector3d& axis, double angle) {
  Eigen::Vector4d q;
  double half_angle = angle * 0.5;
  double s = std::sin(half_angle);
  q[0] = std::cos(half_angle);  // w
  q[1] = axis[0] * s;           // x
  q[2] = axis[1] * s;           // y
  q[3] = axis[2] * s;           // z
  return q.normalized();
}

// Define test parameters
INSTANTIATE_TEST_SUITE_P(
    BasicMotions,
    BasicMotionTestP,
    ::testing::Values(
        // 1. Linear motion along X axis - constant velocity
        MotionParams{
            "X_Axis_ConstVel",
            {0.0, 0.0, 0.0},              // Initial position
            {1.0, 0.0, 0.0, 0.0},         // Initial quaternion (identity)
            {1.0, 0.0, 0.0},              // Linear velocity (1 m/s in X)
            {0.0, 0.0, 0.0},              // Angular velocity
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {1.0, 0.0, 0.0},              // Expected position after 1s
            {1.0, 0.0, 0.0, 0.0}          // Expected quaternion (unchanged)
        },

        // 2. Linear motion along X axis with acceleration
        MotionParams{
            "X_Axis_Accel",
            {0.0, 0.0, 0.0},              // Initial position
            {1.0, 0.0, 0.0, 0.0},         // Initial quaternion (identity)
            {0.0, 0.0, 0.0},              // Linear velocity (starting from rest)
            {0.0, 0.0, 0.0},              // Angular velocity
            {2.0, 0.0, 0.0},              // Linear acceleration (2 m/s² in X)
            {0.0, 0.0, 0.0},              // Angular acceleration
            2.0,                          // dt = 2 seconds
            {4.0, 0.0, 0.0},              // Expected position: x = 0.5*a*t² = 0.5*2*2² = 4
            {1.0, 0.0, 0.0, 0.0}          // Expected quaternion (unchanged)
        },

        // 3. Linear motion along Y axis - constant velocity
        MotionParams{
            "Y_Axis_ConstVel",
            {0.0, 0.0, 0.0},              // Initial position
            {1.0, 0.0, 0.0, 0.0},         // Initial quaternion (identity)
            {0.0, 1.0, 0.0},              // Linear velocity (1 m/s in Y)
            {0.0, 0.0, 0.0},              // Angular velocity
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {0.0, 1.0, 0.0},              // Expected position after 1s
            {1.0, 0.0, 0.0, 0.0}          // Expected quaternion (unchanged)
        },

        // 4. Linear motion along Z axis - constant velocity
        MotionParams{
            "Z_Axis_ConstVel",
            {0.0, 0.0, 0.0},              // Initial position
            {1.0, 0.0, 0.0, 0.0},         // Initial quaternion (identity)
            {0.0, 0.0, 1.0},              // Linear velocity (1 m/s in Z)
            {0.0, 0.0, 0.0},              // Angular velocity
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {0.0, 0.0, 1.0},              // Expected position after 1s
            {1.0, 0.0, 0.0, 0.0}          // Expected quaternion (unchanged)
        },

        // 5. Motion with initial orientation - 45° around Z
        MotionParams{
            "Motion_With_Initial_Rotation",
            {0.0, 0.0, 0.0},              // Initial position
            axisAngleToQuaternion(Eigen::Vector3d(0.0, 0.0, 1.0), M_PI/4.0),  // 45° around Z
            {1.0, 0.0, 0.0},              // Linear velocity (1 m/s in X)
            {0.0, 0.0, 0.0},              // Angular velocity
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {std::cos(M_PI/4.0), std::sin(M_PI/4.0), 0.0},  // Position reflects rotated velocity
            axisAngleToQuaternion(Eigen::Vector3d(0.0, 0.0, 1.0), M_PI/4.0)   // Quaternion unchanged
        },

        // 6. Motion with initial orientation - 90° around Y
        MotionParams{
            "Motion_With_Y_Rotation",
            {0.0, 0.0, 0.0},              // Initial position
            axisAngleToQuaternion(Eigen::Vector3d(0.0, 1.0, 0.0), M_PI/2.0),  // 90° around Y
            {1.0, 0.0, 0.0},              // Linear velocity (1 m/s in X)
            {0.0, 0.0, 0.0},              // Angular velocity
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {0.0, 0.0, -1.0},             // Position reflects rotated velocity (X becomes -Z)
            axisAngleToQuaternion(Eigen::Vector3d(0.0, 1.0, 0.0), M_PI/2.0)   // Quaternion unchanged
        },

        // 7. Combined linear and angular velocity
        MotionParams{
            "Combined_Linear_Angular_Vel",
            {0.0, 0.0, 0.0},              // Initial position
            {1.0, 0.0, 0.0, 0.0},         // Initial quaternion (identity)
            {1.0, 0.0, 0.0},              // Linear velocity (1 m/s in X)
            {0.0, 0.0, M_PI/2.0},         // Angular velocity (90°/s around Z)
            {0.0, 0.0, 0.0},              // Linear acceleration
            {0.0, 0.0, 0.0},              // Angular acceleration
            1.0,                          // dt = 1 second
            {1.0, 0.0, 0.0},              // Expected position after 1s
            axisAngleToQuaternion(Eigen::Vector3d(0.0, 0.0, 1.0), M_PI/2.0)   // Rotated 90° around Z
        }
    ),
    [](const ::testing::TestParamInfo<MotionParams>& info) {
        return info.param.name;  // Use the name field for test case names
    }
);

// Test case for Figure-8 trajectory
INSTANTIATE_TEST_SUITE_P(
    Figure8Trajectories,
    Figure8TestP,
    ::testing::Values(
        Figure8Params{10.0, 0.01, 0.001},   // 10 seconds, 10ms steps, 1mm/s error growth
        Figure8Params{60.0, 0.1, 0.002}     // 60 seconds, 100ms steps, 2mm/s error growth
    )
);

/**
 * @brief Test basic motion prediction accuracy
 */
TEST_P(BasicMotionTestP, PredictionAccuracy) {
  const MotionParams& params = GetParam();

  // Set up initial state
  StateVector state = StateVector::Zero();

  // Set position
  state[SIdx::Position::X] = params.position[0];
  state[SIdx::Position::Y] = params.position[1];
  state[SIdx::Position::Z] = params.position[2];

  // Set orientation
  state[SIdx::Quaternion::W] = params.quaternion[0];
  state[SIdx::Quaternion::X] = params.quaternion[1];
  state[SIdx::Quaternion::Y] = params.quaternion[2];
  state[SIdx::Quaternion::Z] = params.quaternion[3];

  // Set linear velocity
  state[SIdx::LinearVelocity::X] = params.lin_vel[0];
  state[SIdx::LinearVelocity::Y] = params.lin_vel[1];
  state[SIdx::LinearVelocity::Z] = params.lin_vel[2];

  // Set angular velocity
  state[SIdx::AngularVelocity::X] = params.ang_vel[0];
  state[SIdx::AngularVelocity::Y] = params.ang_vel[1];
  state[SIdx::AngularVelocity::Z] = params.ang_vel[2];

  // Set linear acceleration
  state[SIdx::LinearAcceleration::X] = params.lin_acc[0];
  state[SIdx::LinearAcceleration::Y] = params.lin_acc[1];
  state[SIdx::LinearAcceleration::Z] = params.lin_acc[2];

  // Set angular acceleration
  state[SIdx::AngularAcceleration::X] = params.ang_acc[0];
  state[SIdx::AngularAcceleration::Y] = params.ang_acc[1];
  state[SIdx::AngularAcceleration::Z] = params.ang_acc[2];

  // Predict state
  StateVector predicted_state = model_->PredictState(state, params.dt);

  // Extract predicted position
  Eigen::Vector3d predicted_position(
    predicted_state[SIdx::Position::X],
    predicted_state[SIdx::Position::Y],
    predicted_state[SIdx::Position::Z]
  );

  // Extract predicted quaternion
  Eigen::Vector4d predicted_quaternion(
    predicted_state[SIdx::Quaternion::W],
    predicted_state[SIdx::Quaternion::X],
    predicted_state[SIdx::Quaternion::Y],
    predicted_state[SIdx::Quaternion::Z]
  );

  // Normalize quaternions for comparison
  predicted_quaternion = normalizeQuaternion(predicted_quaternion);
  Eigen::Vector4d expected_quaternion = normalizeQuaternion(params.exp_quaternion);

  // Allow small numerical errors in quaternion comparisons
  // Note: Quaternions represent the same rotation when q = -q
  bool quaternion_match =
    (predicted_quaternion - expected_quaternion).norm() < 1e-3 ||
    (predicted_quaternion + expected_quaternion).norm() < 1e-3;

  // Check position accuracy
  EXPECT_NEAR(predicted_position.x(), params.exp_position.x(), 1e-3)
    << "Position X mismatch in test: " << params.name;
  EXPECT_NEAR(predicted_position.y(), params.exp_position.y(), 1e-3)
    << "Position Y mismatch in test: " << params.name;
  EXPECT_NEAR(predicted_position.z(), params.exp_position.z(), 1e-3)
    << "Position Z mismatch in test: " << params.name;

  // Check quaternion accuracy
  EXPECT_TRUE(quaternion_match)
    << "Quaternion mismatch in test: " << params.name
    << "\nExpected: " << expected_quaternion.transpose()
    << "\nActual: " << predicted_quaternion.transpose();
}

/**
 * @brief Test long-term prediction accuracy using Figure-8 trajectory
 */
TEST_P(Figure8TestP, LongTermPrediction) {
  const Figure8Params& params = GetParam();

  // Initial state from Figure-8 trajectory at t=0
  StateVector state = testing::Figure8Trajectory(0.0);

  // Initialize model state - copy only position and orientation
  // (leave velocity/acceleration at zero for initial prediction)
  StateVector prediction_state = StateVector::Zero();
  prediction_state.segment<3>(SIdx::Position::Begin()) = state.segment<3>(SIdx::Position::Begin());
  prediction_state.segment<4>(SIdx::Quaternion::Begin()) = state.segment<4>(SIdx::Quaternion::Begin());

  double time = 0.0;
  double max_position_error = 0.0;

  while (time < params.duration) {
    // Get ground truth state at current time
    StateVector ground_truth = testing::Figure8Trajectory(time);

    // Predict the next state using our model
    StateVector predicted_state = model_->PredictState(prediction_state, params.time_step);

    // After prediction, update the predicted state's velocity and acceleration
    // from ground truth to match reality for the next step (we're only testing
    // position/orientation prediction accuracy)
    predicted_state.segment<3>(SIdx::LinearVelocity::Begin()) =
        ground_truth.segment<3>(SIdx::LinearVelocity::Begin());
    predicted_state.segment<3>(SIdx::AngularVelocity::Begin()) =
        ground_truth.segment<3>(SIdx::AngularVelocity::Begin());
    predicted_state.segment<3>(SIdx::LinearAcceleration::Begin()) =
        ground_truth.segment<3>(SIdx::LinearAcceleration::Begin());
    predicted_state.segment<3>(SIdx::AngularAcceleration::Begin()) =
        ground_truth.segment<3>(SIdx::AngularAcceleration::Begin());

    // Get ground truth for next timestep to compare with our prediction
    StateVector next_ground_truth = testing::Figure8Trajectory(time + params.time_step);

    // Calculate position error between prediction and next ground truth
    Eigen::Vector3d predicted_position = predicted_state.segment<3>(SIdx::Position::Begin());
    Eigen::Vector3d ground_truth_position = next_ground_truth.segment<3>(SIdx::Position::Begin());
    double position_error = (predicted_position - ground_truth_position).norm();

    // Track maximum error
    max_position_error = std::max(max_position_error, position_error);

    // Store predicted state for next iteration
    prediction_state = predicted_state;

    // Advance time
    time += params.time_step;
  }

  // Error should be less than the allowable growth rate multiplied by duration
  double allowable_error = params.error_per_second * params.duration;

  // Use a slightly more relaxed tolerance to account for numerical differences
  double tolerance_factor = 1.1;  // 10% extra tolerance
  EXPECT_LT(max_position_error, allowable_error * tolerance_factor)
    << "Max position error exceeds allowable limit over " << params.duration << " seconds"
    << " (actual: " << max_position_error << "m, allowed: " << allowable_error << "m)";
}

/**
 * @brief Test fixture for quaternion Jacobian verification
 */
class QuaternionJacobianTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_ = std::make_unique<RigidBodyStateModel>();
  }

  std::unique_ptr<RigidBodyStateModel> model_;
};

/**
 * @brief Test the quaternion Jacobian for various angular velocities
 */
TEST_F(QuaternionJacobianTest, AngularVelocityJacobian) {
  // Test parameters
  const double time_step = 1e-8;  // Small time step for better linearization
  const double tolerance = 1e-8;  // Tolerance for comparison

  // Test with different angular velocities
  std::vector<Eigen::Vector3d> test_angular_velocities = {
    {0.1, 0.0, 0.0},  // X-axis rotation
    {0.0, 0.2, 0.0},  // Y-axis rotation
    {0.0, 0.0, 0.3},  // Z-axis rotation
    {0.1, 0.2, 0.3},  // Combined rotation
    {1.0, 0.0, 0.0},  // Faster X-axis rotation
    {0.0, 2.0, 0.0},  // Faster Y-axis rotation
    {0.0, 0.0, 3.0}   // Faster Z-axis rotation
  };

  for (const auto& angular_velocity : test_angular_velocities) {
    // Create a state with identity quaternion and specified angular velocity
    StateVector state = StateVector::Zero();

    // Set quaternion to identity
    state[SIdx::Quaternion::W] = 1.0;
    state[SIdx::Quaternion::X] = 0.0;
    state[SIdx::Quaternion::Y] = 0.0;
    state[SIdx::Quaternion::Z] = 0.0;

    // Set angular velocity
    state[SIdx::AngularVelocity::X] = angular_velocity[0];
    state[SIdx::AngularVelocity::Y] = angular_velocity[1];
    state[SIdx::AngularVelocity::Z] = angular_velocity[2];

    // Get the Jacobian transition matrix
    StateMatrix jacobian = model_->GetTransitionMatrix(state, time_step);

    // Predict next state using nonlinear model
    StateVector nonlinear_prediction = model_->PredictState(state, time_step);

    // Predict next state using linear approximation (Jacobian)
    StateVector linear_prediction = jacobian * state;

    // Extract quaternion components from both predictions
    Eigen::Vector4d nonlinear_quat(
      nonlinear_prediction[SIdx::Quaternion::W],
      nonlinear_prediction[SIdx::Quaternion::X],
      nonlinear_prediction[SIdx::Quaternion::Y],
      nonlinear_prediction[SIdx::Quaternion::Z]
    );

    Eigen::Vector4d linear_quat(
      linear_prediction[SIdx::Quaternion::W],
      linear_prediction[SIdx::Quaternion::X],
      linear_prediction[SIdx::Quaternion::Y],
      linear_prediction[SIdx::Quaternion::Z]
    );

    // Normalize quaternions for comparison
    nonlinear_quat.normalize();
    linear_quat.normalize();

    // Convert to Eigen::Quaterniond for angular difference calculation
    Eigen::Quaterniond q_nonlinear(
      nonlinear_quat[0], nonlinear_quat[1], nonlinear_quat[2], nonlinear_quat[3]
    );

    Eigen::Quaterniond q_linear(
      linear_quat[0], linear_quat[1], linear_quat[2], linear_quat[3]
    );

    // Calculate angular difference between quaternions
    // Account for quaternion double-cover by taking the absolute value of the dot product
    double dot_product = std::abs(q_nonlinear.dot(q_linear));
    dot_product = std::min(dot_product, 1.0);  // Clamp to avoid numerical issues
    double angle_diff_rad = 2.0 * std::acos(dot_product);
    double angle_diff_deg = angle_diff_rad * 180.0 / M_PI;

    // For small time steps, the linear approximation should be very close to nonlinear
    EXPECT_LT(angle_diff_rad, tolerance)
      << "Angular difference too large: " << angle_diff_deg << " degrees for angular velocity: ["
      << angular_velocity.transpose() << "]";

    // Also check individual quaternion components
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(nonlinear_quat[i], linear_quat[i], tolerance)
        << "Quaternion component " << i << " mismatch for angular velocity: ["
        << angular_velocity.transpose() << "]";
    }

    // Output test results for more detailed analysis when needed
    if (::testing::Test::HasFailure()) {
      std::cout << "Angular velocity: [" << angular_velocity.transpose() << "]" << std::endl;
      std::cout << "Nonlinear quaternion: " << nonlinear_quat.transpose() << std::endl;
      std::cout << "Linear quaternion: " << linear_quat.transpose() << std::endl;
      std::cout << "Angular difference: " << angle_diff_deg << " degrees" << std::endl;
      std::cout << "-----------------------------------" << std::endl;
    }
  }
}

} // namespace test
} // namespace models
} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
