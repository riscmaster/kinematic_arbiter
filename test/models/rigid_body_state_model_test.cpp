#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "test/utils/test_trajectories.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <string>

namespace kinematic_arbiter {
namespace models {
namespace test {

// Add these using declarations to bring the types into scope
using StateVector = RigidBodyStateModel::StateVector;
using StateMatrix = RigidBodyStateModel::StateMatrix;

/**
 * @brief Test fixture for RigidBodyStateModel tests
 */
class RigidBodyStateModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create model with suitable noise parameters
    model_ = std::make_unique<RigidBodyStateModel>(
        0.01,   // process_noise_magnitude
        0.1     // initial_variance
    );
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

/**
 * @brief Test basic motion prediction with various configurations
 */
TEST_P(BasicMotionTestP, PredictionAccuracy) {
  const MotionParams& params = GetParam();

  // Create a zero-initialized state vector
  StateVector state = StateVector::Zero();

  // Set initial state
  state[static_cast<int>(StateIndex::kLinearX)] = params.position[0];
  state[static_cast<int>(StateIndex::kLinearY)] = params.position[1];
  state[static_cast<int>(StateIndex::kLinearZ)] = params.position[2];

  state[static_cast<int>(StateIndex::kQuaternionW)] = params.quaternion[0];
  state[static_cast<int>(StateIndex::kQuaternionX)] = params.quaternion[1];
  state[static_cast<int>(StateIndex::kQuaternionY)] = params.quaternion[2];
  state[static_cast<int>(StateIndex::kQuaternionZ)] = params.quaternion[3];

  state[static_cast<int>(StateIndex::kLinearXDot)] = params.lin_vel[0];
  state[static_cast<int>(StateIndex::kLinearYDot)] = params.lin_vel[1];
  state[static_cast<int>(StateIndex::kLinearZDot)] = params.lin_vel[2];

  state[static_cast<int>(StateIndex::kAngularXDot)] = params.ang_vel[0];
  state[static_cast<int>(StateIndex::kAngularYDot)] = params.ang_vel[1];
  state[static_cast<int>(StateIndex::kAngularZDot)] = params.ang_vel[2];

  state[static_cast<int>(StateIndex::kLinearXDDot)] = params.lin_acc[0];
  state[static_cast<int>(StateIndex::kLinearYDDot)] = params.lin_acc[1];
  state[static_cast<int>(StateIndex::kLinearZDDot)] = params.lin_acc[2];

  state[static_cast<int>(StateIndex::kAngularXDDot)] = params.ang_acc[0];
  state[static_cast<int>(StateIndex::kAngularYDDot)] = params.ang_acc[1];
  state[static_cast<int>(StateIndex::kAngularZDDot)] = params.ang_acc[2];

  // Perform prediction
  StateVector predicted_state = model_->PredictState(state, params.dt);

  // Extract predicted position and orientation
  Eigen::Vector3d predicted_position(
    predicted_state[static_cast<int>(StateIndex::kLinearX)],
    predicted_state[static_cast<int>(StateIndex::kLinearY)],
    predicted_state[static_cast<int>(StateIndex::kLinearZ)]
  );

  Eigen::Vector4d predicted_quaternion(
    predicted_state[static_cast<int>(StateIndex::kQuaternionW)],
    predicted_state[static_cast<int>(StateIndex::kQuaternionX)],
    predicted_state[static_cast<int>(StateIndex::kQuaternionY)],
    predicted_state[static_cast<int>(StateIndex::kQuaternionZ)]
  );

  // Define test tolerance
  constexpr double position_tolerance = 1e-10;
  constexpr double orientation_tolerance = 1e-6;  // Quaternion comparisons may need more tolerance

  // Check position (component-wise)
  EXPECT_NEAR(predicted_position[0], params.exp_position[0], position_tolerance)
      << "X position mismatch for test: " << params.name;
  EXPECT_NEAR(predicted_position[1], params.exp_position[1], position_tolerance)
      << "Y position mismatch for test: " << params.name;
  EXPECT_NEAR(predicted_position[2], params.exp_position[2], position_tolerance)
      << "Z position mismatch for test: " << params.name;

  // Normalize quaternions before comparison
  Eigen::Vector4d norm_predicted = predicted_quaternion.normalized();
  Eigen::Vector4d norm_expected = params.exp_quaternion.normalized();

  // Convert quaternions to Eigen::Quaterniond objects
  Eigen::Quaterniond q_predicted(norm_predicted[0], norm_predicted[1], norm_predicted[2], norm_predicted[3]);
  Eigen::Quaterniond q_expected(norm_expected[0], norm_expected[1], norm_expected[2], norm_expected[3]);

  // Calculate the relative rotation between the two quaternions
  Eigen::Quaterniond q_diff = q_predicted * q_expected.inverse();

  // Extract the angular error in radians
  // The angle is 2*acos(|w|) where w is the scalar part of the quaternion
  double angular_error_rad = 2.0 * std::acos(std::min(std::abs(q_diff.w()), 1.0));
  double angular_error_deg = angular_error_rad * 180.0 / M_PI;

  EXPECT_LT(angular_error_rad, orientation_tolerance)
      << "Orientation angular error for test: " << params.name
      << "\nPredicted: " << norm_predicted.transpose()
      << "\nExpected: " << norm_expected.transpose()
      << "\nAngular error: " << angular_error_rad << " rad ("
      << angular_error_deg << " deg)";
}

} // namespace test
} // namespace models
} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
