#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "test/utils/test_trajectories.hpp"
#include <gtest/gtest.h>
#include <memory>

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
 * @brief Test the prediction accuracy with a figure-8 trajectory
 */
TEST_F(RigidBodyStateModelTest, Figure8Test) {
  constexpr double trajectory_duration = 10.0;
  constexpr double error_tolerance_per_second = 8e-5;

  double time = 0.0;
  StateVector true_state = testing::Figure8Trajectory(time);
  StateVector state_estimate = true_state;

  while (time < trajectory_duration) {
    time += testing::kTestTimeStep;
    true_state = testing::Figure8Trajectory(time);

    // Predict using our model
    state_estimate = model_->PredictState(state_estimate, testing::kTestTimeStep);

    // Set velocity and acceleration to actual states (simulating perfect sensors)
    // This allows us to isolate and test just the position prediction
    state_estimate.segment<12>(static_cast<int>(StateIndex::kLinearXDot)) =
        true_state.segment<12>(static_cast<int>(StateIndex::kLinearXDot));

    // Calculate position error
    double position_error = (
        state_estimate.segment<3>(static_cast<int>(StateIndex::kLinearX)) -
        true_state.segment<3>(static_cast<int>(StateIndex::kLinearX))
    ).norm();

    // Calculate acceptable error tolerance (grows with time)
    double error_tolerance =
        time < 1.0 ? error_tolerance_per_second
                   : testing::kFuzzyMoreThanZero + time * error_tolerance_per_second;

    EXPECT_LT(position_error, error_tolerance)
        << "Position error at time " << time << "s exceeds tolerance";
  }
}

/**
 * @brief Test the Jacobian approximation accuracy
 */
TEST_F(RigidBodyStateModelTest, JacobianTest) {
  constexpr double time = 1.0;  // Test at 1 second mark
  constexpr double epsilon = 1e-5;  // Small perturbation for finite difference
  constexpr double tolerance = 1e-3;  // Error tolerance for Jacobian approximation

  // Get a sample state from the figure-8 trajectory
  StateVector state = testing::Figure8Trajectory(time);

  // Get the analytical Jacobian from our model
  StateMatrix analytical_jacobian = model_->GetTransitionMatrix(state, testing::kTestTimeStep);

  // Compute numerical Jacobian with finite differences
  StateMatrix numerical_jacobian = StateMatrix::Zero();

  for (int i = 0; i < state.size(); ++i) {
    // Create perturbed states
    StateVector state_plus = state;
    state_plus[i] += epsilon;

    // Predict states
    StateVector prediction_unperturbed = model_->PredictState(state, testing::kTestTimeStep);
    StateVector prediction_perturbed = model_->PredictState(state_plus, testing::kTestTimeStep);

    // Compute finite difference approximation of derivative
    numerical_jacobian.col(i) = (prediction_perturbed - prediction_unperturbed) / epsilon;
  }

  // Compare analytical and numerical Jacobians
  for (int i = 0; i < state.size(); ++i) {
    for (int j = 0; j < state.size(); ++j) {
      // Skip checking zero or near-zero elements
      if (std::abs(analytical_jacobian(i, j)) > 1e-6 || std::abs(numerical_jacobian(i, j)) > 1e-6) {
        EXPECT_NEAR(analytical_jacobian(i, j), numerical_jacobian(i, j), tolerance)
            << "Jacobian mismatch at (" << i << ", " << j << ")";
      }
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
