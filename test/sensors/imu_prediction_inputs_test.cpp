#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "test/utils/test_trajectories.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using core::StateIndex;
using testing::Figure8Trajectory;

// Define StateVector and StateMatrix with correct template parameters
using StateVector = Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1>;
using StateMatrix = Eigen::Matrix<double, core::StateIndex::kFullStateSize, core::StateIndex::kFullStateSize>;

TEST(ImuPredictionInputsTest, CheckPredictionModelInputs) {
    // Set up the initial conditions
    double t = 0.05;  // Time at which the failure occurs

    // True state at t = 0.05
    StateVector true_state = Figure8Trajectory(t);

    // Predicted state (before update)
    StateVector predicted_state = Figure8Trajectory(0);

    // Create IMU sensor model
    ImuSensorModel imu_model;

    // Create a dummy covariance matrix
    StateMatrix covariance = StateMatrix::Identity() * 0.1;

    // Get the measurement from the true state
    Eigen::Matrix<double, 6, 1> measurement = imu_model.PredictMeasurement(true_state);

    // Calculate prediction model inputs
    Eigen::Matrix<double, 6, 1> inputs = imu_model.GetPredictionModelInputs(
        predicted_state,
        covariance,
        measurement,
        t);

    // Extract true accelerations from the true state
    Eigen::Matrix<double, 6, 1> true_accel;
    true_accel << true_state.segment<3>(core::StateIndex::LinearAcceleration::Begin()),
                  true_state.segment<3>(core::StateIndex::AngularAcceleration::Begin());

    // Check if the calculated inputs match the true accelerations
    ASSERT_NEAR((inputs - true_accel).norm(), 0, 1e-8)
        << "Linear acceleration inputs are not equal to true linear acceleration\n"
        << "Inputs: " << inputs.transpose() << "\n"
        << "True: " << true_accel.transpose() << "\n";
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
