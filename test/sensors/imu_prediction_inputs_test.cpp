#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/trajectory_utils.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using core::StateIndex;
using utils::Figure8Trajectory;

// Define StateVector and StateMatrix with correct template parameters
using StateVector = Eigen::Matrix<double, core::StateIndex::kFullStateSize, 1>;
using StateMatrix = Eigen::Matrix<double, core::StateIndex::kFullStateSize, core::StateIndex::kFullStateSize>;


TEST(ImuPredictionInputsTest, CheckPredictionModelInputs) {
    // Set up the initial conditions
    double t = 0.05;  // Time at which the failure occurs

    // True state at t = 0.05
    StateVector true_state = Figure8Trajectory(t);

    // Predicted state (before update)
    StateVector initial_state = Figure8Trajectory(0);


    // Create IMU sensor model
    ImuSensorModel imu_model;

    // Create a dummy covariance matrix
    StateMatrix covariance = StateMatrix::Identity() * 0.1;

    // Get the measurement from the true state
    Eigen::Matrix<double, 6, 1> measurement = imu_model.PredictMeasurement(true_state);

    // Calculate prediction model inputs
    Eigen::Matrix<double, 6, 1> inputs = imu_model.GetPredictionModelInputs(
        initial_state,
        covariance,
        measurement,
        t);
    models::RigidBodyStateModel rigid_body_motion_model;
    StateVector predicted_state_with_inputs = rigid_body_motion_model.PredictStateWithInputAccelerations(
      initial_state, t, inputs.segment<3>(0),
      inputs.segment<3>(3));
    StateVector predicted_state_without_inputs = rigid_body_motion_model.PredictState(
    initial_state, t);
    initial_state.segment<3>(core::StateIndex::AngularVelocity::Begin()) = measurement.segment<3>(ImuSensorModel::MeasurementIndex::GX);
    StateVector predicted_state_with_vel = rigid_body_motion_model.PredictState(
    initial_state, t);

    StateVector error_with_inputs = predicted_state_with_inputs - true_state;
    StateVector error_without_inputs = predicted_state_without_inputs - true_state;
    StateVector error_with_vel = predicted_state_with_vel - true_state;
    // Position error
    Eigen::Vector3d position_error_with_inputs = error_with_inputs.segment<3>(core::StateIndex::Position::Begin());
    Eigen::Vector3d position_error_without_inputs = error_without_inputs.segment<3>(core::StateIndex::Position::Begin());
    Eigen::Vector3d position_error_with_vel = error_with_vel.segment<3>(core::StateIndex::Position::Begin());
    // Calculate angular differences
    double ang_error_w_inputs = fabs(Eigen::Quaterniond(
      predicted_state_with_inputs.segment<4>(core::StateIndex::Quaternion::Begin())).angularDistance(
        Eigen::Quaterniond(true_state.segment<4>(core::StateIndex::Quaternion::Begin()))));
    double ang_error_wo_inputs = fabs(Eigen::Quaterniond(
      predicted_state_without_inputs.segment<4>(core::StateIndex::Quaternion::Begin())).angularDistance(
        Eigen::Quaterniond(true_state.segment<4>(core::StateIndex::Quaternion::Begin()))));
    double ang_error_with_vel = fabs(Eigen::Quaterniond(predicted_state_with_vel.segment<4>(core::StateIndex::Quaternion::Begin())).angularDistance(
        Eigen::Quaterniond(true_state.segment<4>(core::StateIndex::Quaternion::Begin()))));
    // Velocity error
    Eigen::Matrix<double, 6, 1> velocity_error_with_inputs = error_with_inputs.segment<6>(core::StateIndex::LinearVelocity::Begin());
    Eigen::Matrix<double, 6, 1> velocity_error_without_inputs = error_without_inputs.segment<6>(core::StateIndex::LinearVelocity::Begin());
    Eigen::Matrix<double, 6, 1> velocity_error_with_vel = error_with_vel.segment<6>(core::StateIndex::LinearVelocity::Begin());
    // Acceleration error
    Eigen::Matrix<double, 6, 1> acceleration_error_with_inputs = error_with_inputs.segment<6>(core::StateIndex::LinearAcceleration::Begin());
    Eigen::Matrix<double, 6, 1> acceleration_error_without_inputs = error_without_inputs.segment<6>(core::StateIndex::LinearAcceleration::Begin());
    Eigen::Matrix<double, 6, 1> acceleration_error_with_vel = error_with_vel.segment<6>(core::StateIndex::LinearAcceleration::Begin());

    // Extract true accelerations from the true state
    Eigen::Matrix<double, 6, 1> true_accel;
    true_accel << true_state.segment<3>(core::StateIndex::LinearAcceleration::Begin()),
                  true_state.segment<3>(core::StateIndex::AngularAcceleration::Begin());

    Eigen::Isometry3d sensor_pose;
    EXPECT_TRUE(imu_model.GetSensorPoseInBodyFrame(sensor_pose));
    Eigen::Matrix3d R_SB = sensor_pose.rotation().matrix();

    EXPECT_TRUE(imu_model.CanPredictInputAccelerations());

    EXPECT_LE(position_error_with_inputs.norm(), position_error_without_inputs.norm())
        << "Position error with inputs is not less than position error without inputs\n"
        << "Position error with inputs: " << position_error_with_inputs.transpose() << "\n"
        << "Position error without inputs: " << position_error_without_inputs.transpose() << "\n"
        << "Position error with vel: " << position_error_with_vel.transpose() << "\n";
    EXPECT_LT(ang_error_w_inputs, ang_error_wo_inputs)
        << "Angular error with inputs is not less than angular error without inputs\n"
        << "Angular error with inputs: " << ang_error_w_inputs << "\n"
        << "Angular error without inputs: " << ang_error_wo_inputs << "\n"
        << "Angular error with vel: " << ang_error_with_vel << "\n";

    EXPECT_LT(velocity_error_with_inputs.norm(), velocity_error_without_inputs.norm())
        << "Velocity error with inputs is not less than velocity error without inputs\n"
        << "Velocity error with inputs: " << velocity_error_with_inputs.transpose() << "\n"
        << "Velocity error without inputs: " << velocity_error_without_inputs.transpose() << "\n"
        << "Velocity error with vel: " << velocity_error_with_vel.transpose() << "\n";

    EXPECT_LT(acceleration_error_with_inputs.norm(), acceleration_error_without_inputs.norm())
        << "Acceleration error with inputs is not less than acceleration error without inputs\n"
        << "Acceleration error with inputs: " << acceleration_error_with_inputs.transpose() << "\n"
        << "Acceleration error without inputs: " << acceleration_error_without_inputs.transpose() << "\n"
        << "Acceleration error with vel: " << acceleration_error_with_vel.transpose() << "\n";
    // Check if the calculated inputs are ballpark the true accelerations
    EXPECT_NEAR((inputs.segment<3>(3) - true_accel.segment<3>(3)).norm(), 0, 6.0)
        << "Angular acceleration inputs are not near enough to true Angular acceleration\n"
        << "Difference norm: " << (inputs.segment<3>(3) - true_accel.segment<3>(3)).norm() << "\n"
        << "Inputs: [" << inputs.segment<3>(3).transpose() << "]\n"
        << "True: [" << true_accel.segment<3>(3).transpose() << "]\n"
        << "Sensor to body frame rotation:\n" << R_SB << "\n";
}
}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
