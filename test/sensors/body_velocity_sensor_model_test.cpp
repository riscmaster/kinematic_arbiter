#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace kinematic_arbiter {
namespace sensors {
namespace testing {

class BodyVelocitySensorModelTest : public ::testing::Test {
protected:
  // Define types first before using them
  using StateVector = BodyVelocitySensorModel::StateVector;
  using MeasurementVector = BodyVelocitySensorModel::MeasurementVector;
  using MeasurementJacobian = BodyVelocitySensorModel::MeasurementJacobian;

  // Constants for clearer indexing
  static constexpr int MEAS_LIN_VEL = BodyVelocitySensorModel::MeasurementIndex::VX;
  static constexpr int MEAS_ANG_VEL = BodyVelocitySensorModel::MeasurementIndex::WX;
  static constexpr int STATE_LIN_VEL = core::StateIndex::LinearVelocity::X;
  static constexpr int STATE_ANG_VEL = core::StateIndex::AngularVelocity::X;

  void SetUp() override {
    // Define test velocity values
    body_lin_vel_ << 0.5, -0.3, 0.1;
    body_ang_vel_ << 0.1, 0.2, -0.3;

    // Initialize state vector
    test_state_.setZero();
    setBodyVelocities(test_state_, body_lin_vel_, body_ang_vel_);

    // Create a sensor transform with 1m x-offset and 45° z-rotation
    sensor_offset_ = Eigen::Vector3d(1.0, 0.0, 0.0);
    sensor_rotation_angle_ = M_PI / 4.0;  // 45 degrees
    sensor_rotation_axis_ = Eigen::Vector3d::UnitZ();

    sensor_transform_ = createTransform(sensor_offset_, sensor_rotation_angle_, sensor_rotation_axis_);
  }

  // Helper method to set velocity components in a state vector
  void setBodyVelocities(StateVector& state, const Eigen::Vector3d& lin_vel,
                        const Eigen::Vector3d& ang_vel) const {
    state.segment<3>(STATE_LIN_VEL) = lin_vel;
    state.segment<3>(STATE_ANG_VEL) = ang_vel;
  }

  // Helper method to create a transform from components
  Eigen::Isometry3d createTransform(const Eigen::Vector3d& translation,
                                   double angle, const Eigen::Vector3d& axis) const {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = translation;
    transform.linear() = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    return transform;
  }

  // Helper methods for extracting measurement components
  Eigen::Vector3d getLinearVelocity(const MeasurementVector& measurement) const {
    return measurement.segment<3>(MEAS_LIN_VEL);
  }

  Eigen::Vector3d getAngularVelocity(const MeasurementVector& measurement) const {
    return measurement.segment<3>(MEAS_ANG_VEL);
  }

  // Helper to create skew-symmetric matrix for cross products
  Eigen::Matrix3d createSkewMatrix(const Eigen::Vector3d& vector) const {
    Eigen::Matrix3d skew;
    skew <<  0,         -vector(2),  vector(1),
             vector(2),  0,         -vector(0),
            -vector(1),  vector(0),  0;
    return skew;
  }

  // Helper to compute cross product using the skew matrix
  Eigen::Vector3d crossProduct(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
    return createSkewMatrix(a) * b;
  }

  // Helper to compute expected sensor measurements
  struct ExpectedMeasurement {
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
  };

  ExpectedMeasurement computeExpectedMeasurement(
      const Eigen::Vector3d& body_lin_vel,
      const Eigen::Vector3d& body_ang_vel,
      const Eigen::Isometry3d& sensor_transform) const {

    const Eigen::Vector3d& offset = sensor_transform.translation();
    const Eigen::Matrix3d& rotation = sensor_transform.linear();
    const Eigen::Matrix3d sensor_to_body_rotation = rotation.transpose();

    // v_sensor = R_s_b * (v_body + ω × r)
    Eigen::Vector3d sensor_lin_vel = sensor_to_body_rotation *
        (body_lin_vel + crossProduct(body_ang_vel, offset));

    // ω_sensor = R_s_b * ω_body
    Eigen::Vector3d sensor_ang_vel = sensor_to_body_rotation * body_ang_vel;

    return {sensor_lin_vel, sensor_ang_vel};
  }

  // Test data
  StateVector test_state_;
  Eigen::Vector3d body_lin_vel_;
  Eigen::Vector3d body_ang_vel_;
  Eigen::Vector3d sensor_offset_;
  double sensor_rotation_angle_;
  Eigen::Vector3d sensor_rotation_axis_;
  Eigen::Isometry3d sensor_transform_;
};

// Test default construction
TEST_F(BodyVelocitySensorModelTest, DefaultInitialization) {
  BodyVelocitySensorModel model;

  // Default model should use identity transform
  auto measurement = model.PredictMeasurement(test_state_);

  EXPECT_TRUE(getLinearVelocity(measurement).isApprox(body_lin_vel_));
  EXPECT_TRUE(getAngularVelocity(measurement).isApprox(body_ang_vel_));
}

// Test measurement prediction with identity transform
TEST_F(BodyVelocitySensorModelTest, PredictMeasurementIdentity) {
  const Eigen::Isometry3d identity_transform = Eigen::Isometry3d::Identity();
  BodyVelocitySensorModel model(identity_transform);

  const auto measurement = model.PredictMeasurement(test_state_);
  const auto expected = computeExpectedMeasurement(body_lin_vel_, body_ang_vel_, identity_transform);

  EXPECT_TRUE(getLinearVelocity(measurement).isApprox(expected.linear_velocity));
  EXPECT_TRUE(getAngularVelocity(measurement).isApprox(expected.angular_velocity));
}

// Test measurement prediction with offset transform
TEST_F(BodyVelocitySensorModelTest, PredictMeasurementWithTransform) {
  BodyVelocitySensorModel model(sensor_transform_);

  const auto measurement = model.PredictMeasurement(test_state_);
  const auto expected = computeExpectedMeasurement(body_lin_vel_, body_ang_vel_, sensor_transform_);

  EXPECT_TRUE(getLinearVelocity(measurement).isApprox(expected.linear_velocity));
  EXPECT_TRUE(getAngularVelocity(measurement).isApprox(expected.angular_velocity));
}

// Test the structure of the measurement Jacobian
TEST_F(BodyVelocitySensorModelTest, MeasurementJacobianStructure) {
  const Eigen::Isometry3d identity_transform = Eigen::Isometry3d::Identity();
  BodyVelocitySensorModel model(identity_transform);

  const auto jacobian = model.GetMeasurementJacobian(test_state_);

  // With identity transform, we expect simple structure in the Jacobian
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d zeros = Eigen::Matrix3d::Zero();

  // Extract Jacobian blocks for better readability
  const Eigen::Matrix3d lin_vel_wrt_lin_vel = jacobian.block<3, 3>(MEAS_LIN_VEL, STATE_LIN_VEL);
  const Eigen::Matrix3d lin_vel_wrt_ang_vel = jacobian.block<3, 3>(MEAS_LIN_VEL, STATE_ANG_VEL);
  const Eigen::Matrix3d ang_vel_wrt_ang_vel = jacobian.block<3, 3>(MEAS_ANG_VEL, STATE_ANG_VEL);

  EXPECT_TRUE(lin_vel_wrt_lin_vel.isApprox(identity));
  EXPECT_TRUE(lin_vel_wrt_ang_vel.isApprox(zeros));
  EXPECT_TRUE(ang_vel_wrt_ang_vel.isApprox(identity));
}

// Test measurement Jacobian with offset transform
TEST_F(BodyVelocitySensorModelTest, MeasurementJacobianWithTransform) {
  BodyVelocitySensorModel model(sensor_transform_);

  const auto jacobian = model.GetMeasurementJacobian(test_state_);

  // Extract transform components for verification
  const Eigen::Matrix3d& rotation = sensor_transform_.linear();
  const Eigen::Matrix3d sensor_to_body_rotation = rotation.transpose();

  // Extract Jacobian blocks for better readability
  const Eigen::Matrix3d lin_vel_wrt_lin_vel = jacobian.block<3, 3>(MEAS_LIN_VEL, STATE_LIN_VEL);
  const Eigen::Matrix3d lin_vel_wrt_ang_vel = jacobian.block<3, 3>(MEAS_LIN_VEL, STATE_ANG_VEL);
  const Eigen::Matrix3d ang_vel_wrt_ang_vel = jacobian.block<3, 3>(MEAS_ANG_VEL, STATE_ANG_VEL);

  EXPECT_TRUE(lin_vel_wrt_lin_vel.isApprox(sensor_to_body_rotation));
  EXPECT_TRUE(ang_vel_wrt_ang_vel.isApprox(sensor_to_body_rotation));
  EXPECT_FALSE(lin_vel_wrt_ang_vel.isApprox(Eigen::Matrix3d::Zero()));
}

// Test numerical validation of the Jacobian
TEST_F(BodyVelocitySensorModelTest, NumericalJacobianValidation) {
  BodyVelocitySensorModel model(sensor_transform_);

  // Get analytical Jacobian and baseline measurement
  const auto analytical_jacobian = model.GetMeasurementJacobian(test_state_);
  const auto baseline_measurement = model.PredictMeasurement(test_state_);

  // Define small perturbation for numerical differentiation
  const double epsilon = 1e-6;
  const int measurement_size = baseline_measurement.size();

  // Test each state variable that affects the measurement
  const int velocity_state_dims = 6; // 3 linear + 3 angular
  for (int i = 0; i < velocity_state_dims; i++) {
    // Create perturbed state
    StateVector perturbed_state = test_state_;
    const int state_idx = (i < 3) ? STATE_LIN_VEL + i : STATE_ANG_VEL + (i - 3);
    perturbed_state(state_idx) += epsilon;

    // Compute perturbed measurement and numerical derivative
    const auto perturbed_measurement = model.PredictMeasurement(perturbed_state);
    const Eigen::VectorXd numerical_derivative =
        (perturbed_measurement - baseline_measurement) / epsilon;

    // Compare numerical and analytical derivatives for this state variable
    for (int j = 0; j < measurement_size; j++) {
      const double analytical_value = analytical_jacobian(j, state_idx);
      const double numerical_value = numerical_derivative(j);

      EXPECT_NEAR(numerical_value, analytical_value, 1e-4)
          << "Jacobian mismatch at measurement[" << j << "] for state[" << state_idx << "]";
    }
  }
}

} // namespace testing
} // namespace sensors
} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
