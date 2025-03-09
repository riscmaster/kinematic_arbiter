#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <memory>
#include <iostream>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"

namespace kinematic_arbiter {
namespace sensors {
namespace test {

using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using core::StateIndex;

class ImuSensorModelJacobianTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create sensor model with identity transform
    ImuSensorConfig config;
    config.calibration_enabled = false;
    imu_model_ = std::make_unique<ImuSensorModel>(Eigen::Isometry3d::Identity(), config);

    // Initialize state vector
    state_ = Eigen::VectorXd::Zero(StateIndex::kFullStateSize);

    // Set default gravity
    gravity_ = Vector3d(0, 0, imu_model_->GetGravity());
  }

  // Set quaternion in state vector
  void SetQuaternion(const Quaterniond& q) {
    state_(StateIndex::Quaternion::W) = q.w();
    state_(StateIndex::Quaternion::X) = q.x();
    state_(StateIndex::Quaternion::Y) = q.y();
    state_(StateIndex::Quaternion::Z) = q.z();
  }

  // Set angular velocity in state vector
  void SetAngularVelocity(const Vector3d& omega) {
    state_.segment<3>(StateIndex::AngularVelocity::Begin()) = omega;
  }

  // Set angular acceleration in state vector
  void SetAngularAcceleration(const Vector3d& alpha) {
    state_.segment<3>(StateIndex::AngularAcceleration::Begin()) = alpha;
  }

  // Set linear acceleration in state vector
  void SetLinearAcceleration(const Vector3d& accel) {
    state_.segment<3>(StateIndex::LinearAcceleration::Begin()) = accel;
  }

  // Set sensor-to-body transform
  void SetSensorToBodyTransform(const Eigen::Isometry3d& T_BS) {
    // Recreate the model with the new transform
    ImuSensorConfig config;
    config.calibration_enabled = false;
    imu_model_ = std::make_unique<ImuSensorModel>(T_BS, config);
  }

  // Test if the Jacobian correctly predicts measurement changes for each state component
  void TestJacobianLinearization(const std::string& description) {
    // Get the baseline measurement and Jacobian
    Eigen::VectorXd y0 = imu_model_->PredictMeasurement(state_);
    Eigen::MatrixXd C = imu_model_->GetMeasurementJacobian(state_);

    // Test quaternion perturbations
    bool quat_success = TestQuaternionPerturbations(y0, C, description);

    // Test angular velocity perturbations
    bool ang_vel_success = TestAngularVelocityPerturbations(y0, C, description);

    // Test angular acceleration perturbations
    bool ang_acc_success = TestAngularAccelerationPerturbations(y0, C, description);

    // Test linear acceleration perturbations
    bool lin_acc_success = TestLinearAccelerationPerturbations(y0, C, description);

    // Only print detailed diagnostics if any test failed
    if (!quat_success || !ang_vel_success || !ang_acc_success || !lin_acc_success) {
      std::cout << "\n==== Jacobian linearization failed for " << description << " ====\n";

      // Print the current state
      std::cout << "Current state:\n";
      std::cout << "  Quaternion: w=" << state_(StateIndex::Quaternion::W)
                << ", x=" << state_(StateIndex::Quaternion::X)
                << ", y=" << state_(StateIndex::Quaternion::Y)
                << ", z=" << state_(StateIndex::Quaternion::Z) << "\n";

      std::cout << "  Angular velocity: "
                << state_.segment<3>(StateIndex::AngularVelocity::Begin()).transpose() << "\n";

      std::cout << "  Angular acceleration: "
                << state_.segment<3>(StateIndex::AngularAcceleration::Begin()).transpose() << "\n";

      std::cout << "  Linear acceleration: "
                << state_.segment<3>(StateIndex::LinearAcceleration::Begin()).transpose() << "\n";

      // Print the Jacobian
      std::cout << "Jacobian:\n" << C << "\n";

      // Print the baseline measurement
      std::cout << "Baseline measurement: " << y0.transpose() << "\n";
    }
  }

  // Test quaternion perturbations using proper rotation perturbations
  bool TestQuaternionPerturbations(const Eigen::VectorXd& y0, const Eigen::MatrixXd& C,
                                  const std::string& description) {
    bool all_passed = true;

    // Get current quaternion
    Quaterniond q_current(
        state_(StateIndex::Quaternion::W),
        state_(StateIndex::Quaternion::X),
        state_(StateIndex::Quaternion::Y),
        state_(StateIndex::Quaternion::Z));

    // Test small rotations around each axis
    Vector3d axes[3] = {
        Vector3d(1, 0, 0),  // X-axis
        Vector3d(0, 1, 0),  // Y-axis
        Vector3d(0, 0, 1)   // Z-axis
    };

    // Use a larger angle for better numerical stability
    const double angle = 0.01;  // 0.01 rad ≈ 0.57°

    for (int i = 0; i < 3; i++) {
      // Create a small rotation perturbation
      Quaterniond q_delta(Eigen::AngleAxisd(angle, axes[i]));

      // Apply the perturbation (right multiplication for local frame rotation)
      Quaterniond q_perturbed = q_current * q_delta;
      q_perturbed.normalize();  // Ensure unit quaternion

      // Create perturbed state
      Eigen::VectorXd perturbed_state = state_;
      perturbed_state(StateIndex::Quaternion::W) = q_perturbed.w();
      perturbed_state(StateIndex::Quaternion::X) = q_perturbed.x();
      perturbed_state(StateIndex::Quaternion::Y) = q_perturbed.y();
      perturbed_state(StateIndex::Quaternion::Z) = q_perturbed.z();

      // Get the actual measurement at the perturbed state
      Eigen::VectorXd y1 = imu_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      Eigen::VectorXd actual_delta_y = y1 - y0;

      // Calculate the predicted change using the Jacobian
      Eigen::VectorXd predicted_delta_y = C * (perturbed_state - state_);

      // Calculate the absolute error instead of relative error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Only print on failure
      bool passed = abs_error < 0.01;  // Use absolute error threshold
      if (!passed) {
        all_passed = false;
        std::cout << "  Quaternion rotation around "
                  << (i == 0 ? "X" : i == 1 ? "Y" : "Z") << "-axis: "
                  << abs_error << " absolute error\n";

        std::cout << "    Actual change: " << actual_delta_y.transpose() << "\n";
        std::cout << "    Predicted change: " << predicted_delta_y.transpose() << "\n";
        std::cout << "    State delta: " << (perturbed_state - state_).transpose() << "\n";
        std::cout << "    Perturbed measurement: " << y1.transpose() << "\n";
      }

      EXPECT_LT(abs_error, 0.01) << "Quaternion Jacobian linearization error too large for "
                                << (i == 0 ? "X" : i == 1 ? "Y" : "Z") << "-axis rotation in "
                                << description;
    }

    return all_passed;
  }

  // Test angular velocity perturbations
  bool TestAngularVelocityPerturbations(const Eigen::VectorXd& y0, const Eigen::MatrixXd& C,
                                       const std::string& description) {
    bool all_passed = true;

    // Test perturbation in each component
    for (int i = 0; i < 3; i++) {
      // Create a small perturbation
      Eigen::VectorXd perturbed_state = state_;
      double delta = 0.1;  // 0.1 rad/s
      perturbed_state(StateIndex::AngularVelocity::Begin() + i) += delta;

      // Get the actual measurement at the perturbed state
      Eigen::VectorXd y1 = imu_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      Eigen::VectorXd actual_delta_y = y1 - y0;

      // Calculate the predicted change using the Jacobian
      Eigen::VectorXd predicted_delta_y = C * (perturbed_state - state_);

      // Calculate the absolute error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Only print on failure
      bool passed = abs_error < 0.01;
      if (!passed) {
        all_passed = false;
        std::cout << "  Angular velocity " << (i == 0 ? "ω_x" : i == 1 ? "ω_y" : "ω_z")
                  << ": " << abs_error << " absolute error\n";

        std::cout << "    Actual change: " << actual_delta_y.transpose() << "\n";
        std::cout << "    Predicted change: " << predicted_delta_y.transpose() << "\n";
        std::cout << "    State delta: " << (perturbed_state - state_).transpose() << "\n";
      }

      EXPECT_LT(abs_error, 0.01) << "Angular velocity Jacobian linearization error too large for "
                                << (i == 0 ? "ω_x" : i == 1 ? "ω_y" : "ω_z") << " in "
                                << description;
    }

    return all_passed;
  }

  // Test angular acceleration perturbations
  bool TestAngularAccelerationPerturbations(const Eigen::VectorXd& y0, const Eigen::MatrixXd& C,
                                           const std::string& description) {
    bool all_passed = true;

    // Test perturbation in each component
    for (int i = 0; i < 3; i++) {
      // Create a small perturbation
      Eigen::VectorXd perturbed_state = state_;
      double delta = 0.1;  // 0.1 rad/s²
      perturbed_state(StateIndex::AngularAcceleration::Begin() + i) += delta;

      // Get the actual measurement at the perturbed state
      Eigen::VectorXd y1 = imu_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      Eigen::VectorXd actual_delta_y = y1 - y0;

      // Calculate the predicted change using the Jacobian
      Eigen::VectorXd predicted_delta_y = C * (perturbed_state - state_);

      // Calculate the absolute error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Only print on failure
      bool passed = abs_error < 0.01;
      if (!passed) {
        all_passed = false;
        std::cout << "  Angular acceleration " << (i == 0 ? "α_x" : i == 1 ? "α_y" : "α_z")
                  << ": " << abs_error << " absolute error\n";

        std::cout << "    Actual change: " << actual_delta_y.transpose() << "\n";
        std::cout << "    Predicted change: " << predicted_delta_y.transpose() << "\n";
        std::cout << "    State delta: " << (perturbed_state - state_).transpose() << "\n";
      }

      EXPECT_LT(abs_error, 0.01) << "Angular acceleration Jacobian linearization error too large for "
                                << (i == 0 ? "α_x" : i == 1 ? "α_y" : "α_z") << " in "
                                << description;
    }

    return all_passed;
  }

  // Test linear acceleration perturbations
  bool TestLinearAccelerationPerturbations(const Eigen::VectorXd& y0, const Eigen::MatrixXd& C,
                                          const std::string& description) {
    bool all_passed = true;

    // Test perturbation in each component
    for (int i = 0; i < 3; i++) {
      // Create a small perturbation
      Eigen::VectorXd perturbed_state = state_;
      double delta = 0.1;  // 0.1 m/s²
      perturbed_state(StateIndex::LinearAcceleration::Begin() + i) += delta;

      // Get the actual measurement at the perturbed state
      Eigen::VectorXd y1 = imu_model_->PredictMeasurement(perturbed_state);

      // Calculate the actual change in measurement
      Eigen::VectorXd actual_delta_y = y1 - y0;

      // Calculate the predicted change using the Jacobian
      Eigen::VectorXd predicted_delta_y = C * (perturbed_state - state_);

      // Calculate the absolute error
      double abs_error = (predicted_delta_y - actual_delta_y).norm();

      // Only print on failure
      bool passed = abs_error < 0.01;
      if (!passed) {
        all_passed = false;
        std::cout << "  Linear acceleration " << (i == 0 ? "a_x" : i == 1 ? "a_y" : "a_z")
                  << ": " << abs_error << " absolute error\n";

        std::cout << "    Actual change: " << actual_delta_y.transpose() << "\n";
        std::cout << "    Predicted change: " << predicted_delta_y.transpose() << "\n";
        std::cout << "    State delta: " << (perturbed_state - state_).transpose() << "\n";
      }

      EXPECT_LT(abs_error, 0.01) << "Linear acceleration Jacobian linearization error too large for "
                                << (i == 0 ? "a_x" : i == 1 ? "a_y" : "a_z") << " in "
                                << description;
    }

    return all_passed;
  }

  std::unique_ptr<ImuSensorModel> imu_model_;
  Eigen::VectorXd state_;
  Vector3d gravity_;
};

// Test with different orientations
TEST_F(ImuSensorModelJacobianTest, IdentityQuaternion) {
  SetQuaternion(Quaterniond::Identity());
  TestJacobianLinearization("Identity Quaternion");
}

TEST_F(ImuSensorModelJacobianTest, XRotation90Deg) {
  SetQuaternion(Quaterniond(std::sqrt(2.0)/2.0, std::sqrt(2.0)/2.0, 0.0, 0.0));
  TestJacobianLinearization("90° X-Rotation");
}

TEST_F(ImuSensorModelJacobianTest, YRotation90Deg) {
  SetQuaternion(Quaterniond(std::sqrt(2.0)/2.0, 0.0, std::sqrt(2.0)/2.0, 0.0));
  TestJacobianLinearization("90° Y-Rotation");
}

TEST_F(ImuSensorModelJacobianTest, ZRotation90Deg) {
  SetQuaternion(Quaterniond(std::sqrt(2.0)/2.0, 0.0, 0.0, std::sqrt(2.0)/2.0));
  TestJacobianLinearization("90° Z-Rotation");
}

TEST_F(ImuSensorModelJacobianTest, ArbitraryOrientation) {
  SetQuaternion(Quaterniond(0.5, 0.5, 0.5, 0.5).normalized());
  TestJacobianLinearization("Arbitrary Orientation");
}

// Test with non-zero dynamics
TEST_F(ImuSensorModelJacobianTest, WithAngularVelocity) {
  SetQuaternion(Quaterniond::UnitRandom());
  SetAngularVelocity(Vector3d(0.5, -0.3, 0.2));
  TestJacobianLinearization("With Angular Velocity");
}

TEST_F(ImuSensorModelJacobianTest, WithAngularAcceleration) {
  SetQuaternion(Quaterniond::UnitRandom());
  SetAngularAcceleration(Vector3d(1.0, -0.7, 0.4));
  TestJacobianLinearization("With Angular Acceleration");
}

TEST_F(ImuSensorModelJacobianTest, WithLinearAcceleration) {
  SetQuaternion(Quaterniond::UnitRandom());
  SetLinearAcceleration(Vector3d(0.8, 0.6, -0.9));
  TestJacobianLinearization("With Linear Acceleration");
}

// Test with non-identity sensor transform
TEST_F(ImuSensorModelJacobianTest, WithSensorOffset) {
  Eigen::Isometry3d T_BS = Eigen::Isometry3d::Identity();
  T_BS.translation() = Vector3d(0.1, 0.2, 0.3);
  SetSensorToBodyTransform(T_BS);

  SetQuaternion(Quaterniond::UnitRandom());
  SetAngularVelocity(Vector3d(0.5, -0.3, 0.2));
  SetAngularAcceleration(Vector3d(1.0, -0.7, 0.4));

  TestJacobianLinearization("With Sensor Offset");
}

// Test with high angular velocity
TEST_F(ImuSensorModelJacobianTest, HighAngularVelocity) {
  Eigen::Isometry3d T_BS = Eigen::Isometry3d::Identity();
  T_BS.translation() = Vector3d(0.1, 0.2, 0.3);
  SetSensorToBodyTransform(T_BS);

  SetQuaternion(Quaterniond::UnitRandom());
  SetAngularVelocity(Vector3d(3.0, -2.0, 2.5));  // ~150°/s

  TestJacobianLinearization("High Angular Velocity");
}

// Test with large angular acceleration
TEST_F(ImuSensorModelJacobianTest, LargeAngularAcceleration) {
  Eigen::Isometry3d T_BS = Eigen::Isometry3d::Identity();
  T_BS.translation() = Vector3d(0.1, 0.2, 0.3);
  SetSensorToBodyTransform(T_BS);

  SetQuaternion(Quaterniond::UnitRandom());
  SetAngularAcceleration(Vector3d(10.0, -8.0, 6.0));  // ~500°/s²

  TestJacobianLinearization("Large Angular Acceleration");
}

}  // namespace test
}  // namespace sensors
}  // namespace kinematic_arbiter
