#include <gtest/gtest.h>
#include <drake/common/drake_assert.h>
#include <drake/math/rotation_matrix.h>
#include <drake/math/roll_pitch_yaw.h>

// Simple test to verify Drake integration
TEST(DrakeIntegration, BasicFunctionality) {
  // Create a Drake rotation matrix
  drake::math::RotationMatrix<double> rotation =
      drake::math::RotationMatrix<double>::MakeXRotation(M_PI / 4);

  // Get the rotation angle around X axis using RollPitchYaw
  drake::math::RollPitchYaw<double> rpy(rotation);
  Eigen::Vector3d euler_angles = rpy.vector();

  // Check that the X rotation is approximately PI/4
  EXPECT_NEAR(euler_angles(0), M_PI / 4, 1e-10);

  // Test that the rotation matrix is orthonormal
  EXPECT_TRUE(rotation.IsValid());
}

// Test Drake quaternion functionality
TEST(DrakeIntegration, QuaternionOperations) {
  // Create a quaternion representing 90-degree rotation around Z axis
  Eigen::Quaterniond quaternion(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));

  // Convert to rotation matrix
  drake::math::RotationMatrix<double> rotation(quaternion);

  // Apply rotation to a vector pointing along X axis
  Eigen::Vector3d x_axis(1, 0, 0);
  Eigen::Vector3d rotated = rotation * x_axis;

  // After 90-degree Z rotation, x-axis should point along y-axis
  EXPECT_NEAR(rotated(0), 0, 1e-10);
  EXPECT_NEAR(rotated(1), 1, 1e-10);
  EXPECT_NEAR(rotated(2), 0, 1e-10);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
