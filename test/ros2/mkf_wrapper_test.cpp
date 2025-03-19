#include <gtest/gtest.h>
#include <memory>
#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_eigen/tf2_eigen.h"

namespace kinematic_arbiter {
namespace ros2 {
namespace wrapper {
namespace test {

using FilterWrapper = kinematic_arbiter::ros2::wrapper::FilterWrapper;

class FilterWrapperTest : public ::testing::Test {
protected:
  void SetUp() override {
    wrapper_ = std::make_unique<FilterWrapper>();
    position_sensor_id_ = wrapper_->registerPositionSensor("test_position");
  }

  std::unique_ptr<FilterWrapper> wrapper_;
  std::string position_sensor_id_;
};

// Test basic filter wrapper functionality with a single measurement
TEST_F(FilterWrapperTest, ProcessSingleMeasurement) {
  // Create a position measurement
  geometry_msgs::msg::PointStamped position_msg;
  position_msg.header.stamp = rclcpp::Time(1.0e9); // 1.0 seconds
  position_msg.header.frame_id = "map";
  position_msg.point.x = 1.0;
  position_msg.point.y = 2.0;
  position_msg.point.z = 3.0;

  // Process the measurement
  bool result = wrapper_->processPosition(position_sensor_id_, position_msg);
  EXPECT_TRUE(result);

  // Get expected measurement
  auto expected = wrapper_->getExpectedPosition(position_sensor_id_);

  // Verify header was preserved - convert to seconds for comparison
  EXPECT_DOUBLE_EQ(rclcpp::Time(expected.header.stamp).seconds(),
                   rclcpp::Time(position_msg.header.stamp).seconds());
  EXPECT_EQ(expected.header.frame_id, position_msg.header.frame_id);

  // Verify position is close to the measurement (should be for a new filter)
  EXPECT_NEAR(expected.pose.pose.position.x, position_msg.point.x, 0.1);
  EXPECT_NEAR(expected.pose.pose.position.y, position_msg.point.y, 0.1);
  EXPECT_NEAR(expected.pose.pose.position.z, position_msg.point.z, 0.1);

  // Verify covariance is populated
  for (int i = 0; i < 3; i++) {
    EXPECT_GT(expected.pose.covariance[i*7], 0.0);  // Check diagonal elements
  }

  // Get state estimate at the measurement time
  auto state_estimate = wrapper_->getPoseEstimate(position_msg.header.stamp);

  // Verify the state estimate reflects the measurement
  EXPECT_NEAR(state_estimate.pose.pose.position.x, position_msg.point.x, 0.1);
  EXPECT_NEAR(state_estimate.pose.pose.position.y, position_msg.point.y, 0.1);
  EXPECT_NEAR(state_estimate.pose.pose.position.z, position_msg.point.z, 0.1);

  // Test prediction to a future time
  rclcpp::Time future_time(2.0e9);  // 2.0 seconds
  wrapper_->predictTo(future_time);

  // Get state estimate at future time
  auto future_estimate = wrapper_->getPoseEstimate(future_time);

  // Verify future estimate has the right timestamp - use seconds to compare
  EXPECT_DOUBLE_EQ(rclcpp::Time(future_estimate.header.stamp).seconds(),
                  future_time.seconds());

  // Verify basic velocity and acceleration estimates
  auto velocity = wrapper_->getVelocityEstimate(future_time);
  auto acceleration = wrapper_->getAccelerationEstimate(future_time);

  // Simply check that these methods return properly formed messages
  EXPECT_DOUBLE_EQ(rclcpp::Time(velocity.header.stamp).seconds(), future_time.seconds());
  EXPECT_DOUBLE_EQ(rclcpp::Time(acceleration.header.stamp).seconds(), future_time.seconds());
}

// Test error handling
TEST_F(FilterWrapperTest, ErrorHandling) {
  // Test with invalid sensor ID
  geometry_msgs::msg::PointStamped position_msg;
  position_msg.header.stamp = rclcpp::Time(1.0e9);
  position_msg.point.x = 1.0;

  // Process should fail with invalid sensor ID
  bool result = wrapper_->processPosition("invalid_id", position_msg);
  EXPECT_FALSE(result);

  // Expected measurement should return empty message for invalid sensor
  auto expected = wrapper_->getExpectedPosition("invalid_id");

  // Check if the time is zero by checking both sec and nanosec fields
  EXPECT_EQ(expected.header.stamp.sec, 0);
  EXPECT_EQ(expected.header.stamp.nanosec, 0u);
}

// Test conversion utilities
TEST_F(FilterWrapperTest, ConversionUtilities) {
  // Test Vector3d <-> Point conversions
  Eigen::Vector3d eigen_vector(1.0, 2.0, 3.0);
  auto point_msg = wrapper_->vectorToPointMsg(eigen_vector);

  EXPECT_DOUBLE_EQ(point_msg.x, 1.0);
  EXPECT_DOUBLE_EQ(point_msg.y, 2.0);
  EXPECT_DOUBLE_EQ(point_msg.z, 3.0);

  Eigen::Vector3d converted_back = wrapper_->pointMsgToVector(point_msg);

  EXPECT_DOUBLE_EQ(converted_back.x(), 1.0);
  EXPECT_DOUBLE_EQ(converted_back.y(), 2.0);
  EXPECT_DOUBLE_EQ(converted_back.z(), 3.0);

  // Test Quaterniond <-> Quaternion conversions
  Eigen::Quaterniond eigen_quat(0.7071, 0.0, 0.7071, 0.0); // 90 degree rotation around Y
  eigen_quat.normalize();

  auto quat_msg = wrapper_->quaternionToQuaternionMsg(eigen_quat);

  EXPECT_NEAR(quat_msg.w, 0.7071, 1e-4);
  EXPECT_NEAR(quat_msg.x, 0.0, 1e-4);
  EXPECT_NEAR(quat_msg.y, 0.7071, 1e-4);
  EXPECT_NEAR(quat_msg.z, 0.0, 1e-4);

  Eigen::Quaterniond converted_quat = wrapper_->quaternionMsgToEigen(quat_msg);

  EXPECT_NEAR(converted_quat.w(), 0.7071, 1e-4);
  EXPECT_NEAR(converted_quat.x(), 0.0, 1e-4);
  EXPECT_NEAR(converted_quat.y(), 0.7071, 1e-4);
  EXPECT_NEAR(converted_quat.z(), 0.0, 1e-4);

  // Test Vector3d <-> Vector3 conversions
  Eigen::Vector3d velocity_vector(2.5, -1.0, 0.0);
  auto vector_msg = wrapper_->eigenToVectorMsg(velocity_vector);

  EXPECT_DOUBLE_EQ(vector_msg.x, 2.5);
  EXPECT_DOUBLE_EQ(vector_msg.y, -1.0);
  EXPECT_DOUBLE_EQ(vector_msg.z, 0.0);

  Eigen::Vector3d converted_vector = wrapper_->vectorMsgToEigen(vector_msg);

  EXPECT_DOUBLE_EQ(converted_vector.x(), 2.5);
  EXPECT_DOUBLE_EQ(converted_vector.y(), -1.0);
  EXPECT_DOUBLE_EQ(converted_vector.z(), 0.0);
}

// Test compatibility with tf2_eigen's direct conversions
TEST_F(FilterWrapperTest, Tf2EigenCompatibility) {
  // Create test data
  Eigen::Vector3d position(3.5, 2.1, -1.2);
  Eigen::Quaterniond rotation = Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));

  // Convert using wrapper
  auto point_wrapper = wrapper_->vectorToPointMsg(position);
  auto quat_wrapper = wrapper_->quaternionToQuaternionMsg(rotation);

  // Convert using tf2_eigen directly
  geometry_msgs::msg::Point point_tf2 = tf2::toMsg(position);
  geometry_msgs::msg::Quaternion quat_tf2 = tf2::toMsg(rotation);

  // Compare results - they should be identical
  EXPECT_DOUBLE_EQ(point_wrapper.x, point_tf2.x);
  EXPECT_DOUBLE_EQ(point_wrapper.y, point_tf2.y);
  EXPECT_DOUBLE_EQ(point_wrapper.z, point_tf2.z);

  EXPECT_DOUBLE_EQ(quat_wrapper.w, quat_tf2.w);
  EXPECT_DOUBLE_EQ(quat_wrapper.x, quat_tf2.x);
  EXPECT_DOUBLE_EQ(quat_wrapper.y, quat_tf2.y);
  EXPECT_DOUBLE_EQ(quat_wrapper.z, quat_tf2.z);

  // Test round-trip conversions through both methods
  Eigen::Vector3d position_wrapper_rt = wrapper_->pointMsgToVector(point_wrapper);
  Eigen::Vector3d position_tf2_rt;
  tf2::fromMsg(point_tf2, position_tf2_rt);

  EXPECT_DOUBLE_EQ(position_wrapper_rt.x(), position_tf2_rt.x());
  EXPECT_DOUBLE_EQ(position_wrapper_rt.y(), position_tf2_rt.y());
  EXPECT_DOUBLE_EQ(position_wrapper_rt.z(), position_tf2_rt.z());
}

// Test isometry/transform conversions
TEST_F(FilterWrapperTest, TransformConversions) {
  // We don't have direct transform conversion in the wrapper
  // but let's test a composite conversion to verify compatibility

  // Create a test isometry
  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  Eigen::Quaterniond quat(Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ()));
  isometry.linear() = quat.toRotationMatrix();

  // Convert to transform using tf2_eigen
  geometry_msgs::msg::Transform transform_msg = tf2::toMsg(isometry);

  // Test the individual components with our wrapper
  Eigen::Vector3d extracted_translation = wrapper_->vectorMsgToEigen(transform_msg.translation);
  Eigen::Quaterniond extracted_rotation = wrapper_->quaternionMsgToEigen(transform_msg.rotation);

  // Verify the translation component
  EXPECT_NEAR(extracted_translation.x(), 1.0, 1e-6);
  EXPECT_NEAR(extracted_translation.y(), 2.0, 1e-6);
  EXPECT_NEAR(extracted_translation.z(), 3.0, 1e-6);

  // Verify the rotation component
  EXPECT_NEAR(extracted_rotation.w(), quat.w(), 1e-6);
  EXPECT_NEAR(extracted_rotation.x(), quat.x(), 1e-6);
  EXPECT_NEAR(extracted_rotation.y(), quat.y(), 1e-6);
  EXPECT_NEAR(extracted_rotation.z(), quat.z(), 1e-6);

  // Verify that we can recreate the isometry
  Eigen::Isometry3d rebuilt_isometry = Eigen::Isometry3d::Identity();
  rebuilt_isometry.translation() = extracted_translation;
  rebuilt_isometry.linear() = extracted_rotation.toRotationMatrix();

  // Compare the original and rebuilt transforms
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_NEAR(isometry.matrix()(i, j), rebuilt_isometry.matrix()(i, j), 1e-6);
    }
  }
}

} // namespace test
} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
