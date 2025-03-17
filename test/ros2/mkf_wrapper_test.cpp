#include "gtest/gtest.h"
#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace wrapper {
namespace test {

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

} // namespace test
} // namespace wrapper
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
