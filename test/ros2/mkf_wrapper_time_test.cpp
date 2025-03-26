#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"

using namespace kinematic_arbiter::ros2::wrapper;

class TimeConversionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
    wrapper_ = std::make_shared<FilterWrapper>();
  }

  std::shared_ptr<FilterWrapper> wrapper_;
};

TEST_F(TimeConversionTest, TimeConversionRoundTrip) {
  // Test various times to ensure round-trip conversion works correctly
  std::vector<double> test_times = {
    0.0,
    0.1,
    1.0,
    1.5,
    10.25,
    100.125,
    1000.0625
  };

  for (double time_seconds : test_times) {
    SCOPED_TRACE("Testing time: " + std::to_string(time_seconds) + "s");

    // Convert double to ROS Time
    rclcpp::Time ros_time = wrapper_->doubleTimeToRosTime(time_seconds);

    // Convert back to double
    double round_trip_seconds = wrapper_->rosTimeToSeconds(ros_time);

    // Record property for test report
    RecordProperty("original_time", time_seconds);
    RecordProperty("ros_time_seconds", ros_time.seconds());
    RecordProperty("round_trip_time", round_trip_seconds);

    // Verify round-trip conversion is accurate
    EXPECT_DOUBLE_EQ(time_seconds, round_trip_seconds);
  }
}

TEST_F(TimeConversionTest, SystemTimeToRosTimeAndBack) {
  // Test conversion with system time
  rclcpp::Time now = rclcpp::Clock().now();
  double now_seconds = wrapper_->rosTimeToSeconds(now);
  rclcpp::Time now_reconstructed = wrapper_->doubleTimeToRosTime(now_seconds);

  // Record properties for test report
  RecordProperty("system_now_seconds", now.seconds());
  RecordProperty("now_as_double", now_seconds);
  RecordProperty("reconstructed_seconds", now_reconstructed.seconds());

  // They should be equal or extremely close
  EXPECT_NEAR(now.seconds(), now_reconstructed.seconds(), 1e-9);
}

TEST_F(TimeConversionTest, IncrementalTimeCorrectness) {
  // Start with zero
  double time_seconds = 0.0;
  rclcpp::Time ros_time = wrapper_->doubleTimeToRosTime(time_seconds);

  // Verify initial time
  EXPECT_DOUBLE_EQ(0.0, ros_time.seconds());

  // Increment by small amounts
  for (int i = 1; i <= 100; i++) {
    // Add 0.01 seconds each iteration
    time_seconds += 0.01;
    ros_time = wrapper_->doubleTimeToRosTime(time_seconds);

    // Only add trace for every 10th iteration to avoid flooding output
    if (i % 10 == 0) {
      SCOPED_TRACE("Iteration " + std::to_string(i) + ", time: " + std::to_string(time_seconds) + "s");
      RecordProperty("iteration_" + std::to_string(i) + "_expected", time_seconds);
      RecordProperty("iteration_" + std::to_string(i) + "_actual", ros_time.seconds());
    }

    // Verify time is correct
    EXPECT_NEAR(time_seconds, ros_time.seconds(), 1e-9);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
