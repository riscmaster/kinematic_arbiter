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

    // Record values for test report (only visible on failure)
    RecordProperty("original_time", time_seconds);
    RecordProperty("ros_time_nanosec", ros_time.nanoseconds());
    RecordProperty("round_trip_time", round_trip_seconds);
    RecordProperty("difference", std::abs(time_seconds - round_trip_seconds));

    // Verify round-trip conversion is accurate
    EXPECT_NEAR(time_seconds, round_trip_seconds, 1e-6);  // Allow some floating point error
  }
}

TEST_F(TimeConversionTest, SystemTimeToRosTimeAndBack) {
  // Test conversion with system time
  rclcpp::Time now = rclcpp::Clock().now();
  double now_seconds = wrapper_->rosTimeToSeconds(now);
  rclcpp::Time now_reconstructed = wrapper_->doubleTimeToRosTime(now_seconds);

  // Record properties for test report (only visible on failure)
  RecordProperty("system_now_nanosec", now.nanoseconds());
  RecordProperty("now_as_double", now_seconds);
  RecordProperty("reconstructed_nanosec", now_reconstructed.nanoseconds());
  RecordProperty("nanosec_difference", std::abs(now.nanoseconds() - now_reconstructed.nanoseconds()));

  // For system time values, we need a larger tolerance due to floating-point precision
  EXPECT_NEAR(now.seconds(), now_reconstructed.seconds(), 1e-6);
}

TEST_F(TimeConversionTest, IncrementalTimeCorrectness) {
  // Start with zero
  double time_seconds = 0.0;

  // Increment by small amounts
  for (int i = 1; i <= 100; i++) {
    // Add 0.01 seconds each iteration
    time_seconds += 0.01;

    // Convert to ROS time
    rclcpp::Time ros_time = wrapper_->doubleTimeToRosTime(time_seconds);

    // Convert back to double
    double round_trip = wrapper_->rosTimeToSeconds(ros_time);

    // Only add trace for every 10th iteration to avoid flooding output
    if (i % 10 == 0) {
      // Add context information for the test trace if there's a failure
      SCOPED_TRACE("Iteration " + std::to_string(i) + ", time: " + std::to_string(time_seconds) + "s");

      // Record values for test report (only visible on failure)
      RecordProperty("iteration_" + std::to_string(i) + "_time", time_seconds);
      RecordProperty("iteration_" + std::to_string(i) + "_nanosec", ros_time.nanoseconds());
      RecordProperty("iteration_" + std::to_string(i) + "_round_trip", round_trip);
      RecordProperty("iteration_" + std::to_string(i) + "_diff", std::abs(time_seconds - round_trip));
    }

    // Verify time round-trip is correct
    EXPECT_NEAR(time_seconds, round_trip, 1e-6);
  }
}

TEST_F(TimeConversionTest, NegativeTimeConversion) {
  // Test negative times to ensure conversion works correctly
  std::vector<double> negative_test_times = {
    -0.1,
    -1.0,
    -1.5,
    -10.25,
    -100.125
  };

  for (double time_seconds : negative_test_times) {
    SCOPED_TRACE("Testing negative time: " + std::to_string(time_seconds) + "s");

    // Convert double to ROS Time
    rclcpp::Time ros_time = wrapper_->doubleTimeToRosTime(time_seconds);

    // Convert back to double
    double round_trip_seconds = wrapper_->rosTimeToSeconds(ros_time);

    // Record values for test report (only visible on failure)
    RecordProperty("negative_original_time", time_seconds);
    RecordProperty("negative_ros_time_nanosec", ros_time.nanoseconds());
    RecordProperty("negative_round_trip_time", round_trip_seconds);
    RecordProperty("negative_difference", std::abs(time_seconds - round_trip_seconds));

    // Verify round-trip conversion is accurate
    EXPECT_NEAR(time_seconds, round_trip_seconds, 1e-6);  // Allow some floating point error
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
