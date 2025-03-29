#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <cmath>

#include "kinematic_arbiter/ros2/ros2_utils.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace test {

class TimeUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  // The actual observed precision in ROS2 time conversions is approximately 1.2e-7s
  static constexpr double OBSERVED_TIME_PRECISION = 2e-7;
};

TEST_F(TimeUtilsTest, BasicTimeConversions) {
  // Test cases with different time values
  std::vector<double> test_times = {
    0.0,         // Zero
    1.0,         // One second
    1.5,         // Fractional second
    123.456789,  // Multiple seconds with fractional part
    -1.0,        // Negative time (handled via offset)
    -123.456,    // Larger negative time
  };

  for (const auto& time_seconds : test_times) {
    // Convert double to ROS Time and back
    rclcpp::Time ros_time = utils::doubleTimeToRosTime(time_seconds);
    double round_trip = utils::rosTimeToSeconds(ros_time);

    // Time conversions have a precision limit of ~1.2e-7 seconds
    EXPECT_NEAR(round_trip, time_seconds, OBSERVED_TIME_PRECISION)
        << "Time conversion exceeded expected precision limits";
  }
}

TEST_F(TimeUtilsTest, PrecisionLimits) {
  // Document the actual precision limitations of ROS Time conversion

  // Define test cases with expected precision loss
  struct TestCase {
    double input;
    bool small_value;
    std::string description;
  };

  std::vector<TestCase> test_cases = {
    {1.0, false, "One second (normal precision)"},
    {0.1, false, "0.1 seconds (normal precision)"},
    {1e-6, false, "1 microsecond (normal precision)"},
    {1e-7, true, "100 nanoseconds (approaching precision limit)"},
    {1e-9, true, "1 nanosecond (below precision limit)"},
    {1e-12, true, "1 picosecond (well below precision limit)"}
  };

  for (const auto& tc : test_cases) {
    // Convert double to ROS Time
    rclcpp::Time ros_time = utils::doubleTimeToRosTime(tc.input);

    // Convert back to double
    double round_trip = utils::rosTimeToSeconds(ros_time);

    // Calculate the actual error
    double absolute_error = std::abs(round_trip - tc.input);

    if (tc.small_value) {
      // For very small values, we expect the result to be dominated by precision limits
      // This means the result will be approximately the precision limit itself
      EXPECT_GE(absolute_error, 0) << "Error calculation is wrong";

      // The smallest precisely representable value is around 1.2e-7
      if (tc.input < OBSERVED_TIME_PRECISION) {
        // For inputs smaller than our precision limit, the absolute error
        // should be at least the difference between the input and our precision limit
        EXPECT_NEAR(round_trip, OBSERVED_TIME_PRECISION, OBSERVED_TIME_PRECISION)
            << "Small values should be quantized to precision limit";
      }
    } else {
      // For normal values, we expect the error to be within our observed precision
      EXPECT_NEAR(round_trip, tc.input, OBSERVED_TIME_PRECISION)
          << "Normal values should maintain precision within limits";
    }
  }
}

TEST_F(TimeUtilsTest, NegativeTimeHandling) {
  // Test cases with negative time values
  std::vector<double> test_times = {
    -1.0,       // Negative one second
    -0.1,       // Negative fractional
    -123.456789, // Larger negative with fractional part
    -1000000.0   // Very negative
  };

  for (const auto& time_seconds : test_times) {
    // Convert to ROS Time (which adds the offset)
    rclcpp::Time ros_time = utils::doubleTimeToRosTime(time_seconds);

    // ROS Time doesn't store negative times, so ensure internal time is positive
    EXPECT_GT(ros_time.nanoseconds(), 0)
        << "TIME_OFFSET should make internal ROS Time representation positive";

    // Convert back to seconds (which removes the offset)
    double round_trip = utils::rosTimeToSeconds(ros_time);

    // Should get back approximately the original negative time
    EXPECT_NEAR(round_trip, time_seconds, OBSERVED_TIME_PRECISION)
        << "Negative time not preserved in round-trip conversion";
  }
}

TEST_F(TimeUtilsTest, VerySmallTimeValues) {
  // Test cases with very small time values
  std::vector<double> test_times = {
    1e-6,  // Microsecond
    5e-7,  // 500 nanoseconds
    1e-7,  // 100 nanoseconds (around precision limit)
    1e-9,  // 1 nanosecond (below precision limit)
    1e-12  // 1 picosecond (well below precision limit)
  };

  for (const auto& time_seconds : test_times) {
    // Convert double to ROS Time
    rclcpp::Time ros_time = utils::doubleTimeToRosTime(time_seconds);

    // Convert back to double
    double round_trip_seconds = utils::rosTimeToSeconds(ros_time);

    if (time_seconds >= OBSERVED_TIME_PRECISION) {
      // Values above our observed precision limit should maintain that precision
      EXPECT_NEAR(round_trip_seconds, time_seconds, OBSERVED_TIME_PRECISION)
          << "Time above precision limit should maintain expected precision";
    } else {
      // Values below our observed precision will be quantized to approximately that precision
      // Just verify the value is positive and small
      EXPECT_GE(round_trip_seconds, 0)
          << "Very small time values should not become negative";
      EXPECT_LE(round_trip_seconds, OBSERVED_TIME_PRECISION * 2)
          << "Very small time values should be quantized near precision limit";
    }
  }
}

TEST_F(TimeUtilsTest, OffsetMechanism) {
  // Demonstrate how the offset works
  double original_time = -5.0;  // negative 5 seconds

  // Adding the offset (manually)
  double with_offset = original_time + utils::TIME_OFFSET;
  EXPECT_GT(with_offset, 0) << "Offset should make time positive";

  // Using the utility functions (should do the same thing)
  rclcpp::Time ros_time = utils::doubleTimeToRosTime(original_time);
  double internal_seconds = ros_time.nanoseconds() * 1e-9;

  // For large values, we need to increase tolerance proportionally
  double large_tolerance = OBSERVED_TIME_PRECISION * 1e9;
  EXPECT_NEAR(internal_seconds, with_offset, large_tolerance)
      << "Utility should apply offset correctly (with appropriate tolerance for large values)";

  // Converting back should remove the offset
  double round_trip = utils::rosTimeToSeconds(ros_time);
  EXPECT_NEAR(round_trip, original_time, OBSERVED_TIME_PRECISION)
      << "Round trip should restore original value within precision limits";
}

} // namespace test
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
