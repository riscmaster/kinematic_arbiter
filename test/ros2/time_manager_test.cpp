#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "kinematic_arbiter/ros2/ros2_utils.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace test {
  using TimeManager = ::kinematic_arbiter::ros2::utils::TimeManager;

class TimeManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    time_manager_ = std::make_unique<TimeManager>();
  }

  std::unique_ptr<TimeManager> time_manager_;

  // The actual observed precision in ROS2 time conversions is approximately 1.2e-7s
  static constexpr double OBSERVED_TIME_PRECISION = 2e-7;
};

TEST_F(TimeManagerTest, Initialization) {
  // Check initial state
  EXPECT_FALSE(time_manager_->isInitialized()) << "TimeManager should start uninitialized";

  // Initialize with current time
  rclcpp::Time now = rclcpp::Clock().now();
  time_manager_->setReferenceTime(now);

  // Check initialized state
  EXPECT_TRUE(time_manager_->isInitialized()) << "TimeManager should be initialized after setting reference";

  // Verify reference time
  rclcpp::Time ref_time = time_manager_->getReferenceTime();
  EXPECT_EQ(ref_time.nanoseconds(), now.nanoseconds()) << "Reference time should match initialization time";
}

TEST_F(TimeManagerTest, InitializationIdempotent) {
  // Initialize with first time
  rclcpp::Time first_time = rclcpp::Time(1000000000);  // 1 second
  time_manager_->setReferenceTime(first_time);
  EXPECT_TRUE(time_manager_->isInitialized());

  // Try to initialize with second time (should be ignored)
  rclcpp::Time second_time = rclcpp::Time(2000000000);  // 2 seconds
  time_manager_->setReferenceTime(second_time);

  // Reference time should still be the first time
  rclcpp::Time ref_time = time_manager_->getReferenceTime();
  EXPECT_EQ(ref_time.nanoseconds(), first_time.nanoseconds())
      << "Reference time should not change after first initialization";
}

TEST_F(TimeManagerTest, TimeConversion) {
  // Initialize time manager
  rclcpp::Time ref_time = rclcpp::Time(1000000000);  // 1 second
  time_manager_->setReferenceTime(ref_time);

  // Test cases for conversion
  std::vector<int64_t> test_times_ns = {
    1000000000,   // Same as reference (should convert to 0.0)
    2000000000,   // 1 second after reference
    1500000000,   // 0.5 seconds after reference
    500000000,    // 0.5 seconds before reference
    0,            // 1 second before reference
    3000000000,   // 2 seconds after reference
  };

  for (const auto& time_ns : test_times_ns) {
    rclcpp::Time ros_time(time_ns);

    // Convert to filter time
    double filter_time = time_manager_->rosTimeToFilterTime(ros_time);

    // Expected value (time difference in seconds)
    double expected = (time_ns - ref_time.nanoseconds()) * 1e-9;

    // Check conversion
    EXPECT_NEAR(filter_time, expected, OBSERVED_TIME_PRECISION)
        << "ROS to filter time conversion incorrect";

    // Round trip: convert back to ROS time
    rclcpp::Time round_trip = time_manager_->filterTimeToRosTime(filter_time);

    // Should get back the original ROS time
    EXPECT_NEAR(round_trip.nanoseconds(), time_ns, 1)  // Allow 1ns error for integer rounding
        << "Filter to ROS time round-trip conversion incorrect";
  }
}

TEST_F(TimeManagerTest, ErrorHandling) {
  // Without initialization, conversions should throw
  rclcpp::Time test_time(1000000000);
  EXPECT_THROW(time_manager_->rosTimeToFilterTime(test_time), std::runtime_error);
  EXPECT_THROW(time_manager_->filterTimeToRosTime(1.0), std::runtime_error);
}

TEST_F(TimeManagerTest, LargeTimeValues) {
  // Initialize with reference time
  rclcpp::Time ref_time = rclcpp::Time(1000000000);  // 1 second
  time_manager_->setReferenceTime(ref_time);

  // Test with a large time value (100 days in nanoseconds)
  int64_t large_time_ns = 8640000000000000;  // 100 days
  rclcpp::Time large_ros_time(large_time_ns);

  // Convert to filter time
  double filter_time = time_manager_->rosTimeToFilterTime(large_ros_time);

  // Expected value in seconds (time difference)
  double expected = (large_time_ns - ref_time.nanoseconds()) * 1e-9;

  // Check conversion (large value - need larger tolerance)
  double large_tolerance = OBSERVED_TIME_PRECISION * 1e5;
  EXPECT_NEAR(filter_time, expected, large_tolerance)
      << "Large time conversion should work within appropriate tolerances";

  // Round trip
  rclcpp::Time round_trip = time_manager_->filterTimeToRosTime(filter_time);

  // Round trip should be close to original
  EXPECT_NEAR(double(round_trip.nanoseconds()), double(large_time_ns), large_time_ns * 1e-7)
      << "Round trip for large values should preserve value within tolerance";
}

TEST_F(TimeManagerTest, RelativeTimeValues) {
  // Test how well the relative times work for typical use cases

  // Initialize with a reference time (simulating filter start)
  rclcpp::Time start_time = rclcpp::Time(1000000000);  // 1 second
  time_manager_->setReferenceTime(start_time);

  // Simulate measurements coming in with small relative times
  std::vector<double> relative_times = {
    0.01,    // 10ms after start
    0.1,     // 100ms after start
    1.0,     // 1s after start
    10.0,    // 10s after start
    -0.01,   // 10ms before start
    -0.1,    // 100ms before start
  };

  for (const auto& rel_time : relative_times) {
    // Calculate expected ROS time
    int64_t expected_ns = start_time.nanoseconds() + static_cast<int64_t>(rel_time * 1e9);

    // Convert from filter time to ROS time
    rclcpp::Time ros_time = time_manager_->filterTimeToRosTime(rel_time);

    // Check conversion
    EXPECT_NEAR(ros_time.nanoseconds(), expected_ns, 1)  // Allow 1ns rounding error
        << "Filter to ROS time conversion incorrect for relative time: " << rel_time;

    // Round trip
    double round_trip = time_manager_->rosTimeToFilterTime(ros_time);

    // Should get back the original relative time
    EXPECT_NEAR(round_trip, rel_time, OBSERVED_TIME_PRECISION)
        << "ROS to filter time round-trip conversion incorrect";
  }
}

} // namespace test
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
