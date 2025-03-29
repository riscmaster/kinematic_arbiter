#pragma once

#include "rclcpp/rclcpp.hpp"

// Note: For transform conversions between Eigen and ROS, consider using:
// #include <tf2_eigen/tf2_eigen.hpp>
// See: https://github.com/ros2/geometry2/tree/ros2/tf2_eigen

namespace kinematic_arbiter {
namespace ros2 {
namespace utils {

//------------------------------------------------------------------------------
// Time conversion utilities
//------------------------------------------------------------------------------

/**
 * @brief Time offset constant to match the one used in SensorHandler
 * This ensures positive time values in simulations or with artificial timestamps
 */
static constexpr double TIME_OFFSET = 1e9;  // Large offset to ensure positive time

/**
 * @brief Convert ROS Time to seconds with proper offset handling
 *
 * Matches the implementation in SensorHandler for test compatibility
 *
 * @param time ROS time
 * @return Time in seconds
 */
inline double rosTimeToSeconds(const rclcpp::Time& time) {
  // Convert to nanoseconds, then to seconds with double precision
  double seconds_with_offset = time.nanoseconds() * 1e-9;
  return seconds_with_offset - TIME_OFFSET;
}

/**
 * @brief Convert double time to ROS time with proper offset handling
 *
 * Matches the implementation in SensorHandler for test compatibility
 *
 * @param time Time in seconds
 * @return ROS time
 */
inline rclcpp::Time doubleTimeToRosTime(double time) {
  // Add offset to ensure positive time
  double time_with_offset = time + TIME_OFFSET;

  // Convert to nanoseconds and use the nanoseconds constructor
  int64_t nanoseconds = static_cast<int64_t>(time_with_offset * 1e9);

  // Create time from nanoseconds
  return rclcpp::Time(nanoseconds);
}

/**
 * @brief Simple conversion from seconds to ROS time without offset
 *
 * @param seconds Time in seconds
 * @return rclcpp::Time ROS time
 */
inline rclcpp::Time secondsToRosTime(double seconds) {
  return rclcpp::Time(static_cast<int64_t>(seconds * 1e9));
}

/**
 * @brief Simple conversion from ROS time to seconds without offset
 *
 * @param time ROS time
 * @return double Time in seconds
 */
inline double rosTimeToSecondsSimple(const rclcpp::Time& time) {
  return time.seconds();
}

} // namespace utils
} // namespace ros2
} // namespace kinematic_arbiter
