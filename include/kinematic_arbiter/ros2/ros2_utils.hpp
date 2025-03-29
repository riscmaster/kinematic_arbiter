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
 * @brief Class to handle time conversions between ROS and filter
 *
 * Uses the initialization time as the offset, so filter times are relative
 * to when the filter was started, improving numerical stability
 */
class TimeManager {
public:
  /**
   * @brief Initialize with the current time as reference
   */
  TimeManager() : initialized_(false), reference_time_(0) {}

  /**
   * @brief Initialize with a specific reference time
   *
   * @param reference_time The ROS time to use as reference (t=0 for filter)
   */
  void setReferenceTime(const rclcpp::Time& reference_time) {
    if (!initialized_) {
      reference_time_ = reference_time.nanoseconds();
      initialized_ = true;
    }
  }

  /**
   * @brief Convert ROS Time to filter internal time (seconds relative to reference)
   *
   * @param ros_time ROS time
   * @return Time in seconds relative to reference
   */
  double rosTimeToFilterTime(const rclcpp::Time& ros_time) {
    if (!initialized_) {
      throw std::runtime_error("TimeManager not initialized with reference time");
    }

    // Calculate time difference in seconds
    double filter_time = (ros_time.nanoseconds() - reference_time_) * 1e-9;
    return filter_time;
  }

  /**
   * @brief Convert filter time to ROS time
   *
   * @param filter_time Time in seconds relative to reference
   * @return ROS time
   */
  rclcpp::Time filterTimeToRosTime(double filter_time) {
    if (!initialized_) {
      throw std::runtime_error("TimeManager not initialized with reference time");
    }

    // Calculate absolute time in nanoseconds
    int64_t nanoseconds = reference_time_ + static_cast<int64_t>(filter_time * 1e9);
    return rclcpp::Time(nanoseconds);
  }

  /**
   * @brief Check if the time manager has been initialized
   *
   * @return true if initialized
   */
  bool isInitialized() const {
    return initialized_;
  }

  /**
   * @brief Get the reference time
   *
   * @return The reference time as a ROS time
   */
  rclcpp::Time getReferenceTime() const {
    return rclcpp::Time(reference_time_);
  }

private:
  bool initialized_;
  int64_t reference_time_;  // in nanoseconds
};

} // namespace utils
} // namespace ros2
} // namespace kinematic_arbiter
