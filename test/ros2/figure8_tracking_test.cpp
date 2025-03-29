#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_eigen/tf2_eigen.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace test {

class Figure8TrackingTest : public ::testing::Test {
protected:
  // Test parameters
  const double test_duration_seconds_ = 5.0;
  const double convergence_check_start_ = 3.0; // Start checking for convergence after 3 seconds
  const double max_position_error_ = 0.05;     // 5cm position error tolerance
  const double max_velocity_error_ = 0.1;      // 0.1 m/s velocity error tolerance
  const double max_orientation_error_ = 0.1;   // ~5.7 degrees orientation error tolerance

  // Error tracking
  struct ErrorMetrics {
    double position_error;
    double velocity_error;
    double orientation_error;
    rclcpp::Time timestamp;
  };
  std::vector<ErrorMetrics> error_metrics_;

  // Node and executors
  std::shared_ptr<rclcpp::Node> test_node_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;

  // Subscribers for filter estimates
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_est_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr vel_est_sub_;

  // Subscribers for ground truth
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_truth_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_truth_sub_;

  // Latest messages
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr latest_pose_est_;
  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr latest_vel_est_;
  geometry_msgs::msg::PoseStamped::SharedPtr latest_pose_truth_;
  geometry_msgs::msg::TwistStamped::SharedPtr latest_vel_truth_;

  // Test synchronization
  std::atomic<bool> test_complete_{false};

  void SetUp() override {
    // Initialize ROS
    rclcpp::init(0, nullptr);

    // Create our test node
    test_node_ = std::make_shared<rclcpp::Node>("figure8_tracking_test");
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(test_node_);

    RCLCPP_INFO(test_node_->get_logger(), "Starting Figure-8 tracking test");

    // List available topics to verify what's available
    RCLCPP_INFO(test_node_->get_logger(), "Checking for available topics...");
    auto topics = test_node_->get_topic_names_and_types();
    for (const auto& topic : topics) {
      RCLCPP_INFO(test_node_->get_logger(), "Found topic: %s", topic.first.c_str());
    }

    // Subscribe to filter estimates with explicit QoS settings
    rclcpp::QoS qos(10);
    qos.reliable();

    RCLCPP_INFO(test_node_->get_logger(), "Subscribing to /state/pose");
    pose_est_sub_ = test_node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/state/pose", qos,
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        latest_pose_est_ = msg;
        RCLCPP_INFO_THROTTLE(test_node_->get_logger(),
                           *test_node_->get_clock(),
                           1000 /* ms */,
                           "Received pose estimate");
      });

    vel_est_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "/state/velocity", 10,
      [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
        latest_vel_est_ = msg;
        RCLCPP_INFO_THROTTLE(test_node_->get_logger(),
                           *test_node_->get_clock(),
                           1000 /* ms */,
                           "Received velocity estimate");
      });

    // Subscribe to ground truth
    pose_truth_sub_ = test_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/truth/pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        latest_pose_truth_ = msg;
        RCLCPP_INFO_THROTTLE(test_node_->get_logger(),
                           *test_node_->get_clock(),
                           1000 /* ms */,
                           "Received truth pose");
      });

    vel_truth_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/truth/velocity", 10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        latest_vel_truth_ = msg;
        RCLCPP_INFO_THROTTLE(test_node_->get_logger(),
                           *test_node_->get_clock(),
                           1000 /* ms */,
                           "Received truth velocity");
      });

    // Spin executor briefly to process any pending messages
    executor_->spin_some(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    RCLCPP_INFO(test_node_->get_logger(), "Test teardown");
    executor_.reset();
    test_node_.reset();
    rclcpp::shutdown();
  }

  void processMeasurements() {
    RCLCPP_INFO_THROTTLE(test_node_->get_logger(),
                       *test_node_->get_clock(),
                       1000 /* ms */,
                       "Processing measurements: pose_est=%d, vel_est=%d, pose_truth=%d, vel_truth=%d",
                       latest_pose_est_ ? 1 : 0,
                       latest_vel_est_ ? 1 : 0,
                       latest_pose_truth_ ? 1 : 0,
                       latest_vel_truth_ ? 1 : 0);

    if (!latest_pose_est_ || !latest_vel_est_ || !latest_pose_truth_ || !latest_vel_truth_) {
      return;
    }

    // Ensure timestamps are close
    double time_diff = (rclcpp::Time(latest_pose_est_->header.stamp) -
                        rclcpp::Time(latest_vel_est_->header.stamp)).seconds();
    if (std::abs(time_diff) > 0.01) { // 10ms tolerance
      return;
    }

    // Find closest ground truth messages by timestamp
    double pose_time_diff = (rclcpp::Time(latest_pose_est_->header.stamp) -
                            rclcpp::Time(latest_pose_truth_->header.stamp)).seconds();
    if (std::abs(pose_time_diff) > 0.02) { // 20ms tolerance
      return;
    }

    double vel_time_diff = (rclcpp::Time(latest_vel_est_->header.stamp) -
                           rclcpp::Time(latest_vel_truth_->header.stamp)).seconds();
    if (std::abs(vel_time_diff) > 0.02) { // 20ms tolerance
      return;
    }

    RCLCPP_INFO(test_node_->get_logger(),
               "All messages received with close timestamps - processing metric");

    // Calculate position error
    Eigen::Vector3d true_position(
      latest_pose_truth_->pose.position.x,
      latest_pose_truth_->pose.position.y,
      latest_pose_truth_->pose.position.z
    );

    Eigen::Vector3d est_position(
      latest_pose_est_->pose.pose.position.x,
      latest_pose_est_->pose.pose.position.y,
      latest_pose_est_->pose.pose.position.z
    );

    double position_error = (true_position - est_position).norm();

    // Calculate velocity error
    Eigen::Vector3d true_velocity(
      latest_vel_truth_->twist.linear.x,
      latest_vel_truth_->twist.linear.y,
      latest_vel_truth_->twist.linear.z
    );

    Eigen::Vector3d est_velocity(
      latest_vel_est_->twist.twist.linear.x,
      latest_vel_est_->twist.twist.linear.y,
      latest_vel_est_->twist.twist.linear.z
    );

    double velocity_error = (true_velocity - est_velocity).norm();

    // Calculate orientation error
    Eigen::Quaterniond true_quat(
      latest_pose_truth_->pose.orientation.w,
      latest_pose_truth_->pose.orientation.x,
      latest_pose_truth_->pose.orientation.y,
      latest_pose_truth_->pose.orientation.z
    );

    Eigen::Quaterniond est_quat(
      latest_pose_est_->pose.pose.orientation.w,
      latest_pose_est_->pose.pose.orientation.x,
      latest_pose_est_->pose.pose.orientation.y,
      latest_pose_est_->pose.pose.orientation.z
    );

    double orientation_error = est_quat.angularDistance(true_quat);

    // Store error metrics
    ErrorMetrics metrics;
    metrics.position_error = position_error;
    metrics.velocity_error = velocity_error;
    metrics.orientation_error = orientation_error;
    metrics.timestamp = latest_pose_est_->header.stamp;
    error_metrics_.push_back(metrics);

    // Log progress periodically
    static int count = 0;
    if (++count % 100 == 0) {
      double elapsed = (rclcpp::Time(metrics.timestamp) - test_start_time_).seconds();
      RCLCPP_INFO(test_node_->get_logger(),
          "Test progress: %.1f sec, Current errors - Pos: %.3f m, Vel: %.3f m/s, Orient: %.3f rad",
          elapsed, position_error, velocity_error, orientation_error);
    }

    // Check for test completion
    double elapsed = (rclcpp::Time(metrics.timestamp) - test_start_time_).seconds();
    if (elapsed >= test_duration_seconds_) {
      test_complete_ = true;
    }
  }

  void runTest() {
    // Set up timing
    test_start_time_ = this->test_node_->now();
    rclcpp::Time end_time = test_start_time_ + rclcpp::Duration::from_seconds(test_duration_seconds_);

    // Process messages until test duration elapses
    while (rclcpp::ok() && this->test_node_->now() < end_time) {
      executor_->spin_some(std::chrono::milliseconds(10));
      processMeasurements();
    }

    RCLCPP_INFO(test_node_->get_logger(), "Test complete, analyzing results");

    // Diagnostic output
    RCLCPP_INFO(test_node_->get_logger(),
               "Message reception stats: pose_est=%s, vel_est=%s, pose_truth=%s, vel_truth=%s",
               latest_pose_est_ ? "YES" : "NO",
               latest_vel_est_ ? "YES" : "NO",
               latest_pose_truth_ ? "YES" : "NO",
               latest_vel_truth_ ? "YES" : "NO");

    // Check if we've collected meaningful data
    if (error_metrics_.empty()) {
      FAIL() << "No error metrics collected";
    }

    // Analyze results
    // ...
  }

private:
  rclcpp::Time test_start_time_;
};

// The actual test
TEST_F(Figure8TrackingTest, TrackFigure8Trajectory) {
  runTest();
}

} // namespace test
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
