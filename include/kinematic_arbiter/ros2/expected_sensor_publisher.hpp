#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>

namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Base class for expected sensor publishers
 *
 * Publishes expected value and bounds for a sensor
 */
template <typename MsgType>
class ExpectedSensorPublisher {
public:
  /**
   * @brief Constructor for ExpectedSensorPublisher
   *
   * @param node The ROS2 node to create publishers from
   * @param topic_prefix The prefix for the sensor topics
   */
  ExpectedSensorPublisher(
      rclcpp::Node* node,
      const std::string& topic_prefix)
    : node_(node),
      topic_prefix_(topic_prefix) {

    // Create publishers with consistent naming
    expected_pub_ = node_->create_publisher<MsgType>(
        topic_prefix_ + "/expected", 10);

    upper_bound_pub_ = node_->create_publisher<MsgType>(
        topic_prefix_ + "/upper_bound", 10);

    lower_bound_pub_ = node_->create_publisher<MsgType>(
        topic_prefix_ + "/lower_bound", 10);

    RCLCPP_INFO(node_->get_logger(), "Created expected sensor publisher for '%s'",
                topic_prefix_.c_str());
  }

  /**
   * @brief Publish expected value and bounds
   *
   * @param expected_msg The expected value message
   */
  void publish(const MsgType& expected_msg) {
    // Publish expected value
    expected_pub_->publish(expected_msg);

    // Create and publish upper/lower bounds
    MsgType upper_bound = createBound(expected_msg, 3.0);  // 3-sigma upper bound
    MsgType lower_bound = createBound(expected_msg, -3.0); // 3-sigma lower bound

    upper_bound_pub_->publish(upper_bound);
    lower_bound_pub_->publish(lower_bound);
  }

protected:
  /**
   * @brief Create a bound message from the expected value
   *
   * @param expected_msg The expected value message
   * @param sigma_factor The factor to multiply the standard deviation by (e.g., 3.0 for 3-sigma)
   * @return The bound message
   */
  virtual MsgType createBound(const MsgType& expected_msg, double sigma_factor) = 0;

  rclcpp::Node* node_;
  std::string topic_prefix_;

  // Publishers for expected value and bounds
  typename rclcpp::Publisher<MsgType>::SharedPtr expected_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr upper_bound_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr lower_bound_pub_;
};

/**
 * @brief Expected pose publisher
 */
class ExpectedPosePublisher : public ExpectedSensorPublisher<geometry_msgs::msg::PoseWithCovarianceStamped> {
public:
  using MsgType = geometry_msgs::msg::PoseWithCovarianceStamped;

  ExpectedPosePublisher(rclcpp::Node* node, const std::string& topic_prefix)
    : ExpectedSensorPublisher<MsgType>(node, topic_prefix) {}

protected:
  MsgType createBound(const MsgType& expected_msg, double sigma_factor) override {
    MsgType bound = expected_msg;

    // Position bounds based on position covariance
    for (int i = 0; i < 3; ++i) {
      double std_dev = std::sqrt(expected_msg.pose.covariance[i*6 + i]);
      double offset = std_dev * sigma_factor;

      if (i == 0) bound.pose.pose.position.x += offset;
      if (i == 1) bound.pose.pose.position.y += offset;
      if (i == 2) bound.pose.pose.position.z += offset;
    }

    // We could also adjust orientation but it's more complex
    // Just copy the original orientation for now

    return bound;
  }
};

// Similar classes for ExpectedVelocityPublisher and ExpectedAccelerationPublisher
// ...

} // namespace ros2
} // namespace kinematic_arbiter
