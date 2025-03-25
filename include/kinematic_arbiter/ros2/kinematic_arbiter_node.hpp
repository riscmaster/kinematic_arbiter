// include/kinematic_arbiter/ros2/kinematic_arbiter_node.hpp
#pragma once

#include <memory>
#include <string>
#include <map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"

namespace kinematic_arbiter {
namespace ros2 {

class KinematicArbiterNode : public rclcpp::Node {
public:
  KinematicArbiterNode();
  virtual ~KinematicArbiterNode() = default;

private:
  // Sensor subscription structure
  struct SensorSubscription {
    std::string sensor_id;
    std::string topic;
    rclcpp::SubscriptionBase::SharedPtr subscription;
    rclcpp::PublisherBase::SharedPtr expected_pub;
  };

  // Parameters
  double publish_rate_;
  double max_delay_window_;
  std::string world_frame_id_;  // Frame for publishing state estimates
  std::string body_frame_id_;   // Body/vehicle frame

  // TF components
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Filter wrapper
  std::unique_ptr<wrapper::FilterWrapper> filter_wrapper_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // Sensor subscriptions
  std::vector<SensorSubscription> position_subs_;
  // Future sensor types
  // std::vector<SensorSubscription> pose_subs_;
  // std::vector<SensorSubscription> velocity_subs_;
  // std::vector<SensorSubscription> imu_subs_;

  // Callbacks
  void positionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg,
                        const std::string& sensor_id);
  void publishEstimates();

  // Utility methods
  template<typename MsgType>
  void createSensorPair(const std::string& sensor_type,
                        const std::string& sensor_id,
                        const std::string& topic,
                        std::vector<SensorSubscription>& subscription_list);

  // Method to update sensor transform from TF
  bool updateSensorTransform(const std::string& sensor_id,
                             const std::string& frame_id,
                             const rclcpp::Time& time);
};

} // namespace ros2
} // namespace kinematic_arbiter
