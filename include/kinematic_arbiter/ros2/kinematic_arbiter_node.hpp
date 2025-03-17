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

#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"

namespace kinematic_arbiter {
namespace ros2 {

class KinematicArbiterNode : public rclcpp::Node {
public:
  KinematicArbiterNode();

private:
  // Wrapper instance
  std::unique_ptr<wrapper::FilterWrapper> filter_wrapper_;

  // Main state publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_pub_;

  // Subscriber and associated publisher containers
  struct SensorSubscription {
    std::string sensor_id;
    std::string topic;
    rclcpp::SubscriptionBase::SharedPtr subscription;

    // Publisher for expected measurements (what the filter expects to measure)
    rclcpp::PublisherBase::SharedPtr expected_pub;
  };

  std::vector<SensorSubscription> position_subs_;
//   std::vector<SensorSubscription> pose_subs_;
//   std::vector<SensorSubscription> velocity_subs_;
//   std::vector<SensorSubscription> imu_subs_;

  // Services
  // To be implemented later when interfaces are available
  // rclcpp::Service<RegisterSensorSrv>::SharedPtr register_sensor_srv_;

  // Timer for publishing
  rclcpp::TimerBase::SharedPtr publish_timer_;

  // Parameters
  double publish_rate_;
  double max_delay_window_;
  std::string frame_id_;

  // Callbacks
  void positionCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg, const std::string& sensor_id);
//   void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, const std::string& sensor_id);
//   void velocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg, const std::string& sensor_id);
//   void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg, const std::string& sensor_id);

  // Service callback (to be implemented later)
  // void registerSensorCallback(
  //   const std::shared_ptr<RegisterSensorSrv::Request> request,
  //   const std::shared_ptr<RegisterSensorSrv::Response> response);

  void publishEstimates();

  // Helper methods
  void createSubscription(const std::string& sensor_type, const std::string& sensor_id, const std::string& topic);

  // Helper method to create both subscription and expected measurement publisher
  template<typename MsgType>
  void createSensorPair(
      const std::string& sensor_type,
      const std::string& sensor_id,
      const std::string& topic,
      std::vector<SensorSubscription>& subscription_list);
};

} // namespace ros2
} // namespace kinematic_arbiter
