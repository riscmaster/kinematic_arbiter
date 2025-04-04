
.. _program_listing_file_include_kinematic_arbiter_ros2_kinematic_arbiter_node.hpp:

Program Listing for File kinematic_arbiter_node.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_ros2_kinematic_arbiter_node.hpp>` (``include/kinematic_arbiter/ros2/kinematic_arbiter_node.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   // include/kinematic_arbiter/ros2/kinematic_arbiter_node.hpp
   #pragma once

   #include <memory>
   #include <string>
   #include <vector>

   #include "rclcpp/rclcpp.hpp"
   #include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
   #include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
   #include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
   #include <tf2_ros/transform_listener.h>
   #include <tf2_ros/buffer.h>
   #include "std_srvs/srv/trigger.hpp"

   #include "kinematic_arbiter/ros2/filter_wrapper.hpp"

   namespace kinematic_arbiter {
   namespace ros2 {

   class KinematicArbiterNode : public rclcpp::Node {
   public:
     KinematicArbiterNode();

     virtual ~KinematicArbiterNode() = default;

   private:
     // Parameters
     double publish_rate_;
     std::string world_frame_id_;
     std::string body_frame_id_;

     // TF components
     std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
     std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

     // Filter wrapper
     std::unique_ptr<FilterWrapper> filter_wrapper_;

     // Publishers
     rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
     rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
     rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_pub_;

     // Timer
     rclcpp::TimerBase::SharedPtr publish_timer_;

     // True state subscribers for debugging
     rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth_pose_sub_;
     rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr truth_velocity_sub_;

     // Service for filter reset
     rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

     // Callbacks
     void publishEstimates();

     // Sensor initialization
     void configureSensorsFromLoadedParams();

     std::vector<std::string> getSensorNames(const std::string& sensor_type);

     // Callbacks for true state
     void truthPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
     void truthVelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

     // Reset service callback
     void handleResetService(
         const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
   };

   } // namespace ros2
   } // namespace kinematic_arbiter
