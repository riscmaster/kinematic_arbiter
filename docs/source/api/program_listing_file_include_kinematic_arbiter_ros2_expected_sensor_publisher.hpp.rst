
.. _program_listing_file_include_kinematic_arbiter_ros2_expected_sensor_publisher.hpp:

Program Listing for File expected_sensor_publisher.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_ros2_expected_sensor_publisher.hpp>` (``include/kinematic_arbiter/ros2/expected_sensor_publisher.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <rclcpp/rclcpp.hpp>
   #include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
   #include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
   #include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>

   namespace kinematic_arbiter {
   namespace ros2 {

   template <typename MsgType>
   class ExpectedSensorPublisher {
   public:
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
     virtual MsgType createBound(const MsgType& expected_msg, double sigma_factor) = 0;

     rclcpp::Node* node_;
     std::string topic_prefix_;

     // Publishers for expected value and bounds
     typename rclcpp::Publisher<MsgType>::SharedPtr expected_pub_;
     typename rclcpp::Publisher<MsgType>::SharedPtr upper_bound_pub_;
     typename rclcpp::Publisher<MsgType>::SharedPtr lower_bound_pub_;
   };

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
