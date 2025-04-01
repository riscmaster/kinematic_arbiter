
.. _program_listing_file_include_kinematic_arbiter_ros2_ros2_utils.hpp:

Program Listing for File ros2_utils.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_ros2_ros2_utils.hpp>` (``include/kinematic_arbiter/ros2/ros2_utils.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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

   class TimeManager {
   public:
     TimeManager() : initialized_(false), reference_time_(0) {}

     void setReferenceTime(const rclcpp::Time& reference_time) {
       if (!initialized_) {
         reference_time_ = reference_time.nanoseconds();
         initialized_ = true;
       }
     }

     double rosTimeToFilterTime(const rclcpp::Time& ros_time) {
       if (!initialized_) {
         throw std::runtime_error("TimeManager not initialized with reference time");
       }

       // Calculate time difference in seconds
       double filter_time = (ros_time.nanoseconds() - reference_time_) * 1e-9;
       return filter_time;
     }

     rclcpp::Time filterTimeToRosTime(double filter_time) {
       if (!initialized_) {
         throw std::runtime_error("TimeManager not initialized with reference time");
       }

       // Calculate absolute time in nanoseconds
       int64_t nanoseconds = reference_time_ + static_cast<int64_t>(filter_time * 1e9);
       return rclcpp::Time(nanoseconds);
     }

     bool isInitialized() const {
       return initialized_;
     }

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
