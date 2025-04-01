
.. _program_listing_file_include_kinematic_arbiter_ros2_filter_wrapper.hpp:

Program Listing for File filter_wrapper.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_ros2_filter_wrapper.hpp>` (``include/kinematic_arbiter/ros2/filter_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <memory>
   #include <string>
   #include <vector>

   #include "rclcpp/rclcpp.hpp"
   #include "tf2_ros/buffer.h"
   #include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
   #include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
   #include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"

   #include "kinematic_arbiter/models/rigid_body_state_model.hpp"
   #include "kinematic_arbiter/core/state_index.hpp"
   #include "kinematic_arbiter/core/state_model_interface.hpp"
   #include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
   #include "kinematic_arbiter/ros2/position_sensor_handler.hpp"
   #include "kinematic_arbiter/ros2/pose_sensor_handler.hpp"
   #include "kinematic_arbiter/ros2/velocity_sensor_handler.hpp"
   #include "kinematic_arbiter/ros2/imu_sensor_handler.hpp"
   #include "kinematic_arbiter/ros2/ros2_utils.hpp"

   namespace kinematic_arbiter {
   namespace ros2 {

   class FilterWrapper {
     using StateIndex = ::kinematic_arbiter::core::StateIndex;
     using StateModelInterface = ::kinematic_arbiter::core::StateModelInterface;
     using MediatedKalmanFilter = ::kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, StateModelInterface>;

   public:
     FilterWrapper(
         rclcpp::Node* node,
         std::shared_ptr<tf2_ros::Buffer> tf_buffer,
         const ::kinematic_arbiter::models::RigidBodyStateModel::Params& model_params,
         const std::string& body_frame_id = "base_link",
         const std::string& world_frame_id = "map");

     void reset() {
       filter_->reset();
       time_manager_->setReferenceTime(node_->now());
     }

     void initializeTimeManager(const rclcpp::Time& reference_time) {
       time_manager_->setReferenceTime(reference_time);
       RCLCPP_INFO(node_->get_logger(),
                   "Initialized time manager with reference time: %ld ns",
                   reference_time.nanoseconds());
     }

     bool isInitialized() const;

     void setMaxDelayWindow(double seconds);

     bool addPositionSensor(
         const std::string& sensor_name,
         const std::string& topic,
         const std::string& sensor_frame_id,
         double p2m_noise_ratio = 2.0,
         const std::string& mediation_action = "force_accept");

     bool addPoseSensor(
         const std::string& sensor_name,
         const std::string& topic,
         const std::string& sensor_frame_id,
         double p2m_noise_ratio = 2.0,
         const std::string& mediation_action = "force_accept");

     bool addVelocitySensor(
         const std::string& sensor_name,
         const std::string& topic,
         const std::string& sensor_frame_id,
         double p2m_noise_ratio = 2.0,
         const std::string& mediation_action = "force_accept");

     bool addImuSensor(
         const std::string& sensor_name,
         const std::string& topic,
         const std::string& sensor_frame_id,
         double p2m_noise_ratio = 2.0,
         const std::string& mediation_action = "force_accept");

     geometry_msgs::msg::PoseWithCovarianceStamped getPoseEstimate();

     geometry_msgs::msg::TwistWithCovarianceStamped getVelocityEstimate();

     geometry_msgs::msg::AccelWithCovarianceStamped getAccelerationEstimate();

     void setPoseEstimate(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
     void setVelocityEstimate(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

   private:
     // ROS node and TF buffer
     rclcpp::Node* node_;
     std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

     // Frame IDs
     std::string body_frame_id_;
     std::string world_frame_id_;

     // The filter
     std::shared_ptr<MediatedKalmanFilter> filter_;

     // New TimeManager instance
     std::shared_ptr<utils::TimeManager> time_manager_;

     // Sensor handlers
     std::vector<std::shared_ptr<PositionSensorHandler>> position_handlers_;
     std::vector<std::shared_ptr<PoseSensorHandler>> pose_handlers_;
     std::vector<std::shared_ptr<VelocitySensorHandler>> velocity_handlers_;
     std::vector<std::shared_ptr<ImuSensorHandler>> imu_handlers_;

     // Helper method to convert string to MediationAction enum
     kinematic_arbiter::core::MediationAction stringToMediationAction(const std::string& action_str) const;

     // New helper method to convert ROS time to filter time
     double convertRosTimeToFilterTime(const rclcpp::Time& ros_time) {
       if (!time_manager_->isInitialized()) {
         // Auto-initialize with this time if not already done
         initializeTimeManager(ros_time);
       }
       return time_manager_->rosTimeToFilterTime(ros_time);
     }

     // New helper method to convert filter time to ROS time
     rclcpp::Time convertFilterTimeToRosTime(double filter_time) {
       if (!time_manager_->isInitialized()) {
         // If not initialized, use the node's current time as reference
         initializeTimeManager(node_->now());
       }
       return time_manager_->filterTimeToRosTime(filter_time);
     }
   };

   } // namespace ros2
   } // namespace kinematic_arbiter
