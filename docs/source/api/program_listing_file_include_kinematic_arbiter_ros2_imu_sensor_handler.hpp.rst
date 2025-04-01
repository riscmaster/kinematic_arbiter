
.. _program_listing_file_include_kinematic_arbiter_ros2_imu_sensor_handler.hpp:

Program Listing for File imu_sensor_handler.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_ros2_imu_sensor_handler.hpp>` (``include/kinematic_arbiter/ros2/imu_sensor_handler.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <sensor_msgs/msg/imu.hpp>
   #include "kinematic_arbiter/ros2/sensor_handler.hpp"
   #include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
   #include "kinematic_arbiter/ros2/ros2_utils.hpp"
   namespace kinematic_arbiter {
   namespace ros2 {

   class ImuSensorHandler : public SensorHandler<sensor_msgs::msg::Imu> {
   public:
     using Base = SensorHandler<sensor_msgs::msg::Imu>;
     using ModelType = kinematic_arbiter::sensors::ImuSensorModel;
     using MeasurementIndex = ModelType::MeasurementIndex;

     ImuSensorHandler(
         rclcpp::Node* node,
         std::shared_ptr<Filter> filter,
         std::shared_ptr<tf2_ros::Buffer> tf_buffer,
         std::shared_ptr<utils::TimeManager> time_manager,
         const std::string& sensor_name,
         const std::string& topic,
         const std::string& sensor_frame_id,
         const std::string& reference_frame_id,
         const std::string& body_frame_id)
       : Base(node, filter, tf_buffer, time_manager, sensor_name, topic,
             sensor_frame_id, reference_frame_id, body_frame_id,
             std::make_shared<ModelType>(),
             SensorType::Imu) {}

   protected:

     bool msgToVector(const sensor_msgs::msg::Imu& msg, DynamicVector& vector) override {
       vector.resize(ModelType::kMeasurementDimension);

       // Angular velocity (gyroscope measurements)
       vector(MeasurementIndex::GX) = msg.angular_velocity.x;
       vector(MeasurementIndex::GY) = msg.angular_velocity.y;
       vector(MeasurementIndex::GZ) = msg.angular_velocity.z;

       // Linear acceleration (accelerometer measurements)
       vector(MeasurementIndex::AX) = msg.linear_acceleration.x;
       vector(MeasurementIndex::AY) = msg.linear_acceleration.y;
       vector(MeasurementIndex::AZ) = msg.linear_acceleration.z;

       return true;
     }

     sensor_msgs::msg::Imu vectorToMsg(
         const DynamicVector& vector,
         const std_msgs::msg::Header& header) override {

       sensor_msgs::msg::Imu msg;
       msg.header = header;

       // Angular velocity
       msg.angular_velocity.x = vector(MeasurementIndex::GX);
       msg.angular_velocity.y = vector(MeasurementIndex::GY);
       msg.angular_velocity.z = vector(MeasurementIndex::GZ);

       // Linear acceleration
       msg.linear_acceleration.x = vector(MeasurementIndex::AX);
       msg.linear_acceleration.y = vector(MeasurementIndex::AY);
       msg.linear_acceleration.z = vector(MeasurementIndex::AZ);

       // Orientation (identity quaternion as we don't produce this)
       msg.orientation.w = 1.0;
       msg.orientation.x = 0.0;
       msg.orientation.y = 0.0;
       msg.orientation.z = 0.0;

       // Set orientation covariance to high uncertainty
       for (int i = 0; i < 9; ++i) {
         msg.orientation_covariance[i] = (i % 4 == 0) ? 99999.0 : 0.0;
       }

       return msg;
     }

     sensor_msgs::msg::Imu applyBound(
         const sensor_msgs::msg::Imu& base_msg,
         const DynamicCovariance& covariance,
         bool positive) override {

       auto bounded_msg = base_msg;
       double sign = positive ? 1.0 : -1.0;

       // Angular velocity bounds using measurement indices
       double gx_std_dev = std::sqrt(covariance(MeasurementIndex::GX, MeasurementIndex::GX));
       double gy_std_dev = std::sqrt(covariance(MeasurementIndex::GY, MeasurementIndex::GY));
       double gz_std_dev = std::sqrt(covariance(MeasurementIndex::GZ, MeasurementIndex::GZ));

       bounded_msg.angular_velocity.x += sign * gx_std_dev * SIGMA_BOUND_FACTOR;
       bounded_msg.angular_velocity.y += sign * gy_std_dev * SIGMA_BOUND_FACTOR;
       bounded_msg.angular_velocity.z += sign * gz_std_dev * SIGMA_BOUND_FACTOR;

       // Linear acceleration bounds using measurement indices
       double ax_std_dev = std::sqrt(covariance(MeasurementIndex::AX, MeasurementIndex::AX));
       double ay_std_dev = std::sqrt(covariance(MeasurementIndex::AY, MeasurementIndex::AY));
       double az_std_dev = std::sqrt(covariance(MeasurementIndex::AZ, MeasurementIndex::AZ));

       bounded_msg.linear_acceleration.x += sign * ax_std_dev * SIGMA_BOUND_FACTOR;
       bounded_msg.linear_acceleration.y += sign * ay_std_dev * SIGMA_BOUND_FACTOR;
       bounded_msg.linear_acceleration.z += sign * az_std_dev * SIGMA_BOUND_FACTOR;

       return bounded_msg;
     }
   };

   } // namespace ros2
   } // namespace kinematic_arbiter
