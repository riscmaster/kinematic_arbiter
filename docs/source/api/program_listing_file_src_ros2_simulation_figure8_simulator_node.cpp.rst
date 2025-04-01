
.. _program_listing_file_src_ros2_simulation_figure8_simulator_node.cpp:

Program Listing for File figure8_simulator_node.cpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_ros2_simulation_figure8_simulator_node.cpp>` (``src/ros2/simulation/figure8_simulator_node.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #include <rclcpp/rclcpp.hpp>
   #include <tf2_ros/transform_broadcaster.h>
   #include <geometry_msgs/msg/point_stamped.hpp>
   #include <geometry_msgs/msg/pose_stamped.hpp>
   #include <geometry_msgs/msg/twist_stamped.hpp>
   #include <geometry_msgs/msg/accel_stamped.hpp>
   #include <sensor_msgs/msg/imu.hpp>
   #include <random>
   #include <chrono>
   #include <tf2_ros/static_transform_broadcaster.h>
   #include <geometry_msgs/msg/transform_stamped.hpp>

   #include "kinematic_arbiter/core/state_index.hpp"
   #include "kinematic_arbiter/ros2/simulation/figure8_simulator_node.hpp"
   #include "kinematic_arbiter/sensors/position_sensor_model.hpp"
   #include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
   #include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
   #include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
   #include <tf2_eigen/tf2_eigen.hpp>
   #include "kinematic_arbiter/ros2/simulation/sensor_publisher.hpp"

   namespace kinematic_arbiter {
   namespace ros2 {
   namespace simulation {
   Figure8SimulatorNode::Figure8SimulatorNode() : Node("figure8_simulator") {
       // Trajectory parameters
       this->declare_parameter("trajectory.max_vel", 1.0);
       this->declare_parameter("trajectory.length", 5.0);
       this->declare_parameter("trajectory.width", 3.0);

       // Timing parameters
       this->declare_parameter("main_update_rate", 100.0);  // Default 100Hz
       this->declare_parameter("position_rate", 30.0);
       this->declare_parameter("pose_rate", 50.0);
       this->declare_parameter("velocity_rate", 50.0);
       this->declare_parameter("imu_rate", 100.0);

       // Frame parameters
       this->declare_parameter("world_frame_id", "world");
       this->declare_parameter("body_frame_id", "base_link");

       // Noise/jitter parameters
       this->declare_parameter("noise_sigma", 0.01);
       this->declare_parameter("time_jitter", 0.0);  // Default 0 for no jitter

       // Sensor transform parameters (offsets from the body frame)
       this->declare_parameter("position_sensor.x_offset", 0.0);
       this->declare_parameter("position_sensor.y_offset", 0.0);
       this->declare_parameter("position_sensor.z_offset", 0.0);

       this->declare_parameter("pose_sensor.x_offset", 0.0);
       this->declare_parameter("pose_sensor.y_offset", 0.0);
       this->declare_parameter("pose_sensor.z_offset", 0.0);

       this->declare_parameter("velocity_sensor.x_offset", 0.0);
       this->declare_parameter("velocity_sensor.y_offset", 0.0);
       this->declare_parameter("velocity_sensor.z_offset", 0.0);

       this->declare_parameter("imu_sensor.x_offset", 0.0);
       this->declare_parameter("imu_sensor.y_offset", 0.0);
       this->declare_parameter("imu_sensor.z_offset", 0.0);

       // In the constructor, add these parameter declarations
       this->declare_parameter("position_topic", "/sensors/position");
       this->declare_parameter("pose_topic", "/sensors/pose");
       this->declare_parameter("velocity_topic", "/sensors/velocity");
       this->declare_parameter("imu_topic", "/sensors/imu");

       this->declare_parameter("position_frame", "position_sensor");
       this->declare_parameter("pose_frame", "pose_sensor");
       this->declare_parameter("velocity_frame", "velocity_sensor");
       this->declare_parameter("imu_frame", "imu_sensor");

       this->declare_parameter("truth_pose_topic", "/truth/pose");
       this->declare_parameter("truth_velocity_topic", "/truth/velocity");

       // Get all parameters
       trajectory_config_.max_velocity = this->get_parameter("trajectory.max_vel").as_double();
       trajectory_config_.length = this->get_parameter("trajectory.length").as_double();
       trajectory_config_.width = this->get_parameter("trajectory.width").as_double();

       main_update_rate_ = this->get_parameter("main_update_rate").as_double();
       position_rate_ = this->get_parameter("position_rate").as_double();
       pose_rate_ = this->get_parameter("pose_rate").as_double();
       velocity_rate_ = this->get_parameter("velocity_rate").as_double();
       imu_rate_ = this->get_parameter("imu_rate").as_double();

       world_frame_id_ = this->get_parameter("world_frame_id").as_string();
       body_frame_id_ = this->get_parameter("body_frame_id").as_string();

       // Get frame IDs - store in existing variables to match pattern
       position_sensor_id_ = this->get_parameter("position_frame").as_string();
       pose_sensor_id_ = this->get_parameter("pose_frame").as_string();
       velocity_sensor_id_ = this->get_parameter("velocity_frame").as_string();
       imu_sensor_id_ = this->get_parameter("imu_frame").as_string();

       // Get topic names
       std::string position_topic = this->get_parameter("position_topic").as_string();
       std::string pose_topic = this->get_parameter("pose_topic").as_string();
       std::string velocity_topic = this->get_parameter("velocity_topic").as_string();
       std::string imu_topic = this->get_parameter("imu_topic").as_string();
       std::string truth_pose_topic = this->get_parameter("truth_pose_topic").as_string();
       std::string truth_velocity_topic = this->get_parameter("truth_velocity_topic").as_string();

       noise_sigma_ = this->get_parameter("noise_sigma").as_double();
       time_jitter_ = this->get_parameter("time_jitter").as_double();

       // Initialize filter wrapper for sensor registration
       position_sensor_model_ = std::make_unique<sensors::PositionSensorModel>();
       pose_sensor_model_ = std::make_unique<sensors::PoseSensorModel>();
       velocity_sensor_model_ = std::make_unique<sensors::BodyVelocitySensorModel>();
       imu_sensor_model_ = std::make_unique<sensors::ImuSensorModel>();

       // Set the transforms in the filter
       publishSensorTransforms();

       // Set up random generators
       std::random_device rd;
       generator_ = std::mt19937(rd());
       noise_dist_ = std::normal_distribution<>(0.0, noise_sigma_);
       jitter_dist_ = std::uniform_real_distribution<>(-time_jitter_, time_jitter_);

       // Create sensor message publishers
       position_publisher_ = std::make_unique<PositionPublisher>(
           this, position_topic, noise_sigma_);

       pose_publisher_ = std::make_unique<PosePublisher>(
           this, pose_topic, noise_sigma_);

       velocity_publisher_ = std::make_unique<VelocityPublisher>(
           this, velocity_topic, noise_sigma_);

       imu_publisher_ = std::make_unique<ImuPublisher>(
           this, imu_topic, noise_sigma_);

       // Create ground truth publishers
       truth_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
           truth_pose_topic, 10);
       truth_velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
           truth_velocity_topic, 10);

       // Create TF broadcaster
       tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

       // Track which iterations to publish each sensor type
       position_divider_ = std::max(1, static_cast<int>(main_update_rate_ / position_rate_));
       pose_divider_ = std::max(1, static_cast<int>(main_update_rate_ / pose_rate_));
       velocity_divider_ = std::max(1, static_cast<int>(main_update_rate_ / velocity_rate_));
       imu_divider_ = std::max(1, static_cast<int>(main_update_rate_ / imu_rate_));

       // Create main timer that drives the simulation
       update_timer_ = this->create_wall_timer(
           std::chrono::milliseconds(static_cast<int>(1000.0 / main_update_rate_)),
           std::bind(&Figure8SimulatorNode::update, this));

       start_time_ = this->now();

       RCLCPP_INFO(this->get_logger(), "Figure8 simulator initialized:");
       RCLCPP_INFO(this->get_logger(), "  Trajectory: length=%.1f, width=%.1f, max_vel=%.1f",
                  trajectory_config_.length, trajectory_config_.width, trajectory_config_.max_velocity);
       RCLCPP_INFO(this->get_logger(), "  Update rates: main=%.1fHz, position=%.1fHz, pose=%.1fHz, velocity=%.1fHz, imu=%.1fHz",
                  main_update_rate_, position_rate_, pose_rate_, velocity_rate_, imu_rate_);
       RCLCPP_INFO(this->get_logger(), "  Noise: sigma=%.3f, time_jitter=%.3fs", noise_sigma_, time_jitter_);

     }
     void Figure8SimulatorNode::update() {
       iteration_++;

       // Calculate current time and state
       double elapsed_seconds = (this->now() - start_time_).seconds();
       auto state = kinematic_arbiter::utils::Figure8Trajectory(elapsed_seconds, trajectory_config_);

       // Always publish ground truth and TF
       publishGroundTruth(state);

       // Publish sensor data at their respective rates
       if (iteration_ % position_divider_ == 0) {
         publishPosition(state, elapsed_seconds);
       }

       if (iteration_ % pose_divider_ == 0) {
         publishPose(state, elapsed_seconds);
       }

       if (iteration_ % velocity_divider_ == 0) {
         publishVelocity(state, elapsed_seconds);
       }

       if (iteration_ % imu_divider_ == 0) {
         publishImu(state, elapsed_seconds);
       }
     }

     void Figure8SimulatorNode::publishGroundTruth(const StateVector& state) {
       auto current_time = this->now();

       // Extract components for convenience
       Eigen::Vector3d position = state.segment<3>(SIdx::Position::Begin());
       Eigen::Quaterniond quaternion(
           state[SIdx::Quaternion::W],
           state[SIdx::Quaternion::X],
           state[SIdx::Quaternion::Y],
           state[SIdx::Quaternion::Z]);
       Eigen::Vector3d linear_velocity = state.segment<3>(SIdx::LinearVelocity::Begin());
       Eigen::Vector3d angular_velocity = state.segment<3>(SIdx::AngularVelocity::Begin());

       // Publish truth pose
       auto pose_msg = geometry_msgs::msg::PoseStamped();
       pose_msg.header.stamp = current_time;
       pose_msg.header.frame_id = world_frame_id_;
       pose_msg.pose.position.x = position.x();
       pose_msg.pose.position.y = position.y();
       pose_msg.pose.position.z = position.z();
       pose_msg.pose.orientation.w = quaternion.w();
       pose_msg.pose.orientation.x = quaternion.x();
       pose_msg.pose.orientation.y = quaternion.y();
       pose_msg.pose.orientation.z = quaternion.z();
       truth_pose_pub_->publish(pose_msg);

       // Publish truth velocity
       auto vel_msg = geometry_msgs::msg::TwistStamped();
       vel_msg.header.stamp = current_time;
       vel_msg.header.frame_id = body_frame_id_;
       vel_msg.twist.linear.x = linear_velocity.x();
       vel_msg.twist.linear.y = linear_velocity.y();
       vel_msg.twist.linear.z = linear_velocity.z();
       vel_msg.twist.angular.x = angular_velocity.x();
       vel_msg.twist.angular.y = angular_velocity.y();
       vel_msg.twist.angular.z = angular_velocity.z();
       truth_velocity_pub_->publish(vel_msg);

       // Publish transform
       geometry_msgs::msg::TransformStamped transform;
       transform.header.stamp = current_time;
       transform.header.frame_id = world_frame_id_;
       transform.child_frame_id = body_frame_id_;
       transform.transform.translation.x = position.x();
       transform.transform.translation.y = position.y();
       transform.transform.translation.z = position.z();
       transform.transform.rotation = pose_msg.pose.orientation;
       tf_broadcaster_->sendTransform(transform);
     }

     void Figure8SimulatorNode::publishPosition(const StateVector& state, double elapsed_seconds) {
       // Apply time jitter
       double time_with_jitter = elapsed_seconds + jitter_dist_(generator_);
       auto timestamp = start_time_ + rclcpp::Duration::from_seconds(time_with_jitter);

       // Get expected measurement
       MeasurementModelInterface::DynamicVector measurement = position_sensor_model_->PredictMeasurement(state);

       // Publish all variants
       position_publisher_->publish(measurement, timestamp, world_frame_id_, generator_);
     }

     void Figure8SimulatorNode::publishPose(const StateVector& state, double elapsed_seconds) {
       // Apply time jitter
       double time_with_jitter = elapsed_seconds + jitter_dist_(generator_);
       auto timestamp = start_time_ + rclcpp::Duration::from_seconds(time_with_jitter);

       // Get expected measurement
       MeasurementModelInterface::DynamicVector measurement = pose_sensor_model_->PredictMeasurement(state);

       // Publish all variants
       pose_publisher_->publish(measurement, timestamp, world_frame_id_, generator_);
     }

     void Figure8SimulatorNode::publishVelocity(const StateVector& state, double elapsed_seconds) {
       // Apply time jitter
       double time_with_jitter = elapsed_seconds + jitter_dist_(generator_);
       auto timestamp = start_time_ + rclcpp::Duration::from_seconds(time_with_jitter);

       // Get expected measurement
       MeasurementModelInterface::DynamicVector measurement = velocity_sensor_model_->PredictMeasurement(state);

       // Publish all variants
       velocity_publisher_->publish(measurement, timestamp, world_frame_id_, generator_);
     }

     void Figure8SimulatorNode::publishImu(const StateVector& state, double elapsed_seconds) {
       // Apply time jitter
       double time_with_jitter = elapsed_seconds + jitter_dist_(generator_);
       auto timestamp = start_time_ + rclcpp::Duration::from_seconds(time_with_jitter);

       // Get expected measurement
       MeasurementModelInterface::DynamicVector measurement = imu_sensor_model_->PredictMeasurement(state);

       // Publish all variants
       imu_publisher_->publish(measurement, timestamp, world_frame_id_, generator_);
     }

     void Figure8SimulatorNode::publishSensorTransforms() {
       tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
       // Get sensor offsets from parameters
       double pos_x = this->get_parameter("position_sensor.x_offset").as_double();
       double pos_y = this->get_parameter("position_sensor.y_offset").as_double();
       double pos_z = this->get_parameter("position_sensor.z_offset").as_double();

       double pose_x = this->get_parameter("pose_sensor.x_offset").as_double();
       double pose_y = this->get_parameter("pose_sensor.y_offset").as_double();
       double pose_z = this->get_parameter("pose_sensor.z_offset").as_double();

       double vel_x = this->get_parameter("velocity_sensor.x_offset").as_double();
       double vel_y = this->get_parameter("velocity_sensor.y_offset").as_double();
       double vel_z = this->get_parameter("velocity_sensor.z_offset").as_double();

       double imu_x = this->get_parameter("imu_sensor.x_offset").as_double();
       double imu_y = this->get_parameter("imu_sensor.y_offset").as_double();
       double imu_z = this->get_parameter("imu_sensor.z_offset").as_double();

       // Set up transforms (from body to sensor)
       position_transform_.header.frame_id = body_frame_id_;
       position_transform_.child_frame_id = position_sensor_id_;
       position_transform_.transform.translation.x = pos_x;
       position_transform_.transform.translation.y = pos_y;
       position_transform_.transform.translation.z = pos_z;
       pose_transform_.header.frame_id = body_frame_id_;
       pose_transform_.child_frame_id = pose_sensor_id_;
       pose_transform_.transform.translation.x = pose_x;
       pose_transform_.transform.translation.y = pose_y;
       pose_transform_.transform.translation.z = pose_z;
       velocity_transform_.header.frame_id = body_frame_id_;
       velocity_transform_.child_frame_id = velocity_sensor_id_;
       velocity_transform_.transform.translation.x = vel_x;
       velocity_transform_.transform.translation.y = vel_y;
       velocity_transform_.transform.translation.z = vel_z;
       imu_transform_.header.frame_id = body_frame_id_;
       imu_transform_.child_frame_id = imu_sensor_id_;
       imu_transform_.transform.translation.x = imu_x;
       imu_transform_.transform.translation.y = imu_y;
       imu_transform_.transform.translation.z = imu_z;

       // Create and publish transforms for each sensor
       tf_static_broadcaster_->sendTransform(position_transform_);
       tf_static_broadcaster_->sendTransform(pose_transform_);
       tf_static_broadcaster_->sendTransform(velocity_transform_);
       tf_static_broadcaster_->sendTransform(imu_transform_);
       position_sensor_model_->SetSensorPoseInBodyFrame(tf2::transformToEigen(position_transform_.transform));
       pose_sensor_model_->SetSensorPoseInBodyFrame(tf2::transformToEigen(pose_transform_.transform));
       velocity_sensor_model_->SetSensorPoseInBodyFrame(tf2::transformToEigen(velocity_transform_.transform));
       imu_sensor_model_->SetSensorPoseInBodyFrame(tf2::transformToEigen(imu_transform_.transform));
     }
   } // namespace simulation
   } // namespace ros2
   } // namespace kinematic_arbiter

   int main(int argc, char* argv[]) {
     rclcpp::init(argc, argv);
     auto node = std::make_shared<kinematic_arbiter::ros2::simulation::Figure8SimulatorNode>();
     rclcpp::spin(node);
     rclcpp::shutdown();
     return 0;
   }
