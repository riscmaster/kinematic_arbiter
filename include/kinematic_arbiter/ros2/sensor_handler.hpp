#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/buffer.h>
#include <memory>
#include <string>
#include <Eigen/Dense>

#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"
#include "kinematic_arbiter/core/mediation_types.hpp"

namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief Base template class for sensor handlers
 *
 * Manages subscription, transforms, and expected measurement publishing for a sensor
 */
template <typename MsgType>
class SensorHandler {
public:
  using MeasurementModelInterface = kinematic_arbiter::core::MeasurementModelInterface;
  using DynamicVector = MeasurementModelInterface::DynamicVector;
  using DynamicCovariance = MeasurementModelInterface::DynamicCovariance;
  using Filter = kinematic_arbiter::core::MediatedKalmanFilter<kinematic_arbiter::core::StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>;
  using SensorType = kinematic_arbiter::core::SensorType;

  // Constant for bound calculation
  static constexpr double SIGMA_BOUND_FACTOR = 3.0;

  /**
   * @brief Constructor for SensorHandler
   *
   * @param node ROS2 node to create publishers and subscribers from
   * @param filter Shared pointer to the core filter
   * @param tf_buffer TF buffer for transform lookups
   * @param sensor_name Human-readable sensor name
   * @param topic Topic to subscribe to
   * @param sensor_frame_id Frame ID of the sensor
   * @param reference_frame_id Reference frame for measurements
   * @param body_frame_id Body frame ID for transformations
   * @param sensor_type Type of sensor from core enum
   */
  SensorHandler(
      rclcpp::Node* node,
      std::shared_ptr<Filter> filter,
      std::shared_ptr<tf2_ros::Buffer> tf_buffer,
      std::shared_ptr<utils::TimeManager> time_manager,
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id,
      const std::string& reference_frame_id,
      const std::string& body_frame_id,
      std::shared_ptr<MeasurementModelInterface> sensor_model,
      SensorType sensor_type)
    : node_(node),
      filter_(filter),
      tf_buffer_(tf_buffer),
      time_manager_(time_manager),
      sensor_name_(sensor_name),
      topic_(topic),
      sensor_frame_id_(sensor_frame_id),
      reference_frame_id_(reference_frame_id),
      body_frame_id_(body_frame_id),
      sensor_model_(sensor_model),
      sensor_type_(sensor_type) {

    // Validate critical pointers
    if (!node_) {
      throw std::invalid_argument("SensorHandler constructor given null node pointer");
    }
    if (!filter_) {
      throw std::invalid_argument("SensorHandler constructor given null filter pointer");
    }
    if (!tf_buffer_) {
      throw std::invalid_argument("SensorHandler constructor given null TF buffer pointer");
    }
    if (!sensor_model_) {
      throw std::invalid_argument("SensorHandler constructor given null sensor model");
    }

    // Create unique sensor ID from type and name
    sensor_id_ = kinematic_arbiter::core::SensorTypeToString(sensor_type_) +
                "_" + sensor_name_;

    // Register the sensor with the filter (same for all sensor types)
    size_t sensor_index = filter_->AddSensor(sensor_model_);

    // Store sensor index for faster lookups
    sensor_index_ = sensor_index;

    // Lookup static transform once at initialization
    lookupStaticTransform();

    // Create the subscriber
    subscription_ = node_->create_subscription<MsgType>(
        topic_, 10,
        [this](const typename MsgType::SharedPtr msg) {
          this->messageCallback(msg);
        });

    // Create expected measurement publishers
    expected_pub_ = node_->create_publisher<MsgType>(
        topic_ + "/expected", 10);

    upper_bound_pub_ = node_->create_publisher<MsgType>(
        topic_ + "/expected/upper_bound", 10);

    lower_bound_pub_ = node_->create_publisher<MsgType>(
        topic_ + "/expected/lower_bound", 10);

    RCLCPP_INFO(node_->get_logger(), "Created %s sensor handler '%s' (ID: %s) on topic '%s'",
                kinematic_arbiter::core::SensorTypeToString(sensor_type_).c_str(),
                sensor_name_.c_str(), sensor_id_.c_str(), topic_.c_str());
  }

  /**
   * @brief Get the sensor ID
   */
  const std::string& getSensorId() const {
    return sensor_id_;
  }

  /**
   * @brief Set validation parameters for the sensor model
   *
   * @param p2m_noise_ratio Process to measurement noise ratio
   * @param mediation_action Mediation action
   */
  void setValidationParams(double p2m_noise_ratio, kinematic_arbiter::core::MediationAction mediation_action) {
    if (sensor_model_) {
      // ValidationParams is part of MeasurementModelInterface
      kinematic_arbiter::core::MeasurementModelInterface::ValidationParams params;
      params.process_to_measurement_noise_ratio = p2m_noise_ratio;
      params.mediation_action = mediation_action;
      sensor_model_->SetValidationParams(params);

      RCLCPP_INFO(node_->get_logger(), "Set validation params for %s: p2m_ratio=%.2f, action=%s",
                  sensor_name_.c_str(), p2m_noise_ratio,
                  mediation_action == kinematic_arbiter::core::MediationAction::ForceAccept ? "force_accept" :
                  (mediation_action == kinematic_arbiter::core::MediationAction::AdjustCovariance ? "adjust_covariance" : "reject"));
    }
  }

protected:
  /**
   * @brief Convert message to measurement vector
   */
  virtual bool msgToVector(const MsgType& msg, DynamicVector& vector) = 0;

  /**
   * @brief Convert measurement vector to message
   *
   * The covariance is not needed as a parameter because we can get it directly from the filter
   */
  virtual MsgType vectorToMsg(
      const DynamicVector& vector,
      const std_msgs::msg::Header& header) = 0;

  /**
   * @brief Apply a sigma bound offset to a message
   *
   * Creates a new message with a sigma-bound offset in the positive or negative direction
   *
   * @param base_msg The base message to modify
   * @param covariance The measurement covariance
   * @param positive True for upper bound, false for lower bound
   * @return The bounded message
   */
  virtual MsgType applyBound(
      const MsgType& base_msg,
      const DynamicCovariance& covariance,
      bool positive) = 0;


private:
  // Node and configuration
  rclcpp::Node* node_;
  std::shared_ptr<Filter> filter_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<utils::TimeManager> time_manager_;


  // Sensor information
  std::string sensor_id_;
  std::string sensor_name_;
  std::string topic_;
  std::string sensor_frame_id_;
  std::string reference_frame_id_;
  std::string body_frame_id_;
  std::shared_ptr<MeasurementModelInterface> sensor_model_;
  SensorType sensor_type_;
  size_t sensor_index_;

  // Transform cache
  Eigen::Isometry3d sensor_to_body_transform_;

  // Publishers and subscriber
  typename rclcpp::Subscription<MsgType>::SharedPtr subscription_;
  typename rclcpp::Publisher<MsgType>::SharedPtr expected_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr upper_bound_pub_;
  typename rclcpp::Publisher<MsgType>::SharedPtr lower_bound_pub_;

  // Add this member variable to the SensorHandler class's private section
  bool has_valid_transform_ = false;

  /**
   * @brief Look up the static transform from sensor to body frame
   * @return True if transform was successfully found or already exists
   */
  bool lookupStaticTransform() {
    if (sensor_frame_id_ == body_frame_id_) {
      // Special case: sensor is in the body frame, use identity
      sensor_to_body_transform_ = Eigen::Isometry3d::Identity();
      filter_->SetSensorPoseInBodyFrameByIndex(sensor_index_, sensor_to_body_transform_);
      has_valid_transform_ = true;
      RCLCPP_INFO(node_->get_logger(), "Sensor %s is in body frame %s, using identity transform",
                  sensor_id_.c_str(), body_frame_id_.c_str());
      return true;
    }

    try {
      // Try to look up the static transform
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped = tf_buffer_->lookupTransform(
          body_frame_id_, sensor_frame_id_, tf2::TimePointZero);

      // Convert to Eigen transform
      Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
      transform.translation() = Eigen::Vector3d(
          transform_stamped.transform.translation.x,
          transform_stamped.transform.translation.y,
          transform_stamped.transform.translation.z);

      Eigen::Quaterniond q(
          transform_stamped.transform.rotation.w,
          transform_stamped.transform.rotation.x,
          transform_stamped.transform.rotation.y,
          transform_stamped.transform.rotation.z);
      transform.linear() = q.toRotationMatrix();

      // Store and set transform
      sensor_to_body_transform_ = transform;
      filter_->SetSensorPoseInBodyFrameByIndex(sensor_index_, sensor_to_body_transform_);
      has_valid_transform_ = true;

      RCLCPP_INFO(node_->get_logger(), "Set transform from %s to %s for sensor %s: [%f, %f, %f]",
                  sensor_frame_id_.c_str(), body_frame_id_.c_str(), sensor_id_.c_str(),
                  transform.translation().x(),
                  transform.translation().y(),
                  transform.translation().z());
      return true;
    } catch (const tf2::TransformException& ex) {
      // Report the error without setting any transform
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                 "Waiting for transform from %s to %s for sensor %s: %s",
                 sensor_frame_id_.c_str(), body_frame_id_.c_str(),
                 sensor_name_.c_str(), ex.what());
      has_valid_transform_ = false;
      return false;
    }
  }

  /**
   * @brief Callback for incoming sensor messages
   */
  void messageCallback(const typename MsgType::SharedPtr msg) {
    if(!time_manager_->isInitialized()) {
      RCLCPP_WARN(node_->get_logger(), "Time manager not initialized for sensor %s", sensor_id_.c_str());
      return;
    }

    if (!msg) {
      RCLCPP_WARN(node_->get_logger(), "Received null message for sensor %s", sensor_id_.c_str());
      return;
    }

    // Check for transform if we don't have a valid one yet
    if (!has_valid_transform_) {
      if (!lookupStaticTransform()) {
        // Still don't have transform, skip processing this message
        // Only log at most once per second to avoid spamming
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                            "Skipping measurement for sensor '%s' - waiting for transform from %s to %s",
                            sensor_name_.c_str(), sensor_frame_id_.c_str(), body_frame_id_.c_str());
        return;
      }
    }

    // Convert to vector
    DynamicVector measurement;
    if (!msgToVector(*msg, measurement)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to convert message to measurement vector for sensor %s",
                  sensor_id_.c_str());
      return;
    }

    // Get timestamp from message using our offset-adjusted conversion
    double timestamp = time_manager_->rosTimeToFilterTime(rclcpp::Time(msg->header.stamp));

    // Process the measurement
    bool success = filter_->ProcessMeasurementByIndex(sensor_index_, measurement, timestamp);

    if (!success) {
      RCLCPP_WARN(node_->get_logger(), "Failed to process measurement for sensor %s",
                 sensor_id_.c_str());
    }

    // Always publish expected measurements regardless of process success
    publishExpectedMeasurement(msg->header);
  }

  /**
   * @brief Publish expected measurement and bounds
   */
  void publishExpectedMeasurement(const std_msgs::msg::Header& header) {
    // Get the timestamp using our offset-adjusted conversion
    double timestamp = time_manager_->rosTimeToFilterTime(rclcpp::Time(header.stamp));

    // Predict to the measurement time
    auto state = filter_->GetStateEstimate(timestamp);

    // Get expected measurement for this sensor at this state
    DynamicVector expected_vector;
    if (!filter_->GetExpectedMeasurementByIndex(sensor_index_, expected_vector, state)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to get expected measurement for sensor %s",
                 sensor_id_.c_str());
      return;
    }

    // Get the measurement covariance
    DynamicCovariance covariance;
    if (!filter_->GetSensorCovarianceByIndex(sensor_index_, covariance)) {
      RCLCPP_WARN(node_->get_logger(), "Failed to get covariance for sensor %s",
                 sensor_id_.c_str());
      return;
    }

    // Convert to message
    auto expected_msg = vectorToMsg(expected_vector, header);

    // Apply bounds
    auto upper_bound = applyBound(expected_msg, covariance, true);
    auto lower_bound = applyBound(expected_msg, covariance, false);

    // Publish all three messages
    expected_pub_->publish(expected_msg);
    upper_bound_pub_->publish(upper_bound);
    lower_bound_pub_->publish(lower_bound);
  }
};


} // namespace ros2
} // namespace kinematic_arbiter
