#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/accel_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <tf2_eigen/tf2_eigen.hpp>  // Use angle brackets and .hpp extension
#include "kinematic_arbiter/core/trajectory_utils.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"
#include "kinematic_arbiter/core/statistical_utils.hpp"
#include <random>
#include <memory>
#include <string>
#include <map>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <cmath>

namespace kinematic_arbiter {
namespace ros2 {
namespace simulation {

/**
 * @brief Position sensor simulator that leverages PositionSensorModel
 */
class PositionSensorSimulator {
public:
  using SIdx = core::StateIndex;
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;
  using MeasurementVector = Eigen::Vector3d;
  using MeasurementCovariance = Eigen::Matrix3d;

  PositionSensorSimulator(
      rclcpp::Node* node,
      const std::string& sensor_name,
      const Eigen::Isometry3d& transform,
      double noise_sigma,
      double publish_rate,
      const kinematic_arbiter::utils::Figure8Config& trajectory_config)
    : node_(node),
      sensor_name_(sensor_name),
      sensor_model_(std::make_shared<sensors::PositionSensorModel>(transform)),
      noise_sigma_(noise_sigma),
      publish_rate_(publish_rate),
      trajectory_config_(trajectory_config) {

    // Create publishers
    truth_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "sensors/" + sensor_name_ + "/truth", 10);

    measurement_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "sensors/" + sensor_name_, 10);

    upper_bound_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "sensors/" + sensor_name_ + "/upper_bound", 10);

    lower_bound_pub_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "sensors/" + sensor_name_ + "/lower_bound", 10);

    // Initialize random generator
    std::random_device rd;
    generator_ = std::mt19937(rd());

    // Create covariance from sigma
    updateCovariance();
  }

  void updateConfiguration(
      const Eigen::Isometry3d& transform,
      double noise_sigma,
      double publish_rate) {
    sensor_model_ = std::make_shared<sensors::PositionSensorModel>(transform);
    noise_sigma_ = noise_sigma;
    publish_rate_ = publish_rate;
    updateCovariance();
  }

  void updateTrajectoryConfig(const kinematic_arbiter::utils::Figure8Config& config) {
    trajectory_config_ = config;
  }

  void generateAndPublishMeasurements(double elapsed_seconds) {
    auto current_time = node_->now();

    // Generate trajectory state at current time
    StateVector state = kinematic_arbiter::utils::Figure8Trajectory(
        elapsed_seconds, trajectory_config_);

    // Get perfect measurement from the sensor model
    MeasurementVector true_measurement = sensor_model_->PredictMeasurement(state);

    // Sample noisy measurement using the new utility function
    Eigen::Vector3d noise = kinematic_arbiter::utils::generateMultivariateNoise(
        measurement_covariance_, generator_);
    MeasurementVector noisy_measurement = true_measurement + noise;

    // Calculate 3-sigma bounds
    MeasurementVector lower_bound = true_measurement - MeasurementVector::Constant(3.0 * noise_sigma_);
    MeasurementVector upper_bound = true_measurement + MeasurementVector::Constant(3.0 * noise_sigma_);

    // Publish all variants
    publishPointStamped(truth_pub_, true_measurement, current_time);
    publishPointStamped(measurement_pub_, noisy_measurement, current_time);
    publishPointStamped(lower_bound_pub_, lower_bound, current_time);
    publishPointStamped(upper_bound_pub_, upper_bound, current_time);
  }

  const std::string& getName() const {
    return sensor_name_;
  }

  double getPublishRate() const {
    return publish_rate_;
  }

private:
  void updateCovariance() {
    // Create diagonal covariance matrix from sigma
    measurement_covariance_ = MeasurementCovariance::Identity() * (noise_sigma_ * noise_sigma_);
  }

  void publishPointStamped(
      const rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr& publisher,
      const MeasurementVector& point,
      const rclcpp::Time& time) {
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = time;
    msg.header.frame_id = node_->get_parameter("frame_id").as_string();

    // Use tf2_eigen to convert Eigen vector to ROS point
    geometry_msgs::msg::Point ros_point = tf2::toMsg(point);
    msg.point = ros_point;

    publisher->publish(msg);
  }

  rclcpp::Node* node_;
  std::string sensor_name_;
  std::shared_ptr<sensors::PositionSensorModel> sensor_model_;
  double noise_sigma_;
  double publish_rate_;
  MeasurementCovariance measurement_covariance_;
  kinematic_arbiter::utils::Figure8Config trajectory_config_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr truth_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr measurement_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr upper_bound_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr lower_bound_pub_;

  // Random number generator
  std::mt19937 generator_;
};

// Helper functions for conversions
geometry_msgs::msg::Point toPoint(const Eigen::Vector3d& vec) {
  geometry_msgs::msg::Point point;
  point.x = vec.x();
  point.y = vec.y();
  point.z = vec.z();
  return point;
}

geometry_msgs::msg::Quaternion toQuaternion(const Eigen::Quaterniond& quat) {
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.x = quat.x();
  quaternion.y = quat.y();
  quaternion.z = quat.z();
  quaternion.w = quat.w();
  return quaternion;
}

geometry_msgs::msg::Vector3 toVector3(const Eigen::Vector3d& vec) {
  geometry_msgs::msg::Vector3 vector;
  vector.x = vec.x();
  vector.y = vec.y();
  vector.z = vec.z();
  return vector;
}

/**
 * @brief Main node that generates a Figure-8 trajectory and simulates sensors
 */
class Figure8SimulatorNode : public rclcpp::Node {
public:
  using SIdx = core::StateIndex;
  using StateVector = Eigen::Matrix<double, SIdx::kFullStateSize, 1>;

  Figure8SimulatorNode()
    : Node("figure8_simulator") {

    // Add the parent frequency parameter
    this->declare_parameter("parent_frequency", 20.0);  // Default 20Hz
    double requested_parent_freq = this->get_parameter("parent_frequency").as_double();
    parent_frequency_ = std::min(requested_parent_freq, 100.0);  // Cap at 100Hz

    // Other parameters
    this->declare_parameter("trajectory.max_vel", 1.0);
    this->declare_parameter("trajectory.length", 5.0);
    this->declare_parameter("trajectory.width", 3.0);
    this->declare_parameter("trajectory.width_slope", 0.1);
    this->declare_parameter("trajectory.angular_scale", 0.1);

    this->declare_parameter("frame_id", "world");
    this->declare_parameter("base_frame_id", "base_link");
    this->declare_parameter("publish_rate", 30.0);
    this->declare_parameter("sensors", std::vector<std::string>());

    // Get parameters
    auto max_vel = this->get_parameter("trajectory.max_vel").as_double();
    auto length = this->get_parameter("trajectory.length").as_double();
    auto width = this->get_parameter("trajectory.width").as_double();
    auto width_slope = this->get_parameter("trajectory.width_slope").as_double();
    auto angular_scale = this->get_parameter("trajectory.angular_scale").as_double();

    publish_rate_ = this->get_parameter("publish_rate").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();

    // Configure trajectory
    trajectory_config_.max_velocity = max_vel;
    trajectory_config_.length = length;
    trajectory_config_.width = width;
    trajectory_config_.width_slope = width_slope;
    trajectory_config_.angular_scale = angular_scale;

    RCLCPP_INFO(this->get_logger(), "Loaded trajectory: max_vel=%.2f, length=%.2f, width=%.2f",
               max_vel, length, width);
    // Create publishers
    true_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "trajectory/pose", 10);

    true_velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "trajectory/velocity", 10);

    true_accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelStamped>(
      "trajectory/acceleration", 10);

    // Setup transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Set starting time
    elapsed_seconds_ = 0.0;
    start_time_ = this->now(); // Still store this for logging, but don't use for calculations

    // Initialize sensors
    setupSensors();

    // Create parameter callback
    params_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&Figure8SimulatorNode::parametersCallback, this, std::placeholders::_1));

    // Create parent timer using the parent frequency
    double period_ms = 1000.0 / parent_frequency_;
    trajectory_timer_ = this->create_wall_timer(
      std::chrono::duration<double, std::milli>(period_ms),
      std::bind(&Figure8SimulatorNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Figure8 simulator running at %.1f Hz parent frequency",
                parent_frequency_);
  }

private:
  // Validate if a sensor frequency is compatible with parent frequency
  double validateFrequency(double requested_freq) {
    if (requested_freq > parent_frequency_) {
      RCLCPP_WARN(this->get_logger(),
          "Requested frequency %.1f Hz exceeds parent frequency %.1f Hz. Using parent frequency.",
          requested_freq, parent_frequency_);
      return parent_frequency_;
    }

    // Calculate ticks per publication
    double ticks_per_pub = parent_frequency_ / requested_freq;
    int rounded_ticks = std::round(ticks_per_pub);

    // Calculate actual frequency
    double actual_freq = parent_frequency_ / rounded_ticks;

    // Warn if significant adjustment
    if (std::abs(actual_freq - requested_freq) > 0.01) {
      RCLCPP_WARN(this->get_logger(),
          "Adjusted sensor frequency from %.1f Hz to %.1f Hz to match parent frequency",
          requested_freq, actual_freq);
    }

    return actual_freq;
  }

  // New combined timer callback that handles everything
  void timerCallback() {
    // Update elapsed time
    elapsed_seconds_ += 1.0 / parent_frequency_;

    // Publish trajectory at requested rate
    publishTrajectory();

    // Check which sensors should publish this tick
    for (auto& [name, sensor] : position_sensors_) {
      double sensor_rate = sensor->getPublishRate();
      int ticks_per_pub = std::round(parent_frequency_ / sensor_rate);

      // Check if this sensor should publish based on frame count
      if (tick_counter_ % ticks_per_pub == 0) {
        sensor->generateAndPublishMeasurements(elapsed_seconds_);
      }
    }

    // Increment tick counter
    tick_counter_++;
  }

  void setupSensors() {
    auto sensors = this->get_parameter("sensors").as_string_array();

    // Setup sensors based on parameters
    for (const auto& name : sensors) {
      // Declare parameters for this sensor if not already declared
      declarePositionSensorParams(name);

      // Get parameters
      auto sensor_base = "sensors." + name;
      auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
      auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
      double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
      double requested_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

      // Validate and adjust sensor rate to be compatible with parent frequency
      double actual_rate = validateFrequency(requested_rate);

      // Create transform
      Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
      transform.translation() = Eigen::Vector3d(
          position_param[0], position_param[1], position_param[2]);

      Eigen::Quaterniond q(
          quaternion_param[0],  // w
          quaternion_param[1],  // x
          quaternion_param[2],  // y
          quaternion_param[3]); // z
      q.normalize();
      transform.linear() = q.toRotationMatrix();

      // Create the sensor
      position_sensors_[name] = std::make_unique<PositionSensorSimulator>(
          this, name, transform, noise_sigma, actual_rate, trajectory_config_);

      RCLCPP_INFO(this->get_logger(),
                 "Created position sensor: %s (%.1f Hz, sigma=%.3f)",
                 name.c_str(), actual_rate, noise_sigma);
    }
  }

  void declarePositionSensorParams(const std::string& name) {
    auto prefix = "sensors." + name;

    // Skip if already declared
    if (this->has_parameter(prefix + ".position")) {
      return;
    }

    // Position (x, y, z)
    this->declare_parameter(prefix + ".position", std::vector<double>{0.0, 0.0, 0.0});

    // Orientation (w, x, y, z)
    this->declare_parameter(prefix + ".quaternion", std::vector<double>{1.0, 0.0, 0.0, 0.0});

    // Noise and rate
    this->declare_parameter(prefix + ".noise_sigma", 0.1);
    this->declare_parameter(prefix + ".publish_rate", 10.0);
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
      const std::vector<rclcpp::Parameter>& parameters) {

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    bool update_trajectory = false;
    bool update_sensors = false;
    std::set<std::string> updated_sensors;

    for (const auto& param : parameters) {
      const auto& name = param.get_name();

      // Check for trajectory parameter updates
      if (name.find("trajectory.") == 0) {
        update_trajectory = true;
      }

      // Check for sensor parameter updates
      else if (name.find("sensors.") == 0) {
        // Format: sensors.<name>.<param>
        auto parts = splitString(name, '.');
        if (parts.size() >= 2) {
          updated_sensors.insert(parts[1]);
          update_sensors = true;
        }
      }

      // Check for sensor list update
      else if (name == "sensors") {
        update_sensors = true;
      }

      // Parent frequency update
      else if (name == "parent_frequency") {
        double requested_freq = param.as_double();
        double new_freq = std::min(requested_freq, 100.0); // Cap at 100Hz

        if (parent_frequency_ != new_freq) {
          parent_frequency_ = new_freq;

          // Recreate the timer with new frequency
          double period_ms = 1000.0 / parent_frequency_;
          trajectory_timer_ = this->create_wall_timer(
            std::chrono::duration<double, std::milli>(period_ms),
            std::bind(&Figure8SimulatorNode::timerCallback, this));

          RCLCPP_INFO(this->get_logger(), "Updated parent frequency to %.1f Hz", parent_frequency_);

          // Need to update all sensors to validate their frequencies
          update_sensors = true;
          auto sensors = this->get_parameter("sensors").as_string_array();
          for (const auto& name : sensors) {
            updated_sensors.insert(name);
          }
        }
      }
    }

    // Update trajectory if needed
    if (update_trajectory) {
      auto max_vel = this->get_parameter("trajectory.max_vel").as_double();
      auto length = this->get_parameter("trajectory.length").as_double();
      auto width = this->get_parameter("trajectory.width").as_double();
      auto width_slope = this->get_parameter("trajectory.width_slope").as_double();
      auto angular_scale = this->get_parameter("trajectory.angular_scale").as_double();

      trajectory_config_.max_velocity = max_vel;
      trajectory_config_.length = length;
      trajectory_config_.width = width;
      trajectory_config_.width_slope = width_slope;
      trajectory_config_.angular_scale = angular_scale;

      RCLCPP_INFO(this->get_logger(), "Updated trajectory: max_vel=%.2f, length=%.2f, width=%.2f",
                 max_vel, length, width);

      // Update trajectory config in all sensors
      for (auto& [name, sensor] : position_sensors_) {
        sensor->updateTrajectoryConfig(trajectory_config_);
      }
    }

    // Update sensors if needed
    if (update_sensors) {
      updateSensors(updated_sensors);
    }

    return result;
  }

  void updateSensors(const std::set<std::string>& specific_sensors = {}) {
    // Get current sensor list
    auto current_sensors = this->get_parameter("sensors").as_string_array();
    std::set<std::string> current_set(current_sensors.begin(), current_sensors.end());

    // Find sensors to remove (in position_sensors_ but not in current_sensors)
    std::vector<std::string> to_remove;
    for (const auto& [name, _] : position_sensors_) {
      if (current_set.find(name) == current_set.end()) {
        to_remove.push_back(name);
      }
    }

    // Remove sensors
    for (const auto& name : to_remove) {
      position_sensors_.erase(name);
      RCLCPP_INFO(this->get_logger(), "Removed sensor: %s", name.c_str());
    }

    // Find new sensors to add
    std::vector<std::string> new_sensors;
    for (const auto& name : current_sensors) {
      if (position_sensors_.find(name) == position_sensors_.end()) {
        new_sensors.push_back(name);
      }
    }

    // Update existing sensors if specifically requested
    if (!specific_sensors.empty()) {
      for (const auto& name : specific_sensors) {
        auto it = position_sensors_.find(name);
        if (it != position_sensors_.end()) {
          updateSensorFromParams(name, it->second.get());
        }
      }
    }

    // Add new sensors
    for (const auto& name : new_sensors) {
      if (current_set.find(name) != current_set.end()) {
        // Declare parameters for this sensor
        declarePositionSensorParams(name);

        // Create the sensor
        auto sensor_base = "sensors." + name;
        auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
        auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
        double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
        double requested_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

        // Validate and adjust sensor rate
        double actual_rate = validateFrequency(requested_rate);

        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.translation() = Eigen::Vector3d(
            position_param[0], position_param[1], position_param[2]);

        Eigen::Quaterniond q(
            quaternion_param[0],  // w
            quaternion_param[1],  // x
            quaternion_param[2],  // y
            quaternion_param[3]); // z
        q.normalize();
        transform.linear() = q.toRotationMatrix();

        // Create and store the sensor
        position_sensors_[name] = std::make_unique<PositionSensorSimulator>(
            this, name, transform, noise_sigma, actual_rate, trajectory_config_);

        RCLCPP_INFO(this->get_logger(),
                   "Created position sensor: %s (%.1f Hz, sigma=%.3f)",
                   name.c_str(), actual_rate, noise_sigma);
      }
    }
  }

  void updateSensorFromParams(const std::string& name, PositionSensorSimulator* sensor) {
    auto sensor_base = "sensors." + name;

    // Get updated parameters
    auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
    auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
    double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
    double requested_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

    // Validate and adjust sensor rate
    double actual_rate = validateFrequency(requested_rate);

    // Create transform
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(
        position_param[0], position_param[1], position_param[2]);

    Eigen::Quaterniond q(
        quaternion_param[0],  // w
        quaternion_param[1],  // x
        quaternion_param[2],  // y
        quaternion_param[3]); // z
    q.normalize();
    transform.linear() = q.toRotationMatrix();

    // Update sensor configuration
    sensor->updateConfiguration(transform, noise_sigma, actual_rate);

    RCLCPP_INFO(this->get_logger(), "Updated sensor: %s (%.1f Hz, sigma=%.3f)",
               name.c_str(), actual_rate, noise_sigma);
  }

  void publishTrajectory() {
    // Get current time (for message headers)
    auto current_time = this->now();

    // Generate trajectory state at current elapsed time
    StateVector state = kinematic_arbiter::utils::Figure8Trajectory(
        elapsed_seconds_, trajectory_config_);

    // Extract components for convenience
    Eigen::Vector3d position = state.segment<3>(SIdx::Position::Begin());
    Eigen::Quaterniond orientation(
        state[SIdx::Quaternion::W],
        state[SIdx::Quaternion::X],
        state[SIdx::Quaternion::Y],
        state[SIdx::Quaternion::Z]);
    Eigen::Vector3d lin_velocity = state.segment<3>(SIdx::LinearVelocity::Begin());
    Eigen::Vector3d ang_velocity = state.segment<3>(SIdx::AngularVelocity::Begin());
    Eigen::Vector3d lin_accel = state.segment<3>(SIdx::LinearAcceleration::Begin());
    Eigen::Vector3d ang_accel = state.segment<3>(SIdx::AngularAcceleration::Begin());

    // Publish ground truth pose
    auto pose_msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
    pose_msg->header.stamp = current_time;
    pose_msg->header.frame_id = frame_id_;
    pose_msg->pose.position = toPoint(position);
    pose_msg->pose.orientation = toQuaternion(orientation);
    true_pose_pub_->publish(std::move(pose_msg));

    // Publish ground truth velocity
    auto vel_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    vel_msg->header.stamp = current_time;
    vel_msg->header.frame_id = frame_id_;
    vel_msg->twist.linear = toVector3(lin_velocity);
    vel_msg->twist.angular = toVector3(ang_velocity);
    true_velocity_pub_->publish(std::move(vel_msg));

    // Publish ground truth acceleration
    auto accel_msg = std::make_unique<geometry_msgs::msg::AccelStamped>();
    accel_msg->header.stamp = current_time;
    accel_msg->header.frame_id = frame_id_;
    accel_msg->accel.linear = toVector3(lin_accel);
    accel_msg->accel.angular = toVector3(ang_accel);
    true_accel_pub_->publish(std::move(accel_msg));

    // Publish transform
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = current_time;
    transform.header.frame_id = frame_id_;
    transform.child_frame_id = base_frame_id_;
    transform.transform.translation.x = position.x();
    transform.transform.translation.y = position.y();
    transform.transform.translation.z = position.z();
    transform.transform.rotation = toQuaternion(orientation);
    tf_broadcaster_->sendTransform(transform);
  }

  std::vector<std::string> splitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr true_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr true_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr true_accel_pub_;

  // TF broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Timer
  rclcpp::TimerBase::SharedPtr trajectory_timer_;

  // Parameter callback handle
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;

  // Position sensors
  std::map<std::string, std::unique_ptr<PositionSensorSimulator>> position_sensors_;

  // Trajectory configuration
  kinematic_arbiter::utils::Figure8Config trajectory_config_;

  // Other parameters
  double publish_rate_;
  std::string frame_id_;
  std::string base_frame_id_;

  // Parent frequency
  double parent_frequency_ = 20.0;
  int tick_counter_ = 0;

  // Timing - keep start_time_ for logging but don't use for calculations
  rclcpp::Time start_time_;
  double elapsed_seconds_ = 0.0;

  // Random generator
  std::mt19937 generator_;
};

}  // namespace simulation
}  // namespace ros2
}  // namespace kinematic_arbiter

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<kinematic_arbiter::ros2::simulation::Figure8SimulatorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
