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
      const rclcpp::Time& start_time,
      const kinematic_arbiter::utils::Figure8Config& trajectory_config)
    : node_(node),
      sensor_name_(sensor_name),
      sensor_model_(std::make_shared<sensors::PositionSensorModel>(transform)),
      noise_sigma_(noise_sigma),
      publish_rate_(publish_rate),
      start_time_(start_time),
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

    // Create timer with sensor-specific publish rate
    timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
        std::bind(&PositionSensorSimulator::generateAndPublishMeasurements, this));
  }

  void updateConfiguration(
      const Eigen::Isometry3d& transform,
      double noise_sigma,
      double publish_rate) {
    sensor_model_ = std::make_shared<sensors::PositionSensorModel>(transform);
    noise_sigma_ = noise_sigma;
    updateCovariance();

    // Update publish rate if changed
    if (publish_rate_ != publish_rate) {
      publish_rate_ = publish_rate;
      timer_ = node_->create_wall_timer(
          std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
          std::bind(&PositionSensorSimulator::generateAndPublishMeasurements, this));
    }
  }

  void updateTrajectoryConfig(const kinematic_arbiter::utils::Figure8Config& config) {
    trajectory_config_ = config;
  }

  void updateStartTime(const rclcpp::Time& start_time) {
    start_time_ = start_time;
  }

  // Generate and publish measurements based on sensor's own timer
  void generateAndPublishMeasurements() {
    auto current_time = node_->now();
    double elapsed_seconds = (current_time - start_time_).seconds();

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
  rclcpp::Time start_time_;
  kinematic_arbiter::utils::Figure8Config trajectory_config_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr truth_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr measurement_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr upper_bound_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr lower_bound_pub_;

  // Timer for this sensor's publishing
  rclcpp::TimerBase::SharedPtr timer_;

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

  Figure8SimulatorNode() : Node("figure8_simulator"), generator_(std::random_device{}()) {
    // Declare base parameters
    this->declare_parameter("publish_rate", 50.0);
    this->declare_parameter("frame_id", "odom");
    this->declare_parameter("base_frame_id", "base_link");

    // Trajectory parameters
    this->declare_parameter("trajectory.max_velocity", 1.0);
    this->declare_parameter("trajectory.length", 5.0);
    this->declare_parameter("trajectory.width", 2.5);
    this->declare_parameter("trajectory.width_slope", 0.1);
    this->declare_parameter("trajectory.angular_scale", 0.5);

    // Position sensors parameter (list of sensor names)
    this->declare_parameter("position_sensors", std::vector<std::string>{"position1"});

    // Load parameters
    loadParameters();

    // Create trajectory publishers
    true_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "ground_truth/pose", 10);
    true_velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        "ground_truth/velocity", 10);
    true_accel_pub_ = this->create_publisher<geometry_msgs::msg::AccelStamped>(
        "ground_truth/acceleration", 10);

    // Create TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Set up parameter callback
    params_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&Figure8SimulatorNode::parametersCallback, this, std::placeholders::_1));

    // Create sensors based on parameter list
    createSensorsFromParams();

    // Record start time
    start_time_ = this->now();

    // Create timer for trajectory updates
    trajectory_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
        std::bind(&Figure8SimulatorNode::publishTrajectory, this));
  }

private:
  void loadParameters() {
    // Get base parameters
    publish_rate_ = this->get_parameter("publish_rate").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();

    // Load trajectory configuration
    trajectory_config_.max_velocity = this->get_parameter("trajectory.max_velocity").as_double();
    trajectory_config_.length = this->get_parameter("trajectory.length").as_double();
    trajectory_config_.width = this->get_parameter("trajectory.width").as_double();
    trajectory_config_.width_slope = this->get_parameter("trajectory.width_slope").as_double();
    trajectory_config_.angular_scale = this->get_parameter("trajectory.angular_scale").as_double();

    RCLCPP_INFO(this->get_logger(),
                "Loaded trajectory: max_vel=%.2f, length=%.2f, width=%.2f",
                trajectory_config_.max_velocity,
                trajectory_config_.length,
                trajectory_config_.width);
  }

  void createSensorsFromParams() {
    // Get list of position sensors
    auto sensor_names = this->get_parameter("position_sensors").as_string_array();

    for (const auto& name : sensor_names) {
      // Declare parameters for this sensor if they don't exist
      declarePositionSensorParams(name);

      // Create the sensor
      auto sensor_base = "sensors." + name;
      auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
      auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
      double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
      double sensor_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

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
          this, name, transform, noise_sigma, sensor_rate, start_time_, trajectory_config_);

      RCLCPP_INFO(this->get_logger(),
                  "Created position sensor: %s (%.1f Hz, sigma=%.3f)",
                  name.c_str(), sensor_rate, noise_sigma);
    }
  }

  void declarePositionSensorParams(const std::string& name) {
    auto sensor_base = "sensors." + name;

    // Default sensor positioned at the origin
    this->declare_parameter(sensor_base + ".position", std::vector<double>{0.0, 0.0, 0.0});

    // Default identity quaternion (w, x, y, z)
    this->declare_parameter(sensor_base + ".quaternion", std::vector<double>{1.0, 0.0, 0.0, 0.0});

    // Default noise sigma
    this->declare_parameter(sensor_base + ".noise_sigma", 0.05);

    // Default publish rate (10 Hz)
    this->declare_parameter(sensor_base + ".publish_rate", 10.0);
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
      const std::vector<rclcpp::Parameter>& parameters) {

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "Success";

    bool update_trajectory_config = false;
    bool update_trajectory_rate = false;
    std::set<std::string> sensors_to_update;

    for (const auto& param : parameters) {
      const std::string& name = param.get_name();

      // Handle trajectory parameters
      if (name == "trajectory.max_velocity") {
        trajectory_config_.max_velocity = param.as_double();
        update_trajectory_config = true;
      } else if (name == "trajectory.length") {
        trajectory_config_.length = param.as_double();
        update_trajectory_config = true;
      } else if (name == "trajectory.width") {
        trajectory_config_.width = param.as_double();
        update_trajectory_config = true;
      } else if (name == "trajectory.width_slope") {
        trajectory_config_.width_slope = param.as_double();
        update_trajectory_config = true;
      } else if (name == "trajectory.angular_scale") {
        trajectory_config_.angular_scale = param.as_double();
        update_trajectory_config = true;
      } else if (name == "publish_rate") {
        publish_rate_ = param.as_double();
        update_trajectory_rate = true;
      } else if (name == "frame_id") {
        frame_id_ = param.as_string();
      } else if (name == "base_frame_id") {
        base_frame_id_ = param.as_string();
      } else if (name == "position_sensors") {
        // Handle changes to the sensor list
        auto new_sensor_names = param.as_string_array();
        updateSensorList(new_sensor_names);
      } else if (name.find("sensors.") == 0) {
        // Extract sensor name from parameter
        size_t first_dot = name.find('.');
        size_t second_dot = name.find('.', first_dot + 1);
        if (second_dot != std::string::npos) {
          std::string sensor_name = name.substr(first_dot + 1, second_dot - first_dot - 1);
          sensors_to_update.insert(sensor_name);
        }
      }
    }

    // Update trajectory configuration for all sensors if changed
    if (update_trajectory_config) {
      for (auto& [name, sensor] : position_sensors_) {
        sensor->updateTrajectoryConfig(trajectory_config_);
      }

      RCLCPP_INFO(this->get_logger(),
                "Updated trajectory: max_vel=%.2f, length=%.2f, width=%.2f",
                trajectory_config_.max_velocity,
                trajectory_config_.length,
                trajectory_config_.width);
    }

    // Update trajectory timer if rate changed
    if (update_trajectory_rate) {
      trajectory_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
          std::bind(&Figure8SimulatorNode::publishTrajectory, this));

      RCLCPP_INFO(this->get_logger(), "Updated publish rate: %.2f Hz", publish_rate_);
    }

    // Update any sensors whose parameters have changed
    for (const auto& sensor_name : sensors_to_update) {
      auto it = position_sensors_.find(sensor_name);
      if (it != position_sensors_.end()) {
        updateSensorFromParams(sensor_name, it->second.get());
      }
    }

    return result;
  }

  void updateSensorList(const std::vector<std::string>& new_sensor_names) {
    // Find sensors to add and remove
    std::set<std::string> current_sensors;
    for (const auto& [name, _] : position_sensors_) {
      current_sensors.insert(name);
    }

    std::set<std::string> new_sensors(new_sensor_names.begin(), new_sensor_names.end());

    // Remove sensors that are no longer in the list
    for (const auto& name : current_sensors) {
      if (new_sensors.find(name) == new_sensors.end()) {
        position_sensors_.erase(name);
        RCLCPP_INFO(this->get_logger(), "Removed sensor: %s", name.c_str());
      }
    }

    // Add new sensors
    for (const auto& name : new_sensors) {
      if (current_sensors.find(name) == current_sensors.end()) {
        // Declare parameters for this sensor
        declarePositionSensorParams(name);

        // Create the sensor
        auto sensor_base = "sensors." + name;
        auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
        auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
        double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
        double sensor_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

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
            this, name, transform, noise_sigma, sensor_rate, start_time_, trajectory_config_);

        RCLCPP_INFO(this->get_logger(),
                   "Created position sensor: %s (%.1f Hz, sigma=%.3f)",
                   name.c_str(), sensor_rate, noise_sigma);
      }
    }
  }

  void updateSensorFromParams(const std::string& name, PositionSensorSimulator* sensor) {
    auto sensor_base = "sensors." + name;

    // Get updated parameters
    auto position_param = this->get_parameter(sensor_base + ".position").as_double_array();
    auto quaternion_param = this->get_parameter(sensor_base + ".quaternion").as_double_array();
    double noise_sigma = this->get_parameter(sensor_base + ".noise_sigma").as_double();
    double sensor_rate = this->get_parameter(sensor_base + ".publish_rate").as_double();

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
    sensor->updateConfiguration(transform, noise_sigma, sensor_rate);

    RCLCPP_INFO(this->get_logger(), "Updated sensor: %s (%.1f Hz, sigma=%.3f)",
               name.c_str(), sensor_rate, noise_sigma);
  }

  void publishTrajectory() {
    // Get current time
    auto current_time = this->now();
    double elapsed_seconds = (current_time - start_time_).seconds();

    // Generate trajectory state at current time
    StateVector state = kinematic_arbiter::utils::Figure8Trajectory(
        elapsed_seconds, trajectory_config_);

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

  // Timing
  rclcpp::Time start_time_;

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
