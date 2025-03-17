#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <cmath>
#include <vector>

namespace kinematic_arbiter {
namespace ros2 {
namespace simulation {

class Figure8SimulatorNode : public rclcpp::Node {
public:
  Figure8SimulatorNode() : Node("figure8_simulator") {
    // Declare parameters
    this->declare_parameter("publish_rate", 50.0);
    this->declare_parameter("figure8_scale_x", 5.0);
    this->declare_parameter("figure8_scale_y", 2.5);
    this->declare_parameter("period", 20.0);  // seconds for one complete figure-8
    this->declare_parameter("sensor_noise_stddev", 0.05);  // meters
    this->declare_parameter("frame_id", "odom");
    this->declare_parameter("base_frame_id", "base_link");

    // Parameters for sensor positions (offsets from base_link)
    this->declare_parameter("sensor1_position", std::vector<double>{1.0, 0.5, 0.2});  // front-right
    this->declare_parameter("sensor2_position", std::vector<double>{1.0, -0.5, 0.2}); // front-left

    // Get parameters
    publish_rate_ = this->get_parameter("publish_rate").as_double();
    scale_x_ = this->get_parameter("figure8_scale_x").as_double();
    scale_y_ = this->get_parameter("figure8_scale_y").as_double();
    period_ = this->get_parameter("period").as_double();
    sensor_noise_stddev_ = this->get_parameter("sensor_noise_stddev").as_double();
    frame_id_ = this->get_parameter("frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();

    sensor1_position_ = this->get_parameter("sensor1_position").as_double_array();
    sensor2_position_ = this->get_parameter("sensor2_position").as_double_array();

    // Create publishers
    true_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "ground_truth/pose", 10);
    true_velocity_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        "ground_truth/velocity", 10);

    // Create simulated sensor publishers
    position_sensor1_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
        "sensors/position1", 10);
    position_sensor2_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
        "sensors/position2", 10);

    // Create transform broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Timer for publishing trajectory and simulated measurements
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
        std::bind(&Figure8SimulatorNode::publishTrajectory, this));

    start_time_ = this->now();
    RCLCPP_INFO(this->get_logger(), "Figure 8 simulator started");
  }

private:
  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr true_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr true_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr position_sensor1_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr position_sensor2_pub_;

  // TF broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters
  double publish_rate_;
  double scale_x_;
  double scale_y_;
  double period_;
  double sensor_noise_stddev_;
  std::string frame_id_;
  std::string base_frame_id_;

  // Sensor positions (in base_link frame)
  std::vector<double> sensor1_position_;
  std::vector<double> sensor2_position_;

  // State
  rclcpp::Time start_time_;

  void publishTrajectory() {
    auto current_time = this->now();
    double elapsed = (current_time - start_time_).seconds();

    // Compute figure-8 position at current time
    double phase = 2.0 * M_PI * elapsed / period_;
    double x = scale_x_ * std::sin(phase);
    double y = scale_y_ * std::sin(2.0 * phase);
    double z = 0.0;

    // Compute velocity
    double dx = scale_x_ * std::cos(phase) * 2.0 * M_PI / period_;
    double dy = scale_y_ * std::cos(2.0 * phase) * 4.0 * M_PI / period_;
    double dz = 0.0;

    // Compute orientation (tangent to the curve)
    double yaw = std::atan2(dy, dx);

    // Create and publish ground truth pose
    auto pose_msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
    pose_msg->header.stamp = current_time;
    pose_msg->header.frame_id = frame_id_;
    pose_msg->pose.position.x = x;
    pose_msg->pose.position.y = y;
    pose_msg->pose.position.z = z;

    // Convert yaw to quaternion
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    pose_msg->pose.orientation.w = cy;
    pose_msg->pose.orientation.x = 0.0;
    pose_msg->pose.orientation.y = 0.0;
    pose_msg->pose.orientation.z = sy;

    true_pose_pub_->publish(std::move(pose_msg));

    // Create and publish ground truth velocity
    auto vel_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    vel_msg->header.stamp = current_time;
    vel_msg->header.frame_id = frame_id_;
    vel_msg->twist.linear.x = dx;
    vel_msg->twist.linear.y = dy;
    vel_msg->twist.linear.z = dz;

    // Angular velocity around z-axis (needed for turning)
    double angular_z = (dx * -dy + dy * dx) / (dx * dx + dy * dy);
    vel_msg->twist.angular.x = 0.0;
    vel_msg->twist.angular.y = 0.0;
    vel_msg->twist.angular.z = angular_z;

    true_velocity_pub_->publish(std::move(vel_msg));

    // Publish TF from odom to base_link
    publishTransform(x, y, z, yaw, frame_id_, base_frame_id_, current_time);

    // Publish TF from base_link to sensor frames
    publishSensorTransforms(current_time);

    // Create and publish simulated noisy position measurements
    publishNoisyPositionMeasurement(position_sensor1_pub_,
        "sensor1_frame", current_time);
    publishNoisyPositionMeasurement(position_sensor2_pub_,
        "sensor2_frame", current_time);
  }

  void publishTransform(double x, double y, double z, double yaw,
                         const std::string& parent_frame,
                         const std::string& child_frame,
                         const rclcpp::Time& time) {
    geometry_msgs::msg::TransformStamped transform;

    transform.header.stamp = time;
    transform.header.frame_id = parent_frame;
    transform.child_frame_id = child_frame;

    transform.transform.translation.x = x;
    transform.transform.translation.y = y;
    transform.transform.translation.z = z;

    // Convert yaw to quaternion
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    transform.transform.rotation.w = cy;
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = sy;

    tf_broadcaster_->sendTransform(transform);
  }

  void publishSensorTransforms(const rclcpp::Time& time) {
    // Sensor 1
    geometry_msgs::msg::TransformStamped sensor1_transform;
    sensor1_transform.header.stamp = time;
    sensor1_transform.header.frame_id = base_frame_id_;
    sensor1_transform.child_frame_id = "sensor1_frame";

    sensor1_transform.transform.translation.x = sensor1_position_[0];
    sensor1_transform.transform.translation.y = sensor1_position_[1];
    sensor1_transform.transform.translation.z = sensor1_position_[2];

    // Identity rotation
    sensor1_transform.transform.rotation.w = 1.0;
    sensor1_transform.transform.rotation.x = 0.0;
    sensor1_transform.transform.rotation.y = 0.0;
    sensor1_transform.transform.rotation.z = 0.0;

    tf_broadcaster_->sendTransform(sensor1_transform);

    // Sensor 2
    geometry_msgs::msg::TransformStamped sensor2_transform;
    sensor2_transform.header.stamp = time;
    sensor2_transform.header.frame_id = base_frame_id_;
    sensor2_transform.child_frame_id = "sensor2_frame";

    sensor2_transform.transform.translation.x = sensor2_position_[0];
    sensor2_transform.transform.translation.y = sensor2_position_[1];
    sensor2_transform.transform.translation.z = sensor2_position_[2];

    // Identity rotation
    sensor2_transform.transform.rotation.w = 1.0;
    sensor2_transform.transform.rotation.x = 0.0;
    sensor2_transform.transform.rotation.y = 0.0;
    sensor2_transform.transform.rotation.z = 0.0;

    tf_broadcaster_->sendTransform(sensor2_transform);
  }

  void publishNoisyPositionMeasurement(
      const rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr& publisher,
      const std::string& sensor_frame,
      const rclcpp::Time& time) {

    // Add Gaussian noise to the position
    auto msg = std::make_unique<geometry_msgs::msg::PointStamped>();
    msg->header.stamp = time;
    msg->header.frame_id = sensor_frame;

    // Since we're in the sensor frame, the position is (0,0,0) with noise
    double noise_x = ((double)rand() / RAND_MAX - 0.5) * 2.0 * sensor_noise_stddev_;
    double noise_y = ((double)rand() / RAND_MAX - 0.5) * 2.0 * sensor_noise_stddev_;
    double noise_z = ((double)rand() / RAND_MAX - 0.5) * 2.0 * sensor_noise_stddev_;

    msg->point.x = noise_x;
    msg->point.y = noise_y;
    msg->point.z = noise_z;

    publisher->publish(std::move(msg));
  }
};

} // namespace simulation
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<kinematic_arbiter::ros2::simulation::Figure8SimulatorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
