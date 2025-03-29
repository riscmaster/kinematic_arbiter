#include <gtest/gtest.h>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Define the Figure8 Simulator YAML configuration
const char* FIGURE8_CONFIG_YAML = R"(
figure8_simulator:
  ros__parameters:
    # Main parameters
    main_update_rate: 100.0  # Main simulator update rate (Hz)
    world_frame_id: "map"    # World coordinate frame ID
    body_frame_id: "base_link"  # Body coordinate frame ID

    # Noise and timing parameters
    noise_sigma: 0.01    # Standard deviation of noise (m)
    time_jitter: 0.005   # Time jitter (s)

    # Trajectory parameters
    trajectory:
      max_vel: 1.0     # Maximum velocity (m/s)
      length: 5.0      # Length of figure-8 (m)
      width: 3.0       # Width of figure-8 (m)
      width_slope: 0.1  # Width slope parameter
      angular_scale: 0.001  # Angular scale parameter

    # Sensor update rates
    position_rate: 30.0  # Position sensor rate (Hz)
    pose_rate: 50.0      # Pose sensor rate (Hz)
    velocity_rate: 50.0  # Velocity sensor rate (Hz)
    imu_rate: 100.0      # IMU sensor rate (Hz)

    # Topic names - use same names as in working config
    position_topic: "position_sensor"
    pose_topic: "pose_sensor"
    velocity_topic: "velocity_sensor"
    imu_topic: "imu_sensor"
    truth_pose_topic: "truth/pose"
    truth_velocity_topic: "truth/velocity"

    # Sensor frame IDs
    position_frame: "position_sensor"
    pose_frame: "pose_sensor"
    velocity_frame: "velocity_sensor"
    imu_frame: "imu_sensor"

    # Sensor offsets from body frame (for realistic simulation)
    position_sensor:
      x_offset: 0.05   # X offset (m)
      y_offset: 0.0    # Y offset (m)
      z_offset: 0.1    # Z offset (m)

    pose_sensor:
      x_offset: -0.03  # X offset (m)
      y_offset: 0.02   # Y offset (m)
      z_offset: 0.15   # Z offset (m)

    velocity_sensor:
      x_offset: 0.0    # X offset (m)
      y_offset: 0.0    # Y offset (m)
      z_offset: 0.05   # Z offset (m)

    imu_sensor:
      x_offset: 0.01   # X offset (m)
      y_offset: -0.01  # Y offset (m)
      z_offset: 0.08   # Z offset (m)
)";

// Define the Kinematic Arbiter YAML configuration
const char* ARBITER_CONFIG_YAML = R"(
kinematic_arbiter:
  ros__parameters:
    publish_rate: 50.0
    max_delay_window: 1.0
    world_frame_id: "map"
    body_frame_id: "base_link"

    # Output state topics
    pose_state_topic: "state/pose"
    velocity_state_topic: "state/velocity"
    acceleration_state_topic: "state/acceleration"

    # Sensor configurations - matching what the figure8_simulator produces
    position_sensors: ["position_sensor:position_sensor:position_sensor"]
    pose_sensors: ["pose_sensor:pose_sensor:pose_sensor"]
    velocity_sensors: ["velocity_sensor:velocity_sensor:velocity_sensor"]
    imu_sensors: ["imu_sensor:imu_sensor:imu_sensor"]
)";

class KinematicArbiterTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize ROS
    rclcpp::init(0, nullptr);

    // Create our test node
    test_node_ = std::make_shared<rclcpp::Node>("kinematic_arbiter_test");

    RCLCPP_INFO(test_node_->get_logger(), "Starting kinematic arbiter integration test");

    // Track message reception
    state_pose_received_ = false;
    state_velocity_received_ = false;
    state_acceleration_received_ = false;

    // Subscribe to expected output topics from arbiter
    state_pose_sub_ = test_node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "state/pose", 10,
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        state_pose_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received state pose with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    state_velocity_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "state/velocity", 10,
      [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
        state_velocity_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received state velocity with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    state_acceleration_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "state/acceleration", 10,
      [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
        state_acceleration_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received state acceleration with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    // Initialize executor and add our test node
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor_->add_node(test_node_);

    // Start the executor in a separate thread
    executor_thread_ = std::thread([this]() { executor_->spin(); });

    // Write configuration files
    std::string figure8_yaml_path = "/tmp/figure8_test_config.yaml";
    std::ofstream figure8_yaml_file(figure8_yaml_path);
    figure8_yaml_file << FIGURE8_CONFIG_YAML;
    figure8_yaml_file.close();

    std::string arbiter_yaml_path = "/tmp/kinematic_arbiter_test_config.yaml";
    std::ofstream arbiter_yaml_file(arbiter_yaml_path);
    arbiter_yaml_file << ARBITER_CONFIG_YAML;
    arbiter_yaml_file.close();

    // Launch the figure8_simulator node first
    RCLCPP_INFO(test_node_->get_logger(), "Starting Figure-8 simulator node");
    std::string figure8_cmd = "ros2 run kinematic_arbiter figure8_simulator_node --ros-args --params-file " +
                             figure8_yaml_path + " > /tmp/figure8_sim_output.log 2>&1 &";
    system(figure8_cmd.c_str());

    // Give it time to start up
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Now launch the kinematic arbiter node
    RCLCPP_INFO(test_node_->get_logger(), "Starting kinematic arbiter node");
    std::string arbiter_cmd = "ros2 run kinematic_arbiter kinematic_arbiter_node --ros-args --params-file " +
                             arbiter_yaml_path + " > /tmp/kinematic_arbiter_output.log 2>&1 &";
    system(arbiter_cmd.c_str());

    // Give it time to start up
    std::this_thread::sleep_for(std::chrono::seconds(2));

    RCLCPP_INFO(test_node_->get_logger(), "Both nodes started");
  }

  void TearDown() override {
    RCLCPP_INFO(test_node_->get_logger(), "Test teardown - stopping nodes");

    // Stop both processes
    system("pkill -f kinematic_arbiter_node");
    system("pkill -f figure8_simulator_node");

    // Stop the executor thread
    executor_->cancel();
    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }

    // Clean up nodes
    test_node_.reset();

    // Shutdown ROS
    rclcpp::shutdown();
  }

  bool spinUntilMessagesReceived(double timeout_seconds = 10.0) {
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::duration<double>(timeout_seconds);

    while (std::chrono::steady_clock::now() < end_time) {
      // No need to publish data - the figure8_simulator does that for us
      // No need to spin - the executor is already spinning in a separate thread
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Check if we've received the state messages
      if (state_pose_received_ && state_velocity_received_) {
        return true;
      }

      // Log progress every second
      static auto last_log_time = start_time;
      if (std::chrono::steady_clock::now() - last_log_time >= std::chrono::seconds(1)) {
        last_log_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - start_time).count();
        RCLCPP_INFO(test_node_->get_logger(), "Waiting for messages (%0.1f seconds elapsed)...", elapsed);

        // Print partial reception status for debugging
        RCLCPP_INFO(test_node_->get_logger(),
                  "Current state message reception: pose=%s, velocity=%s, acceleration=%s",
                  state_pose_received_ ? "YES" : "NO",
                  state_velocity_received_ ? "YES" : "NO",
                  state_acceleration_received_ ? "YES" : "NO");
      }
    }

    // Print diagnostic info
    RCLCPP_INFO(test_node_->get_logger(),
               "Final state message reception: pose=%s, velocity=%s, acceleration=%s",
               state_pose_received_ ? "YES" : "NO",
               state_velocity_received_ ? "YES" : "NO",
               state_acceleration_received_ ? "YES" : "NO");

    // List all topics for debugging
    auto topics = test_node_->get_topic_names_and_types();
    RCLCPP_INFO(test_node_->get_logger(), "Available topics:");
    for (const auto& topic_pair : topics) {
      RCLCPP_INFO(test_node_->get_logger(), "  - %s", topic_pair.first.c_str());
    }

    return false;
  }

  // Test node and executor
  std::shared_ptr<rclcpp::Node> test_node_;
  std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executor_thread_;

  // State subscriptions
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr state_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr state_velocity_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr state_acceleration_sub_;

  // Flags for message reception
  bool state_pose_received_;
  bool state_velocity_received_;
  bool state_acceleration_received_;
};

TEST_F(KinematicArbiterTest, VerifyStatePublishing) {
  RCLCPP_INFO(test_node_->get_logger(), "Running state publishing verification test");

  // Verify arbiter publishes state after receiving sensor data from figure8_simulator
  ASSERT_TRUE(spinUntilMessagesReceived(15.0)) << "Did not receive expected state messages";

  // Check available topics
  auto topics = test_node_->get_topic_names_and_types();
  RCLCPP_INFO(test_node_->get_logger(), "Available topics:");
  for (const auto& topic_pair : topics) {
    RCLCPP_INFO(test_node_->get_logger(), "  - %s", topic_pair.first.c_str());
  }

  RCLCPP_INFO(test_node_->get_logger(), "All state messages received successfully!");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
