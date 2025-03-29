#include <gtest/gtest.h>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Define the YAML configuration as a string using the working configuration
const char* CONFIG_YAML = R"(
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

    # Topic names - use the same names that worked in the launch file
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

class Figure8SimulatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize ROS
    rclcpp::init(0, nullptr);

    // Create our test node
    test_node_ = std::make_shared<rclcpp::Node>("figure8_simulator_test");

    // Track message reception
    truth_pose_received_ = false;
    truth_velocity_received_ = false;
    position_received_ = false;
    pose_received_ = false;
    velocity_received_ = false;
    imu_received_ = false;

    // Subscribe to expected topics - MATCHING the topic names in the YAML
    truth_pose_sub_ = test_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "truth/pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        truth_pose_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received truth pose with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    truth_velocity_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistStamped>(
      "truth/velocity", 10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        truth_velocity_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received truth velocity with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    // Update subscriptions to match the actual topic names from the YAML
    position_sub_ = test_node_->create_subscription<geometry_msgs::msg::PointStamped>(
      "position_sensor", 10,
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        position_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received position with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    pose_sub_ = test_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "pose_sensor", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        pose_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received pose with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    velocity_sub_ = test_node_->create_subscription<geometry_msgs::msg::TwistStamped>(
      "velocity_sensor", 10,
      [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        velocity_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received velocity with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    imu_sub_ = test_node_->create_subscription<sensor_msgs::msg::Imu>(
      "imu_sensor", 10,
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_received_ = true;
        RCLCPP_INFO_ONCE(test_node_->get_logger(), "Received IMU with frame_id: %s",
                         msg->header.frame_id.c_str());
      });

    // Initialize executor and add our test node
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor_->add_node(test_node_);

    // Start the executor in a separate thread
    executor_thread_ = std::thread([this]() { executor_->spin(); });

    // Start the simulator in a separate process using system command
    RCLCPP_INFO(test_node_->get_logger(), "Starting Figure-8 simulator node");

    // Write the YAML to a temporary file
    std::string temp_yaml_path = "/tmp/figure8_sim_test_config.yaml";
    std::ofstream yaml_file(temp_yaml_path);
    yaml_file << CONFIG_YAML;
    yaml_file.close();

    // Launch the figure8_simulator node with the config file
    std::string launch_cmd = "ros2 run kinematic_arbiter figure8_simulator_node --ros-args --params-file " +
                             temp_yaml_path + " > /tmp/figure8_sim_output.log 2>&1 &";
    system(launch_cmd.c_str());

    // Give it time to start up
    std::this_thread::sleep_for(std::chrono::seconds(2));

    RCLCPP_INFO(test_node_->get_logger(), "Figure-8 simulator node started");
  }

  void TearDown() override {
    RCLCPP_INFO(test_node_->get_logger(), "Test teardown - stopping simulator");

    // Stop the simulator process
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

  bool spinUntilMessagesReceived(double timeout_seconds = 5.0) {
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::duration<double>(timeout_seconds);

    while (std::chrono::steady_clock::now() < end_time) {
      // We don't need to spin here as the executor is already spinning in a separate thread
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Check if we've received all expected messages
      if (truth_pose_received_ && truth_velocity_received_ &&
          position_received_ && pose_received_ &&
          velocity_received_ && imu_received_) {
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
                  "Current reception: truth_pose=%s, truth_velocity=%s, position=%s, pose=%s, velocity=%s, imu=%s",
                  truth_pose_received_ ? "YES" : "NO",
                  truth_velocity_received_ ? "YES" : "NO",
                  position_received_ ? "YES" : "NO",
                  pose_received_ ? "YES" : "NO",
                  velocity_received_ ? "YES" : "NO",
                  imu_received_ ? "YES" : "NO");
      }
    }

    // Print final diagnostic info
    RCLCPP_INFO(test_node_->get_logger(),
               "Final message reception: truth_pose=%s, truth_velocity=%s, position=%s, pose=%s, velocity=%s, imu=%s",
               truth_pose_received_ ? "YES" : "NO",
               truth_velocity_received_ ? "YES" : "NO",
               position_received_ ? "YES" : "NO",
               pose_received_ ? "YES" : "NO",
               velocity_received_ ? "YES" : "NO",
               imu_received_ ? "YES" : "NO");

    return false;
  }

  // Test node and executor
  std::shared_ptr<rclcpp::Node> test_node_;
  std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executor_thread_;

  // Subscriptions
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr truth_velocity_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr position_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  // Flags for message reception
  bool truth_pose_received_;
  bool truth_velocity_received_;
  bool position_received_;
  bool pose_received_;
  bool velocity_received_;
  bool imu_received_;
};

TEST_F(Figure8SimulatorTest, VerifyTopicPublication) {
  RCLCPP_INFO(test_node_->get_logger(), "Running topic publication verification test");

  // Verify topics are being published with a longer timeout
  ASSERT_TRUE(spinUntilMessagesReceived(15.0)) << "Did not receive all expected messages";

  // Check available topics
  auto topics = test_node_->get_topic_names_and_types();
  RCLCPP_INFO(test_node_->get_logger(), "Available topics:");
  for (const auto& topic_pair : topics) {
    RCLCPP_INFO(test_node_->get_logger(), "  - %s", topic_pair.first.c_str());
  }

  RCLCPP_INFO(test_node_->get_logger(), "All messages received successfully!");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
