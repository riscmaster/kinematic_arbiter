#include "rclcpp/rclcpp.hpp"
#include "kinematic_arbiter/ros2/kinematic_arbiter_node.hpp"

int main(int argc, char * argv[])
{
  // Initialize ROS
  rclcpp::init(argc, argv);

  // Create node
  auto node = std::make_shared<kinematic_arbiter::ros2::KinematicArbiterNode>();

  // Spin the node
  rclcpp::spin(node);

  // Shutdown
  rclcpp::shutdown();

  return 0;
}
