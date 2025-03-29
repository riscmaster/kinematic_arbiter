#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

class ParameterTestNode : public rclcpp::Node {
public:
  ParameterTestNode()
      : Node("parameter_test_node", rclcpp::NodeOptions().allow_undeclared_parameters(true)) {

    RCLCPP_INFO(this->get_logger(), "Parameter Test Node initialized");

    // First, dump all parameters at the root level
    auto params = this->list_parameters({}, 0);
    RCLCPP_INFO(this->get_logger(), "Root parameters (%zu):", params.names.size());
    for (const auto& name : params.names) {
      RCLCPP_INFO(this->get_logger(), "  %s", name.c_str());
    }

    // Try with different depths
    testDepth("", 1);
    testDepth("", 2);
    testDepth("", 3);

    // Try explicit sensor types
    testDepth("position_sensors", 1);
    testDepth("pose_sensors", 1);
    testDepth("velocity_sensors", 1);
    testDepth("imu_sensors", 1);

    // Try explicit sensor types with deeper depth
    testDepth("position_sensors", 2);
    testDepth("pose_sensors", 2);
    testDepth("velocity_sensors", 2);
    testDepth("imu_sensors", 2);

    // Try to directly access some parameters we expect
    tryGetParameter("position_sensors.position_sensor.topic");
    tryGetParameter("position_sensors.position_sensor.frame_id");
    tryGetParameter("pose_sensors.pose_sensor.topic");
    tryGetParameter("imu_sensors.imu_sensor.topic");

    // Try alternative method - traverse through prefixes
    explorePrefix("position_sensors");
    explorePrefix("pose_sensors");
    explorePrefix("velocity_sensors");
    explorePrefix("imu_sensors");

    // Try ROS 2 service API directly
    callListParametersService();
  }

private:
  void testDepth(const std::string& prefix, uint64_t depth) {
    auto params = this->list_parameters({prefix}, depth);

    RCLCPP_INFO(this->get_logger(), "\nTesting with prefix='%s', depth=%lu",
               prefix.c_str(), depth);

    RCLCPP_INFO(this->get_logger(), "  Parameters (%zu):", params.names.size());
    for (const auto& name : params.names) {
      RCLCPP_INFO(this->get_logger(), "    %s", name.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "  Prefixes (%zu):", params.prefixes.size());
    for (const auto& prefix : params.prefixes) {
      RCLCPP_INFO(this->get_logger(), "    %s", prefix.c_str());
    }
  }

  void tryGetParameter(const std::string& param_name) {
    RCLCPP_INFO(this->get_logger(), "\nTrying to get parameter: %s", param_name.c_str());

    // First check if it exists
    if (this->has_parameter(param_name)) {
      auto param = this->get_parameter(param_name);
      RCLCPP_INFO(this->get_logger(), "  Found! Type: %d, Value: %s",
                 static_cast<int>(param.get_type()), param.value_to_string().c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "  Parameter not found");

      // Try to declare it anyway
      RCLCPP_INFO(this->get_logger(), "  Trying to declare and get it...");
      this->declare_parameter(param_name, "");

      if (this->has_parameter(param_name)) {
        auto param = this->get_parameter(param_name);
        RCLCPP_INFO(this->get_logger(), "  Now found! Type: %d, Value: %s",
                   static_cast<int>(param.get_type()), param.value_to_string().c_str());
      } else {
        RCLCPP_INFO(this->get_logger(), "  Still not found after declaration");
      }
    }
  }

  void explorePrefix(const std::string& prefix) {
    RCLCPP_INFO(this->get_logger(), "\nExploring prefix: %s", prefix.c_str());

    // Get first level of children
    auto params = this->list_parameters({prefix}, 1);

    for (const auto& child_prefix : params.prefixes) {
      RCLCPP_INFO(this->get_logger(), "  Found child prefix: %s", child_prefix.c_str());

      // For each child, get parameters beneath it
      auto child_params = this->list_parameters({child_prefix}, 1);

      for (const auto& param_name : child_params.names) {
        auto param = this->get_parameter(param_name);
        RCLCPP_INFO(this->get_logger(), "    %s = %s",
                   param_name.c_str(), param.value_to_string().c_str());
      }
    }
  }

  void callListParametersService() {
    RCLCPP_INFO(this->get_logger(), "\nDirectly calling list_parameters service");

    auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(this);
    if (!parameters_client->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_ERROR(this->get_logger(), "Parameters service not available");
      return;
    }

    // Try different depths
    for (uint64_t depth = 0; depth <= 3; depth++) {
      RCLCPP_INFO(this->get_logger(), "  Service call with depth=%lu", depth);

      auto result = parameters_client->list_parameters({}, depth);

      RCLCPP_INFO(this->get_logger(), "    Parameters (%zu):", result.names.size());
      for (size_t i = 0; i < result.names.size() && i < 10; i++) {
        RCLCPP_INFO(this->get_logger(), "      %s", result.names[i].c_str());
      }
      if (result.names.size() > 10) {
        RCLCPP_INFO(this->get_logger(), "      ... and %zu more", result.names.size() - 10);
      }

      RCLCPP_INFO(this->get_logger(), "    Prefixes (%zu):", result.prefixes.size());
      for (const auto& prefix : result.prefixes) {
        RCLCPP_INFO(this->get_logger(), "      %s", prefix.c_str());
      }
    }
  }
};

// Add a proper main function instead of using Google Test
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ParameterTestNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
