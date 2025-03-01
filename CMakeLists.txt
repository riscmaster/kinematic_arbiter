cmake_minimum_required(VERSION 3.8)
project(kinematic_arbiter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FilterMediation.action"
  DEPENDENCIES action_msgs
)

# Install Python package
ament_python_install_package(
  ${PROJECT_NAME}
  PACKAGE_DIR "src"
)

# Make scripts executable before installing
execute_process(
  COMMAND chmod +x
    ${CMAKE_CURRENT_SOURCE_DIR}/src/single_dof_demo/ros2/nodes/signal_generator_node.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/single_dof_demo/ros2/nodes/kalman_filter_node.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/single_dof_demo/ros2/nodes/mediated_filter_node.py
)

# Install Python executables
install(PROGRAMS
  src/single_dof_demo/ros2/nodes/signal_generator_node.py
  src/single_dof_demo/ros2/nodes/kalman_filter_node.py
  src/single_dof_demo/ros2/nodes/mediated_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
