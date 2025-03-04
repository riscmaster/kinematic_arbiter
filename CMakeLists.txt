cmake_minimum_required(VERSION 3.8)
project(kinematic_arbiter)

# Option to build only Python components
option(BUILD_PYTHON_ONLY "Build only Python components without Drake dependency" OFF)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find required dependencies for both Python and C++
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces for Python demo
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FilterMediation.action"
  DEPENDENCIES action_msgs
)

# Python installation - simplified approach
# Install Python modules directly
install(DIRECTORY src/single_dof_demo/
  DESTINATION lib/python${PYTHON_VERSION}/site-packages/single_dof_demo
  PATTERN "*.py"
)

# Make Python scripts executable before installing
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

# Install config files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# C++ components that depend on Drake
if(NOT BUILD_PYTHON_ONLY)
  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  # Find Drake
  find_package(drake REQUIRED)
  find_package(rclcpp REQUIRED)

  # C++ libraries for core implementation
  add_library(mkf_core
    src/core/mediated_kalman_filter.cpp
  )

  # Include directories
  target_include_directories(mkf_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # Link against Drake
  target_link_libraries(mkf_core PUBLIC
    drake::drake
  )

  # Install C++ target
  install(TARGETS mkf_core
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  # Install C++ headers
  install(DIRECTORY include/
    DESTINATION include
  )

  # Tests
  if(BUILD_TESTING)
    # C++ tests with Drake
    find_package(ament_cmake_gtest REQUIRED)

    # Add Drake test
    ament_add_gtest(test_drake_integration
      test/test_drake_integration.cpp
    )
    target_link_libraries(test_drake_integration
      mkf_core
      drake::drake
    )
  endif()

  ament_export_include_directories(include)
  ament_export_libraries(mkf_core)
  ament_export_targets(export_${PROJECT_NAME})
endif()

# Python tests can run regardless of BUILD_PYTHON_ONLY
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
