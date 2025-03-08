# Basic minimum CMake version
cmake_minimum_required(VERSION 3.8)
project(kinematic_arbiter)

# First, let's check what's being processed
message(STATUS "CMAKE_CURRENT_LIST_FILE: ${CMAKE_CURRENT_LIST_FILE}")
message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# Get build directory information
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

# Check what CMake is actually running
execute_process(
  COMMAND find ${CMAKE_BINARY_DIR} -name "CMakeLists.txt"
  OUTPUT_VARIABLE found_cmakelists
)
message(STATUS "Found CMakeLists.txt files in build dir: ${found_cmakelists}")

# Get original CMakeLists content
file(READ ${CMAKE_CURRENT_LIST_FILE} original_content)
string(LENGTH "${original_content}" content_length)
message(STATUS "Original CMakeLists.txt length: ${content_length}")

# Option to build only Python components
option(BUILD_PYTHON_ONLY "Build only Python components without Drake dependency" OFF)

# Set C++20 as the standard to handle designated initializers
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(drake QUIET)

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

  # Use global include directories instead of target-specific ones
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  # Tests - Both Drake integration and state model testing
  if(BUILD_TESTING)
    message(STATUS "Build testing is enabled")

    # Add the test directory to the include path
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    # Find GTest
    find_package(GTest REQUIRED)

    # Add the Drake test
    add_executable(test_drake_integration
      test/test_drake_integration.cpp)

    # Link against libraries
    target_link_libraries(test_drake_integration
      drake::drake
      GTest::gtest
      GTest::gtest_main)

    # Register as a test with CTest
    add_test(NAME test_drake_integration
             COMMAND test_drake_integration)

    # Add the state model test
    add_executable(rigid_body_state_model_test
      test/models/rigid_body_state_model_test.cpp)

    # Link against Drake and GTest
    target_link_libraries(rigid_body_state_model_test
      drake::drake
      GTest::gtest
      GTest::gtest_main)

    # Register as a test
    add_test(NAME rigid_body_state_model_test
             COMMAND rigid_body_state_model_test)
  endif()
endif()

# Ensure these headers are installed
install(DIRECTORY include/
  DESTINATION include
)

# Export includes and dependencies - MUST come before ament_package
ament_export_include_directories(include)
ament_export_libraries(kinematic_arbiter_core)
ament_export_targets(export_${PROJECT_NAME})

# Build the core library before anything else
add_library(kinematic_arbiter_core
  src/kinematic_arbiter/core/statistical_utils.cpp
  src/sensors/imu_bias_estimator.cpp
  # Add other source files as needed
)

# Set properties and dependencies for the core library
target_include_directories(kinematic_arbiter_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

target_compile_features(kinematic_arbiter_core PUBLIC cxx_std_20)

# Build tests
if(BUILD_TESTING)
  # Let the existing test structure handle the tests
  include(test/CMakeLists.txt)
  # Don't use add_subdirectory to avoid binary directory conflicts
  # add_subdirectory(test)
endif()

# Install targets
install(TARGETS kinematic_arbiter_core
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

# MUST be the last command
ament_package()

# Print line 150 (if it exists)
string(REPLACE "\n" ";" content_lines "${original_content}")
list(LENGTH content_lines num_lines)
message(STATUS "Total lines in original CMakeLists.txt: ${num_lines}")

if(num_lines GREATER_EQUAL 150)
  list(GET content_lines 149 line_150) # 0-based index so 149 is line 150
  message(STATUS "Line 150 content: '${line_150}'")
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Add include directories for the project
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)
