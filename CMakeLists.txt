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
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Generate interfaces for Python demo
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FilterMediation.action"
  DEPENDENCIES action_msgs
)

# Python installation - ensure modules are in Python's site-packages
install(DIRECTORY src/single_dof_demo/
  DESTINATION lib/python${PYTHON_VERSION}/site-packages/single_dof_demo
  PATTERN "*.py"
)

# Create a Python setup.py file for proper Python packaging
ament_python_install_package(single_dof_demo
  PACKAGE_DIR src/single_dof_demo
)

# Install Python executables
install(PROGRAMS
  src/single_dof_demo/ros2/nodes/signal_generator_node.py
  src/single_dof_demo/ros2/nodes/kalman_filter_node.py
  src/single_dof_demo/ros2/nodes/mediated_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
endif()
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)

# C++ components
if(NOT BUILD_PYTHON_ONLY)
  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()

  # Include directories
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}  # Add Eigen includes here for all targets
  )

  # Print the current directory structure for debugging
  execute_process(
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src -type f -name "*.cpp"
    OUTPUT_VARIABLE found_source_files
  )
  message(STATUS "Found source files: ${found_source_files}")

  # 1. Build the core library WITHOUT ROS2 dependencies
  add_library(kinematic_arbiter_core SHARED)

  # Add core source files if they exist
  foreach(file
      src/models/rigid_body_state_model.cpp
      src/sensors/position_sensor_model.cpp
      src/sensors/pose_sensor_model.cpp
      src/sensors/body_velocity_sensor_model.cpp
      src/sensors/imu_sensor_model.cpp
      src/sensors/imu_bias_estimator.cpp
      src/core/mediated_kalman_filter.cpp
      src/core/statistical_utils.cpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      target_sources(kinematic_arbiter_core PRIVATE "${file}")
    else()
      message(STATUS "Source file not found: ${file}")
    endif()
  endforeach()

  # Configure the core library
  target_include_directories(kinematic_arbiter_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
  )
  target_link_libraries(kinematic_arbiter_core PUBLIC
    Eigen3::Eigen
  )

  # 2. Build the ROS2 wrapper library separately
  add_library(kinematic_arbiter_ros2 SHARED
    src/ros2/filter_wrapper.cpp
  )

  # The wrapper depends on the core and ROS2 libraries
  target_include_directories(kinematic_arbiter_ros2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${rclcpp_INCLUDE_DIRS}
    ${geometry_msgs_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${std_srvs_INCLUDE_DIRS}
    ${tf2_INCLUDE_DIRS}
    ${tf2_ros_INCLUDE_DIRS}
    ${tf2_eigen_INCLUDE_DIRS}
    ${tf2_geometry_msgs_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
  )

  target_link_libraries(kinematic_arbiter_ros2 PUBLIC
    kinematic_arbiter_core
    ${rclcpp_LIBRARIES}
    ${geometry_msgs_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${std_srvs_LIBRARIES}
    ${tf2_LIBRARIES}
    ${tf2_ros_LIBRARIES}
    ${tf2_eigen_LIBRARIES}
    ${tf2_geometry_msgs_LIBRARIES}
  )

  # Tests - delegate to test/CMakeLists.txt
  if(BUILD_TESTING)
    message(STATUS "Build testing is enabled")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test)
  endif()

  # Add ROS2 node executables - now depending on the wrapper library
  add_executable(figure8_simulator_node
    src/ros2/simulation/figure8_simulator_node.cpp
  )
  target_link_libraries(figure8_simulator_node PUBLIC
    kinematic_arbiter_ros2
  )

  # Only build kinematic_arbiter_node if the file exists
  set(CPP_TARGETS "")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/ros2/kinematic_arbiter_node.cpp")
    add_executable(kinematic_arbiter_node
      src/ros2/main.cpp
      src/ros2/kinematic_arbiter_node.cpp
    )
    target_link_libraries(kinematic_arbiter_node PUBLIC
      kinematic_arbiter_ros2
    )

    # Add to install targets
    list(APPEND CPP_TARGETS kinematic_arbiter_node)
  endif()

  # Install C++ targets
  install(TARGETS
    kinematic_arbiter_core
    kinematic_arbiter_ros2
    figure8_simulator_node
    ${CPP_TARGETS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  # Install header files
  install(DIRECTORY include/
    DESTINATION include
  )
endif()

# Register the package with ament index
ament_package()

# Print line 150 (if it exists)
string(REPLACE "\n" ";" content_lines "${original_content}")
list(LENGTH content_lines num_lines)
message(STATUS "Total lines in original CMakeLists.txt: ${num_lines}")

if(num_lines GREATER_EQUAL 150)
  list(GET content_lines 149 line_150) # 0-based index so 149 is line 150
  message(STATUS "Line 150 content: '${line_150}'")
endif()
