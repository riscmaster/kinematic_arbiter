cmake_minimum_required(VERSION 3.8)

# Find GTest
find_package(GTest REQUIRED)

# Find required ROS2 packages for certain tests
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Function to create test executables with consistent properties
function(add_test_executable test_name test_file)
  message(STATUS "Adding test: ${test_name}")

  # Create an executable with the given test file
  add_executable(${test_name} ${test_file})

  # Link with common dependencies using keyword syntax
  target_link_libraries(${test_name} PRIVATE
    kinematic_arbiter_core
    gtest
    gtest_main
    ${ARGN}  # Additional libraries passed as arguments
  )

  # Add the test to CTest
  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
  )

  # Set common test properties
  set_tests_properties(${test_name}
    PROPERTIES
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test:$ENV{PYTHONPATH}"
  )
endfunction()

# Function to create ROS2 test executables
function(add_ros2_test_executable test_name test_file)
  message(STATUS "Adding ROS2 test: ${test_name}")

  # Create an executable with the given test file
  add_executable(${test_name} ${test_file})

  # Link with ROS2 libraries using keyword syntax
  target_link_libraries(${test_name} PRIVATE
    kinematic_arbiter_ros2
    gtest
    gtest_main
    ${ARGN}  # Additional libraries passed as arguments
  )

  # Add the test to CTest
  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
  )

  # Set common test properties
  set_tests_properties(${test_name}
    PROPERTIES
    TIMEOUT 60
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/test:$ENV{PYTHONPATH}"
    LABELS "ros2"
  )
endfunction()

# Core tests
add_test_executable(rigid_body_state_model_test
  "${CMAKE_SOURCE_DIR}/test/models/rigid_body_state_model_test.cpp")
set_tests_properties(rigid_body_state_model_test PROPERTIES LABELS "models")

add_test_executable(mediated_kalman_filter_test
  "${CMAKE_SOURCE_DIR}/test/core/mediated_kalman_filter_test.cpp")
set_tests_properties(mediated_kalman_filter_test PROPERTIES LABELS "filters")

add_test_executable(filter_initialization_test
  "${CMAKE_SOURCE_DIR}/test/core/filter_initialization_test.cpp")
set_tests_properties(filter_initialization_test PROPERTIES LABELS "filters")

add_test_executable(statistical_utils_test
  "${CMAKE_SOURCE_DIR}/test/core/statistical_utils_test.cpp")
set_tests_properties(statistical_utils_test PROPERTIES LABELS "core")

add_test_executable(trajectory_utils_test
  "${CMAKE_SOURCE_DIR}/test/core/trajectory_utils_test.cpp")
set_tests_properties(trajectory_utils_test PROPERTIES LABELS "core")

# Sensor tests
add_test_executable(body_velocity_sensor_model_test
  "${CMAKE_SOURCE_DIR}/test/sensors/body_velocity_sensor_model_test.cpp")

add_test_executable(pose_sensor_model_test
  "${CMAKE_SOURCE_DIR}/test/sensors/pose_sensor_model_test.cpp")

add_test_executable(position_sensor_model_test
  "${CMAKE_SOURCE_DIR}/test/sensors/position_sensor_model_test.cpp")

# IMU sensor tests
add_test_executable(imu_bias_estimator_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_bias_estimator_test.cpp")

add_test_executable(imu_sensor_model_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_model_test.cpp")

add_test_executable(imu_sensor_jacobian_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_jacobian_test.cpp")

add_test_executable(imu_sensor_mounting_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_mounting_test.cpp")

add_test_executable(imu_stationary_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_stationary_test.cpp")

add_test_executable(imu_prediction_inputs_test
  "${CMAKE_SOURCE_DIR}/test/sensors/imu_prediction_inputs_test.cpp")

set_tests_properties(
  body_velocity_sensor_model_test
  pose_sensor_model_test
  position_sensor_model_test
  imu_bias_estimator_test
  imu_sensor_jacobian_test
  imu_sensor_mounting_test
  imu_stationary_test
  imu_sensor_model_test
  imu_prediction_inputs_test
  PROPERTIES LABELS "sensors"
)

# ROS2 tests
add_ros2_test_executable(mkf_wrapper_test
  "${CMAKE_SOURCE_DIR}/test/ros2/mkf_wrapper_test.cpp"
  ${tf2_geometry_msgs_LIBRARIES})

add_ros2_test_executable(mkf_wrapper_time_test
  "${CMAKE_SOURCE_DIR}/test/ros2/mkf_wrapper_time_test.cpp")

message(STATUS "Added all tests to test framework")
