# Enable testing with CTest
include(CTest)
enable_testing()

# Find required packages
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

# Add test_utils.hpp directory to includes
include_directories("${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/core")
include_directories("${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/models")

# Define a function to add a test with common settings
function(add_test_executable TEST_NAME)
  # Check if target already exists to avoid duplicate target errors
  if(NOT TARGET ${TEST_NAME})
    add_executable(${TEST_NAME} ${ARGN})

    target_include_directories(${TEST_NAME} PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}
      ${EIGEN3_INCLUDE_DIR}
    )

    target_link_libraries(${TEST_NAME} PRIVATE
      kinematic_arbiter_core
      GTest::gtest
      GTest::gtest_main
      ${EXTRA_LIBS}
    )

    target_compile_features(${TEST_NAME} PUBLIC cxx_std_20)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
  endif()
endfunction()

# Models tests
add_test_executable(rigid_body_state_model_test
  ${CMAKE_SOURCE_DIR}/test/models/rigid_body_state_model_test.cpp
)
set_tests_properties(rigid_body_state_model_test PROPERTIES LABELS "models")

# Core filter tests
add_test_executable(mediated_kalman_filter_test
  ${CMAKE_SOURCE_DIR}/test/mediated_kalman_filter_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)
set_tests_properties(mediated_kalman_filter_test PROPERTIES LABELS "filters")

# Add the filter initialization test using the same pattern as the mediated_kalman_filter_test
add_test_executable(filter_initialization_test
  ${CMAKE_SOURCE_DIR}/test/filter_initialization_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)
set_tests_properties(filter_initialization_test PROPERTIES LABELS "filters")

# Sensor tests

# Body velocity and pose sensor tests
add_test_executable(body_velocity_sensor_model_test
  ${CMAKE_SOURCE_DIR}/test/sensors/body_velocity_sensor_model_test.cpp
)

add_test_executable(pose_sensor_model_test
  ${CMAKE_SOURCE_DIR}/test/sensors/pose_sensor_model_test.cpp
)

# Position sensor test
add_test_executable(position_sensor_model_test
  ${CMAKE_SOURCE_DIR}/test/sensors/position_sensor_model_test.cpp
)

# IMU sensor tests
add_test_executable(imu_bias_estimator_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_bias_estimator_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_bias_estimator.cpp
)

add_test_executable(imu_sensor_model_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_model_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)

add_test_executable(imu_sensor_jacobian_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_jacobian_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)

add_test_executable(imu_sensor_mounting_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_sensor_mounting_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)

add_test_executable(imu_stationary_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_stationary_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)

add_test_executable(imu_prediction_inputs_test
  ${CMAKE_SOURCE_DIR}/test/sensors/imu_prediction_inputs_test.cpp
  ${CMAKE_SOURCE_DIR}/src/sensors/imu_sensor_model.cpp
)

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

# Integration tests
set(EXTRA_LIBS "drake::drake")
add_test_executable(test_drake_integration
  ${CMAKE_SOURCE_DIR}/test/integration/test_drake_integration.cpp
)
set_tests_properties(test_drake_integration PROPERTIES LABELS "integration")


# Add ROS2 wrapper test
add_test_executable(mkf_wrapper_test
  ${CMAKE_SOURCE_DIR}/test/ros2/mkf_wrapper_test.cpp
  ${CMAKE_SOURCE_DIR}/src/ros2/mkf_wrapper.cpp
)

# Find required ROS2 packages
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Link with the main library and ROS2 dependencies
target_link_libraries(mkf_wrapper_test PRIVATE
  kinematic_arbiter_core
  ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
)

# Add ROS2-specific include directories
target_include_directories(mkf_wrapper_test PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)

set_tests_properties(mkf_wrapper_test PROPERTIES LABELS "ros2;wrapper")


message(STATUS "Added all tests to test framework")
