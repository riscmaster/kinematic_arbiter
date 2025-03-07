# Enable testing with CTest
include(CTest)
enable_testing()

# Find required packages
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

# Use absolute path to the test files
set(STATISTICAL_UTILS_TEST_PATH "${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/core/statistical_utils_test.cpp")
set(DYNAMIC_COVARIANCE_TEST_PATH "${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/core/dynamic_covariance_test.cpp")

# Add test_utils.hpp directory to includes
include_directories("${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/core")

# Check if the files exist
if(NOT EXISTS "${STATISTICAL_UTILS_TEST_PATH}")
  message(FATAL_ERROR "Test file not found at: ${STATISTICAL_UTILS_TEST_PATH}")
endif()

if(NOT EXISTS "${DYNAMIC_COVARIANCE_TEST_PATH}")
  message(FATAL_ERROR "Test file not found at: ${DYNAMIC_COVARIANCE_TEST_PATH}")
endif()

# Add the statistical utils test with absolute path
add_executable(statistical_utils_test
  ${STATISTICAL_UTILS_TEST_PATH}
)

target_include_directories(statistical_utils_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(statistical_utils_test PRIVATE
  kinematic_arbiter_core
  GTest::gtest
  GTest::gtest_main
)

target_compile_features(statistical_utils_test PUBLIC cxx_std_20)

# Add the dynamic covariance test
add_executable(dynamic_covariance_test
  ${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/core/dynamic_covariance_test.cpp
)

target_include_directories(dynamic_covariance_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For test_utils.hpp
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(dynamic_covariance_test PRIVATE
  kinematic_arbiter_core
  GTest::gtest
  GTest::gtest_main
)

target_compile_features(dynamic_covariance_test PUBLIC cxx_std_20)

# Add all tests to CTest
add_test(
  NAME statistical_utils_test
  COMMAND statistical_utils_test
)

add_test(
  NAME dynamic_covariance_test
  COMMAND dynamic_covariance_test
)

# Set test properties
set_tests_properties(statistical_utils_test PROPERTIES
  LABELS "core"
)

set_tests_properties(dynamic_covariance_test PROPERTIES
  LABELS "core"
)

message(STATUS "Added statistical_utils_test and dynamic_covariance_test to test framework")

# Add sensor model tests
add_executable(pose_sensor_model_test
  ${CMAKE_SOURCE_DIR}/test/kinematic_arbiter/sensors/pose_sensor_model_test.cpp
)
target_link_libraries(pose_sensor_model_test
  PRIVATE
    kinematic_arbiter_core
    GTest::gtest
    GTest::gtest_main
)
add_test(
  NAME pose_sensor_model_test
  COMMAND pose_sensor_model_test
)
