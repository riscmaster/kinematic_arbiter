#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <random>
#include <Eigen/Dense>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "kinematic_arbiter/ros2/pose_sensor_handler.hpp"
#include "kinematic_arbiter/ros2/velocity_sensor_handler.hpp"
#include "kinematic_arbiter/ros2/imu_sensor_handler.hpp"
#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/models/rigid_body_state_model.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"
#include "kinematic_arbiter/ros2/position_sensor_handler.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"

namespace kinematic_arbiter {
namespace ros2 {
namespace test {

  using namespace ::kinematic_arbiter::ros2::utils;

// Test wrappers to expose protected methods
class TestPositionSensorHandler : public PositionSensorHandler {
public:
  using PositionSensorHandler::PositionSensorHandler;
  using PositionSensorHandler::msgToVector;
  using PositionSensorHandler::vectorToMsg;
};

class TestPoseSensorHandler : public PoseSensorHandler {
public:
  using PoseSensorHandler::PoseSensorHandler;
  using PoseSensorHandler::msgToVector;
  using PoseSensorHandler::vectorToMsg;
};

class TestVelocitySensorHandler : public VelocitySensorHandler {
public:
  using VelocitySensorHandler::VelocitySensorHandler;
  using VelocitySensorHandler::msgToVector;
  using VelocitySensorHandler::vectorToMsg;
};

class TestImuSensorHandler : public ImuSensorHandler {
public:
  using ImuSensorHandler::ImuSensorHandler;
  using ImuSensorHandler::msgToVector;
  using ImuSensorHandler::vectorToMsg;
};

class SensorHandlersTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    // Create a node with a unique name to avoid conflicts
    static int counter = 0;
    std::string node_name = "sensor_handlers_test_" + std::to_string(counter++);
    node_ = std::make_shared<rclcpp::Node>(node_name);

    // Create a TF buffer and listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Add TF broadcaster and publish required transforms
    tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);

    // Publish the transforms we need - note the frame order
    // lookupTransform is called with (target, source) parameters, so we need
    // to set these frames up to match what the handlers expect
    publishTransform("base_link", "position_frame"); // base_link is parent
    publishTransform("base_link", "pose_frame");
    publishTransform("base_link", "velocity_frame");
    publishTransform("base_link", "imu_frame");

    // Give time for transforms to propagate through the system
    // This is critical for TF to work properly
    const auto start = node_->now();
    while ((node_->now() - start).seconds() < 0.5) {
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Create and initialize the state model
    auto state_model = std::make_shared<kinematic_arbiter::models::RigidBodyStateModel>();

    // Create a filter
    using StateIndex = kinematic_arbiter::core::StateIndex;
    filter_ = std::make_shared<kinematic_arbiter::core::MediatedKalmanFilter<StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>>(state_model);

    // Create test wrapper handlers
    position_handler_ = std::make_shared<TestPositionSensorHandler>(
        node_.get(), filter_, tf_buffer_,
        "test_position", "position_topic", "position_frame", "map", "base_link");

    pose_handler_ = std::make_shared<TestPoseSensorHandler>(
        node_.get(), filter_, tf_buffer_,
        "test_pose", "pose_topic", "pose_frame", "map", "base_link");

    velocity_handler_ = std::make_shared<TestVelocitySensorHandler>(
        node_.get(), filter_, tf_buffer_,
        "test_velocity", "velocity_topic", "velocity_frame", "map", "base_link");

    imu_handler_ = std::make_shared<TestImuSensorHandler>(
        node_.get(), filter_, tf_buffer_,
        "test_imu", "imu_topic", "imu_frame", "map", "base_link");

    // Process any pending callbacks to ensure publishers/subscribers are established
    rclcpp::spin_some(node_);
  }

  void TearDown() override {
    // Process any pending callbacks before cleanup
    if (rclcpp::ok()) {
      rclcpp::spin_some(node_);
    }

    // Explicit cleanup in reverse order of creation
    imu_handler_.reset();
    velocity_handler_.reset();
    pose_handler_.reset();
    position_handler_.reset();
    filter_.reset();
    tf_buffer_.reset();
    tf_broadcaster_.reset();
    tf_listener_.reset();
    node_.reset();
  }

  // Generate a random vector using Eigen's random function
  template<core::SensorType Type>
  Eigen::VectorXd generateRandomVector() {
    constexpr int dimension = core::MeasurementDimension<Type>::value;

    // Create random vector with values between -10 and 10
    Eigen::VectorXd vector = Eigen::VectorXd::Random(dimension) * 10.0;

    // Handle special case for pose (normalize quaternion)
    if constexpr (Type == core::SensorType::Pose) {
      // For poses, indices 3-6 represent a quaternion (w,x,y,z)
      Eigen::Vector4d quat = vector.segment<4>(3);

      // Normalize the quaternion
      quat.normalize();

      // Copy back to the vector
      vector.segment<4>(3) = quat;
    }

    return vector;
  }

  // Create a standard header for testing
  std_msgs::msg::Header createTestHeader(double time_sec = 1.0) {
    std_msgs::msg::Header header;
    header.frame_id = "test_frame";
    header.stamp = utils::doubleTimeToRosTime(time_sec);
    return header;
  }

  // Generic test method for round-trip conversion
  template<typename HandlerType, core::SensorType Type>
  void testRoundTripConversion(std::shared_ptr<HandlerType> handler) {
    constexpr int dimension = core::MeasurementDimension<Type>::value;

    // Run multiple tests with different random vectors
    for (int i = 0; i < 10; i++) {
      // Generate a random vector for this test iteration
      Eigen::VectorXd original_vector = generateRandomVector<Type>();

      // Create a header
      std_msgs::msg::Header header = createTestHeader(static_cast<double>(i));

      // Convert vector to message
      auto msg = handler->vectorToMsg(original_vector, header);

      // Convert message back to vector - use DynamicVector for non-const reference
      typename HandlerType::DynamicVector converted_vector(dimension);
      ASSERT_TRUE(handler->msgToVector(msg, converted_vector))
          << "Failed to convert message back to vector in iteration " << i;

      // Verify that the vectors are close (using Eigen's norm operation)
      const double tolerance = 1e-5;
      EXPECT_LT((original_vector - converted_vector).norm(), tolerance)
          << "Vector mismatch in iteration " << i
          << "\nOriginal: " << original_vector.transpose()
          << "\nConverted: " << converted_vector.transpose();
    }
  }

  // Helper method to publish transforms
  void publishTransform(const std::string& parent, const std::string& child) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = node_->get_clock()->now();
    transform.header.frame_id = parent;
    transform.child_frame_id = child;

    // Identity transform
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.w = 1.0;
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;

    tf_broadcaster_->sendTransform(transform);
  }

  // Shared test objects
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<kinematic_arbiter::core::MediatedKalmanFilter<kinematic_arbiter::core::StateIndex::kFullStateSize, kinematic_arbiter::core::StateModelInterface>> filter_;
  std::shared_ptr<TestPositionSensorHandler> position_handler_;
  std::shared_ptr<TestPoseSensorHandler> pose_handler_;
  std::shared_ptr<TestVelocitySensorHandler> velocity_handler_;
  std::shared_ptr<TestImuSensorHandler> imu_handler_;
};

// Test for PositionSensorHandler
TEST_F(SensorHandlersTest, PositionRoundTripConversion) {
  testRoundTripConversion<TestPositionSensorHandler, core::SensorType::Position>(position_handler_);
}

// Test for PoseSensorHandler
TEST_F(SensorHandlersTest, PoseRoundTripConversion) {
  testRoundTripConversion<TestPoseSensorHandler, core::SensorType::Pose>(pose_handler_);
}

// Test for VelocitySensorHandler
TEST_F(SensorHandlersTest, VelocityRoundTripConversion) {
  testRoundTripConversion<TestVelocitySensorHandler, core::SensorType::BodyVelocity>(velocity_handler_);
}

// Test for ImuSensorHandler
TEST_F(SensorHandlersTest, ImuRoundTripConversion) {
  testRoundTripConversion<TestImuSensorHandler, core::SensorType::Imu>(imu_handler_);
}

// Verify publishers/subscribers are created
TEST_F(SensorHandlersTest, VerifyPublishersAndSubscribers) {
  // Verify publishers and subscribers for each sensor type

  // Position sensor
  EXPECT_GT(node_->count_subscribers("position_topic"), 0)
      << "Should have at least one subscriber for position topic";
  EXPECT_GT(node_->count_publishers("position_topic/expected"), 0)
      << "Should have expected measurement publisher for position";
  EXPECT_GT(node_->count_publishers("position_topic/upper_bound"), 0)
      << "Should have upper bound publisher for position";
  EXPECT_GT(node_->count_publishers("position_topic/lower_bound"), 0)
      << "Should have lower bound publisher for position";

  // Pose sensor
  EXPECT_GT(node_->count_subscribers("pose_topic"), 0)
      << "Should have at least one subscriber for pose topic";
  EXPECT_GT(node_->count_publishers("pose_topic/expected"), 0)
      << "Should have expected measurement publisher for pose";
  EXPECT_GT(node_->count_publishers("pose_topic/upper_bound"), 0)
      << "Should have upper bound publisher for pose";
  EXPECT_GT(node_->count_publishers("pose_topic/lower_bound"), 0)
      << "Should have lower bound publisher for pose";

  // Velocity sensor
  EXPECT_GT(node_->count_subscribers("velocity_topic"), 0)
      << "Should have at least one subscriber for velocity topic";
  EXPECT_GT(node_->count_publishers("velocity_topic/expected"), 0)
      << "Should have expected measurement publisher for velocity";
  EXPECT_GT(node_->count_publishers("velocity_topic/upper_bound"), 0)
      << "Should have upper bound publisher for velocity";
  EXPECT_GT(node_->count_publishers("velocity_topic/lower_bound"), 0)
      << "Should have lower bound publisher for velocity";

  // IMU sensor
  EXPECT_GT(node_->count_subscribers("imu_topic"), 0)
      << "Should have at least one subscriber for imu topic";
  EXPECT_GT(node_->count_publishers("imu_topic/expected"), 0)
      << "Should have expected measurement publisher for imu";
  EXPECT_GT(node_->count_publishers("imu_topic/upper_bound"), 0)
      << "Should have upper bound publisher for imu";
  EXPECT_GT(node_->count_publishers("imu_topic/lower_bound"), 0)
      << "Should have lower bound publisher for imu";
}

} // namespace test
} // namespace ros2
} // namespace kinematic_arbiter

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
