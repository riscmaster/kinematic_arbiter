#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "kinematic_arbiter/ros2/mkf_wrapper.hpp"
#include "kinematic_arbiter/core/trajectory_utils.hpp"
#include "kinematic_arbiter/sensors/body_velocity_sensor_model.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/sensors/pose_sensor_model.hpp"
#include "kinematic_arbiter/sensors/position_sensor_model.hpp"


using namespace kinematic_arbiter;
using namespace kinematic_arbiter::ros2::wrapper;

class FilterWrapperTest : public ::testing::Test {
  using MeasurementModelInterface = kinematic_arbiter::core::MeasurementModelInterface;
  using BodyVelocitySensorModel = kinematic_arbiter::sensors::BodyVelocitySensorModel;
  using ImuSensorModel = kinematic_arbiter::sensors::ImuSensorModel;
  using PoseSensorModel = kinematic_arbiter::sensors::PoseSensorModel;
  using PositionSensorModel = kinematic_arbiter::sensors::PositionSensorModel;
protected:
  void SetUp() override {
    // Initialize ROS context if needed
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }

    // Create wrapper with default parameters
    wrapper_ = std::make_shared<FilterWrapper>();

    // Register sensors
    position_sensor_id_ = wrapper_->registerPositionSensor("test_position");
    pose_sensor_id_ = wrapper_->registerPoseSensor("test_pose");
    velocity_sensor_id_ = wrapper_->registerBodyVelocitySensor("test_velocity");
    imu_sensor_id_ = wrapper_->registerImuSensor("test_imu");
    wrapper_->setMaxDelayWindow(kMaxDelayWindow);
  }

  void TearDown() override {
    // Cleanup
    wrapper_.reset();
  }
  // Create position message using expected measurement
  geometry_msgs::msg::PointStamped createPositionMsg(double t) {
    MeasurementModelInterface::DynamicVector measurement;
    EXPECT_TRUE(wrapper_->getExpectedMeasurementByID(position_sensor_id_,
      measurement, utils::Figure8Trajectory(t)));

    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = wrapper_->doubleTimeToRosTime(t);
    msg.header.frame_id = "world";

    // Position from expected measurement
    msg.point.x = measurement[PositionSensorModel::MeasurementIndex::X]; // X
    msg.point.y = measurement[PositionSensorModel::MeasurementIndex::Y]; // Y
    msg.point.z = measurement[PositionSensorModel::MeasurementIndex::Z]; // Z

    return msg;
  }

  // Create pose message using expected measurement
  geometry_msgs::msg::PoseStamped createPoseMsg(double t) {
    MeasurementModelInterface::DynamicVector measurement;
    EXPECT_TRUE(wrapper_->getExpectedMeasurementByID(pose_sensor_id_,
      measurement, utils::Figure8Trajectory(t)));

    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = wrapper_->doubleTimeToRosTime(t);
    msg.header.frame_id = "world";

    // Position
    msg.pose.position.x = measurement[PoseSensorModel::MeasurementIndex::X]; // X
    msg.pose.position.y = measurement[PoseSensorModel::MeasurementIndex::Y]; // Y
    msg.pose.position.z = measurement[PoseSensorModel::MeasurementIndex::Z]; // Z

    // Orientation (quaternion)
    msg.pose.orientation.w = measurement[PoseSensorModel::MeasurementIndex::QW]; // W
    msg.pose.orientation.x = measurement[PoseSensorModel::MeasurementIndex::QX]; // X
    msg.pose.orientation.y = measurement[PoseSensorModel::MeasurementIndex::QY]; // Y
    msg.pose.orientation.z = measurement[PoseSensorModel::MeasurementIndex::QZ]; // Z

    return msg;
  }

  // Create velocity message using expected measurement
  geometry_msgs::msg::TwistStamped createVelocityMsg(double t) {
    MeasurementModelInterface::DynamicVector measurement;
    EXPECT_TRUE(wrapper_->getExpectedMeasurementByID(velocity_sensor_id_,
      measurement, utils::Figure8Trajectory(t)));

    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = wrapper_->doubleTimeToRosTime(t);
    msg.header.frame_id = "body";

    // Linear velocity
    msg.twist.linear.x = measurement[BodyVelocitySensorModel::MeasurementIndex::VX]; // Vx
    msg.twist.linear.y = measurement[BodyVelocitySensorModel::MeasurementIndex::VY]; // Vy
    msg.twist.linear.z = measurement[BodyVelocitySensorModel::MeasurementIndex::VZ]; // Vz

    // Angular velocity
    msg.twist.angular.x = measurement[BodyVelocitySensorModel::MeasurementIndex::WX]; // Wx
    msg.twist.angular.y = measurement[BodyVelocitySensorModel::MeasurementIndex::WY]; // Wy
    msg.twist.angular.z = measurement[BodyVelocitySensorModel::MeasurementIndex::WZ]; // Wz

    return msg;
  }

  sensor_msgs::msg::Imu createImuMsg(double t) {
    MeasurementModelInterface::DynamicVector measurement;
    EXPECT_TRUE( wrapper_->getExpectedMeasurementByID(imu_sensor_id_,
    measurement, utils::Figure8Trajectory(t)));

    sensor_msgs::msg::Imu msg;
    msg.header.stamp = wrapper_->doubleTimeToRosTime(t);
    msg.header.frame_id = "imu";

    // Angular velocity
    msg.angular_velocity.x = measurement[ImuSensorModel::MeasurementIndex::GX];
    msg.angular_velocity.y = measurement[ImuSensorModel::MeasurementIndex::GY];
    msg.angular_velocity.z = measurement[ImuSensorModel::MeasurementIndex::GZ];

    // Linear acceleration
    msg.linear_acceleration.x = measurement[ImuSensorModel::MeasurementIndex::AX];
    msg.linear_acceleration.y = measurement[ImuSensorModel::MeasurementIndex::AY];
    msg.linear_acceleration.z = measurement[ImuSensorModel::MeasurementIndex::AZ];

    return msg;
  }

  // Utility to set sensor transforms
  void setSensorTransforms() {
    // Create identity transforms for testing
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = "body";
    transform.child_frame_id = "sensor";
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.w = 1.0;
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;

    // Set for each sensor
    ASSERT_TRUE(wrapper_->setSensorTransform(position_sensor_id_, transform));
    ASSERT_TRUE(wrapper_->setSensorTransform(pose_sensor_id_, transform));
    ASSERT_TRUE(wrapper_->setSensorTransform(velocity_sensor_id_, transform));
    ASSERT_TRUE(wrapper_->setSensorTransform(imu_sensor_id_, transform));
  }

  std::shared_ptr<FilterWrapper> wrapper_;
  std::string position_sensor_id_;
  std::string pose_sensor_id_;
  std::string velocity_sensor_id_;
  std::string imu_sensor_id_;
  const double kMaxDelayWindow = 1.0;
};

// Test sensor registration and ID management
TEST_F(FilterWrapperTest, SensorRegistration) {
  // Check that sensor IDs are not empty
  EXPECT_FALSE(position_sensor_id_.empty());
  EXPECT_FALSE(pose_sensor_id_.empty());
  EXPECT_FALSE(velocity_sensor_id_.empty());
  EXPECT_FALSE(imu_sensor_id_.empty());

  // Registering the same sensor type with the same name should return the same ID
  std::string position_sensor_id2 = wrapper_->registerPositionSensor("test_position");
  EXPECT_EQ(position_sensor_id_, position_sensor_id2);

  // Registering a different sensor type with the same name should return a different ID
  std::string pose_sensor_id2 = wrapper_->registerPoseSensor("test_position");
  EXPECT_NE(position_sensor_id_, pose_sensor_id2);

  // Re-registering the same sensor type and name should return the same ID again
  std::string pose_sensor_id3 = wrapper_->registerPoseSensor("test_position");
  EXPECT_EQ(pose_sensor_id2, pose_sensor_id3);

  // Now try with a completely new name
  std::string position_sensor_id3 = wrapper_->registerPositionSensor("new_position_sensor");
  EXPECT_NE(position_sensor_id_, position_sensor_id3);

  // Re-registering the new name should give the same ID
  std::string position_sensor_id4 = wrapper_->registerPositionSensor("new_position_sensor");
  EXPECT_EQ(position_sensor_id3, position_sensor_id4);
}

// Test setting and getting transforms
TEST_F(FilterWrapperTest, SensorTransforms) {
  // Create a non-identity transform
  geometry_msgs::msg::TransformStamped input_transform;
  input_transform.header.frame_id = "body";
  input_transform.child_frame_id = "sensor";
  input_transform.transform.translation.x = 1.0;
  input_transform.transform.translation.y = 2.0;
  input_transform.transform.translation.z = 3.0;

  // Use 90 degree rotation around Z
  input_transform.transform.rotation.w = 0.7071;
  input_transform.transform.rotation.x = 0.0;
  input_transform.transform.rotation.y = 0.0;
  input_transform.transform.rotation.z = 0.7071;

  // Set transform for position sensor
  ASSERT_TRUE(wrapper_->setSensorTransform(position_sensor_id_, input_transform));

  // Get back the transform
  geometry_msgs::msg::TransformStamped output_transform;
  ASSERT_TRUE(wrapper_->getSensorTransform(position_sensor_id_, "body", output_transform));

  // Verify transform values
  EXPECT_DOUBLE_EQ(output_transform.transform.translation.x, 1.0);
  EXPECT_DOUBLE_EQ(output_transform.transform.translation.y, 2.0);
  EXPECT_DOUBLE_EQ(output_transform.transform.translation.z, 3.0);
  EXPECT_NEAR(output_transform.transform.rotation.w, 0.7071, 1e-4);
  EXPECT_NEAR(output_transform.transform.rotation.x, 0.0, 1e-4);
  EXPECT_NEAR(output_transform.transform.rotation.y, 0.0, 1e-4);
  EXPECT_NEAR(output_transform.transform.rotation.z, 0.7071, 1e-4);
}

// Test filter initialization with position measurement
TEST_F(FilterWrapperTest, InitializeWithPosition) {
  // Create and process position measurement
  double t = 0.0;
  auto pos_msg = createPositionMsg(t);

  ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg));

  // Get pose estimate and check position
  auto pose_estimate = wrapper_->getPoseEstimate(pos_msg.header.stamp);

  EXPECT_NEAR(pose_estimate.pose.pose.position.x, pos_msg.point.x, 1e-4);
  EXPECT_NEAR(pose_estimate.pose.pose.position.y, pos_msg.point.y, 1e-4);
  EXPECT_NEAR(pose_estimate.pose.pose.position.z, pos_msg.point.z, 1e-4);
}

// Test dead reckoning
TEST_F(FilterWrapperTest, DeadReckoning) {
  // Initialize with first position measurement
  double t = 0.0;
  auto pos_msg = createPositionMsg(t);
  ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg));

  // Jump ahead in time and predict
  double t_future = 0.5;  // 0.5 seconds later
  wrapper_->predictTo(wrapper_->doubleTimeToRosTime(t_future));

  // Get the predicted state
  auto pose_estimate = wrapper_->getPoseEstimate(wrapper_->doubleTimeToRosTime(t_future));

  // Verify prediction is reasonable (not exact due to prediction error)
  auto true_state = utils::Figure8Trajectory(t_future);
  EXPECT_NEAR(pose_estimate.pose.pose.position.x,
              true_state[core::StateIndex::Position::X], 0.5);
  EXPECT_NEAR(pose_estimate.pose.pose.position.y,
              true_state[core::StateIndex::Position::Y], 0.5);
  EXPECT_NEAR(pose_estimate.pose.pose.position.z,
              true_state[core::StateIndex::Position::Z], 0.5);
}

// Test expected measurement
TEST_F(FilterWrapperTest, ExpectedMeasurement) {
  // Initialize with first position measurement
  double t = 0.0;
  auto pos_msg = createPositionMsg(t);
  ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg));

  // Get expected position measurement
  auto expected_pos = wrapper_->getExpectedPosition(position_sensor_id_);

  // Should be close to actual input measurement
  EXPECT_NEAR(expected_pos.pose.pose.position.x, pos_msg.point.x, 1e-4);
  EXPECT_NEAR(expected_pos.pose.pose.position.y, pos_msg.point.y, 1e-4);
  EXPECT_NEAR(expected_pos.pose.pose.position.z, pos_msg.point.z, 1e-4);
}

TEST_F(FilterWrapperTest, ProcessAndEstimateTrajectory) {
  setSensorTransforms();
  ASSERT_NEAR(wrapper_->GetCurrentTime(), std::numeric_limits<double>::lowest(), 1e-9);

  // Start at t=0 with no initial offset for stability
  double t = 0.0;
  double dt = 0.01;  // 100Hz

  // Vectors to track error over time
  std::vector<double> pos_errors;
  std::vector<double> vel_errors;
  std::vector<double> quat_errors;

  // Process measurements along the Figure-8 trajectory
  for (int i = 1; i <= 100; i++) {  // Use 100 iterations (1 second)
    t += dt;

    // Create measurements with exact timing
    auto pos_msg = createPositionMsg(t);
    auto pose_msg = createPoseMsg(t);
    auto vel_msg = createVelocityMsg(t);
    auto imu_msg = createImuMsg(t);

    // Process all measurements in a fixed order
    ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg));
    ASSERT_TRUE(wrapper_->processPose(pose_sensor_id_, pose_msg));
    ASSERT_TRUE(wrapper_->processBodyVelocity(velocity_sensor_id_, vel_msg));
    ASSERT_TRUE(wrapper_->processImu(imu_sensor_id_, imu_msg));

  ASSERT_NEAR(wrapper_->GetCurrentTime(), t, 1e-9);

    // Get expected state at this time
    auto true_state = utils::Figure8Trajectory(t);

    // Get state estimates at the exact time
    auto pose_estimate = wrapper_->getPoseEstimate(wrapper_->doubleTimeToRosTime(t));
    auto velocity_estimate = wrapper_->getVelocityEstimate(wrapper_->doubleTimeToRosTime(t));

    // Extract position, velocity, and orientation
    Eigen::Vector3d position(
      pose_estimate.pose.pose.position.x,
      pose_estimate.pose.pose.position.y,
      pose_estimate.pose.pose.position.z
    );

    Eigen::Vector3d velocity(
      velocity_estimate.twist.twist.linear.x,
      velocity_estimate.twist.twist.linear.y,
      velocity_estimate.twist.twist.linear.z
    );

    Eigen::Quaterniond quat_estimate(
      pose_estimate.pose.pose.orientation.w,
      pose_estimate.pose.pose.orientation.x,
      pose_estimate.pose.pose.orientation.y,
      pose_estimate.pose.pose.orientation.z
    );

    // Calculate errors
    double pos_error = (position - true_state.segment<3>(core::StateIndex::Position::Begin())).norm();
    double vel_error = (velocity - true_state.segment<3>(core::StateIndex::LinearVelocity::Begin())).norm();

    Eigen::Quaterniond quat_true(
      true_state[core::StateIndex::Quaternion::W],
      true_state[core::StateIndex::Quaternion::X],
      true_state[core::StateIndex::Quaternion::Y],
      true_state[core::StateIndex::Quaternion::Z]
    );

    double quat_error = quat_estimate.angularDistance(quat_true);

    // Store errors
    pos_errors.push_back(pos_error);
    vel_errors.push_back(vel_error);
    quat_errors.push_back(quat_error);

    // Record errors at specific iterations (like the core test)
    if (i % 20 == 0 || i == 100) {
      RecordProperty("iteration_" + std::to_string(i) + "_pos_error", pos_error);
      RecordProperty("iteration_" + std::to_string(i) + "_vel_error", vel_error);
      RecordProperty("iteration_" + std::to_string(i) + "_quat_error", quat_error);
    }
  }

  // Check final errors with tolerances similar to the core test
  double final_pos_error = pos_errors.back();
  double final_vel_error = vel_errors.back();
  double final_quat_error = quat_errors.back();

  RecordProperty("final_pos_error", final_pos_error);
  RecordProperty("final_vel_error", final_vel_error);
  RecordProperty("final_quat_error", final_quat_error);

  // Use tolerances from the core test that passes
  EXPECT_LT(final_pos_error, 0.003) << "Position did not converge";
  EXPECT_LT(final_vel_error, 0.06) << "Velocity did not converge";
  EXPECT_LT(final_quat_error, 0.4) << "Orientation did not converge";
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
