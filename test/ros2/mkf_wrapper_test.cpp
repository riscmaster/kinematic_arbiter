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

    // Set a reasonable delay window
    wrapper_->setMaxDelayWindow(kMaxDelayWindow);  // 1 second
  }

  void TearDown() override {
    // Cleanup
    wrapper_.reset();
  }

  // Helper to create a timestamp
  rclcpp::Time createTime(double seconds) {
    int32_t sec = static_cast<int32_t>(seconds);
    uint32_t nanosec = static_cast<uint32_t>((seconds - sec) * 1e9);
    return rclcpp::Time(sec, nanosec);
  }

  // Create position message using expected measurement
  geometry_msgs::msg::PointStamped createPositionMsg(double t) {
    MeasurementModelInterface::DynamicVector measurement;
    EXPECT_TRUE(wrapper_->getExpectedMeasurementByID(position_sensor_id_,
      measurement, utils::Figure8Trajectory(t)));

    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = createTime(t);
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
    msg.header.stamp = createTime(t);
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
    msg.header.stamp = createTime(t);
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
    msg.header.stamp = createTime(t);
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
  wrapper_->predictTo(createTime(t_future));

  // Get the predicted state
  auto pose_estimate = wrapper_->getPoseEstimate(createTime(t_future));

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
  ASSERT_DOUBLE_EQ(wrapper_->GetCurrentTime(), std::numeric_limits<double>::lowest());

  // Start at t=0, but with initial offset to test convergence
  double t = 0.0;
  double dt = 0.01;  // 100Hz
  double time_step_jitter = 0.03;  // 1ms jitter

  // Process first measurement with intentionally wrong position to test convergence
  auto pos_msg = createPositionMsg(t);
  // Add an initial error to test convergence
  pos_msg.point.x += 1.0;  // 1m offset in X
  pos_msg.point.y += 0.5;  // 0.5m offset in Y
  ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg));

  // Random number generator for time jitter
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> jitter_dist(-2*time_step_jitter, 0.0); // No future measurements

  // Process measurements along the Figure-8 trajectory
  for (int i = 1; i < 100; i++) {  // 10 seconds of data
    t += dt;

    // Add random jitter to each sensor's timestamp
    double t_pos = t + jitter_dist(gen);
    double t_pose = t + jitter_dist(gen);
    double t_vel = t + jitter_dist(gen);
    double t_imu = t + jitter_dist(gen);

    // Create measurements from the true trajectory with perturbed timestamps
    auto pos_msg = createPositionMsg(t_pos);
    auto pose_msg = createPoseMsg(t_pose);
    auto vel_msg = createVelocityMsg(t_vel);
    auto imu_msg = createImuMsg(t_imu);
    // Process all measurements
    ASSERT_TRUE(wrapper_->processPose(pose_sensor_id_, pose_msg))
        << "Failed to process pose measurement at time " << t_pose;
    ASSERT_TRUE(wrapper_->processPosition(position_sensor_id_, pos_msg))
        << "Failed to process position measurement at time " << t_pos;
    ASSERT_TRUE(wrapper_->processBodyVelocity(velocity_sensor_id_, vel_msg))
        << "Failed to process velocity measurement at time " << t_vel;
    ASSERT_TRUE(wrapper_->processImu(imu_sensor_id_, imu_msg))
        << "Failed to process IMU measurement at time " << t_imu;

    ASSERT_DOUBLE_EQ(wrapper_->GetCurrentTime(), std::max({t_pos, t_pose, t_vel, t_imu}));

    // Get expected state at this time
    auto true_state = utils::Figure8Trajectory(t);

    // Get state estimates
    auto pose_estimate = wrapper_->getPoseEstimate(createTime(t));
    auto velocity_estimate = wrapper_->getVelocityEstimate(createTime(t));

    Eigen::Vector3d position(pose_estimate.pose.pose.position.x, pose_estimate.pose.pose.position.y, pose_estimate.pose.pose.position.z);
    Eigen::Vector3d velocity(velocity_estimate.twist.twist.linear.x, velocity_estimate.twist.twist.linear.y, velocity_estimate.twist.twist.linear.z);
    // Calculate position and velocity errors
   double pos_error = (position - true_state.segment<3>(core::StateIndex::Position::Begin())).norm();
   double vel_error = (velocity - true_state.segment<3>(core::StateIndex::LinearVelocity::Begin())).norm();

    Eigen::Quaterniond quat_estimate(pose_estimate.pose.pose.orientation.w, pose_estimate.pose.pose.orientation.x, pose_estimate.pose.pose.orientation.y, pose_estimate.pose.pose.orientation.z);
    Eigen::Quaterniond quat_true(true_state[core::StateIndex::Quaternion::W], true_state[core::StateIndex::Quaternion::X], true_state[core::StateIndex::Quaternion::Y], true_state[core::StateIndex::Quaternion::Z]);
    double quat_error = quat_estimate.angularDistance(quat_true);

    // Start with larger tolerance, decrease over time as filter converges
    double tolerance_factor = std::max(0.1, 1.0 - i * 0.01);  // Decreases from 1.0 to 0.1

    EXPECT_LT(pos_error, tolerance_factor * 0.1);
    EXPECT_LT(vel_error, tolerance_factor * 0.1);
    EXPECT_LT(quat_error, tolerance_factor * 0.1);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
