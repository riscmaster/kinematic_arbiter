
.. _program_listing_file_include_kinematic_arbiter_sensors_body_velocity_sensor_model.hpp:

Program Listing for File body_velocity_sensor_model.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_sensors_body_velocity_sensor_model.hpp>` (``include/kinematic_arbiter/sensors/body_velocity_sensor_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include "kinematic_arbiter/core/measurement_model_interface.hpp"
   #include "kinematic_arbiter/core/state_index.hpp"
   #include "kinematic_arbiter/core/sensor_types.hpp"

   namespace kinematic_arbiter {
   namespace sensors {

   class BodyVelocitySensorModel : public core::MeasurementModelInterface {
   public:
     // Type definitions for clarity
     using Base = core::MeasurementModelInterface;
     static constexpr int kMeasurementDimension = core::MeasurementDimension<core::SensorType::BodyVelocity>::value;
     using StateVector = typename Base::StateVector;
     using Vector = Eigen::Matrix<double, kMeasurementDimension, 1>;
     using Jacobian = Eigen::Matrix<double, kMeasurementDimension, core::StateIndex::kFullStateSize>;
     using Covariance = Eigen::Matrix<double, kMeasurementDimension, kMeasurementDimension>;
     using StateFlags = typename Base::StateFlags;

     struct MeasurementIndex {
       // Linear velocity indices
       static constexpr int VX = 0;
       static constexpr int VY = 1;
       static constexpr int VZ = 2;

       // Angular velocity indices
       static constexpr int WX = 3;
       static constexpr int WY = 4;
       static constexpr int WZ = 5;
     };

     explicit BodyVelocitySensorModel(
         const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
         const ValidationParams& params = ValidationParams())
       : Base(core::SensorType::BodyVelocity, sensor_pose_in_body_frame, params, Covariance::Identity()) {}

       void reset() override {
         measurement_covariance_ = Covariance::Identity();
       }

     DynamicVector PredictMeasurement(const StateVector& state) const override {
       Vector predicted_measurement = Vector::Zero();

       // Extract linear velocity from state
       Eigen::Vector3d body_lin_vel = state.segment<3>(core::StateIndex::LinearVelocity::X);

       // Extract angular velocity from state
       Eigen::Vector3d body_ang_vel = state.segment<3>(core::StateIndex::AngularVelocity::X);

       // Extract the sensor-to-body transform components
       Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();
       Eigen::Matrix3d rot_b_s = sensor_pose_in_body_frame_.rotation();

       // Calculate the skew-symmetric matrix for the cross product: ω × r
       auto skew_matrix = [](const Eigen::Vector3d& v) -> Eigen::Matrix3d {
         Eigen::Matrix3d skew;
         skew << 0, -v(2), v(1),
                 v(2), 0, -v(0),
                 -v(1), v(0), 0;
         return skew;
       };

       // Calculate linear velocity at the sensor location in body frame
       // v_sensor = v_body + ω × r, where r is the position of the sensor in the body frame
       Eigen::Vector3d body_sensor_lin_vel = body_lin_vel + skew_matrix(body_ang_vel) * trans_b_s;

       // Transform velocities to sensor frame
       Eigen::Vector3d sensor_lin_vel = rot_b_s.transpose() * body_sensor_lin_vel;
       Eigen::Vector3d sensor_ang_vel = rot_b_s.transpose() * body_ang_vel;

       // Fill measurement vector
       predicted_measurement.segment<3>(MeasurementIndex::VX) = sensor_lin_vel;
       predicted_measurement.segment<3>(MeasurementIndex::WX) = sensor_ang_vel;

       return predicted_measurement;
     }

     DynamicJacobian GetMeasurementJacobian(const StateVector&) const override {
       Jacobian jacobian = Jacobian::Zero();

       // Extract transform parameters
       Eigen::Vector3d sensor_offset = sensor_pose_in_body_frame_.translation();
       Eigen::Matrix3d body_to_sensor_rotation = sensor_pose_in_body_frame_.rotation().transpose();

       // ===== LINEAR VELOCITY JACOBIAN =====

       // Linear velocity w.r.t. linear velocity - simple rotation from body to sensor frame
       jacobian.block<3, 3>(MeasurementIndex::VX, core::StateIndex::LinearVelocity::X) = body_to_sensor_rotation;

       // Linear velocity w.r.t. angular velocity - lever arm effect
       // The lever arm effect can be represented by partial derivatives of the cross product
       // These derivatives have a standard form for cross products

       // Define cross product derivatives in a more compact way
       // Following standard notation where [r×] is the skew-symmetric matrix of vector r
       // For a sensor at position r, the derivatives of ω×r with respect to each ω component
       // correspond to the columns of a skew matrix of r

       // Create skew-symmetric matrix of the sensor offset
       Eigen::Matrix3d skew_offset = Eigen::Matrix3d::Zero();
       skew_offset <<
           0,              -sensor_offset.z(),  sensor_offset.y(),
           sensor_offset.z(),  0,              -sensor_offset.x(),
          -sensor_offset.y(),  sensor_offset.x(),  0;

       // Each column of this matrix represents the lever arm effect for one angular velocity component
       // Extract columns for better readability
       Eigen::Vector3d lever_arm_effect_wx = -skew_offset.col(0);  // Effect of ω_x
       Eigen::Vector3d lever_arm_effect_wy = -skew_offset.col(1);  // Effect of ω_y
       Eigen::Vector3d lever_arm_effect_wz = -skew_offset.col(2);  // Effect of ω_z

       // Apply body-to-sensor rotation and fill Jacobian
       jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::X) =
           body_to_sensor_rotation * lever_arm_effect_wx;
       jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::Y) =
           body_to_sensor_rotation * lever_arm_effect_wy;
       jacobian.block<3, 1>(MeasurementIndex::VX, core::StateIndex::AngularVelocity::Z) =
           body_to_sensor_rotation * lever_arm_effect_wz;

       // ===== ANGULAR VELOCITY JACOBIAN =====

       // Angular velocity w.r.t. angular velocity - simple rotation from body to sensor frame
       jacobian.block<3, 3>(MeasurementIndex::WX, core::StateIndex::AngularVelocity::X) = body_to_sensor_rotation;

       return jacobian;
     }

     StateFlags GetInitializableStates() const override {
       StateFlags flags = StateFlags::Zero();

       // Body velocity sensor can initialize all velocity states
       flags[core::StateIndex::LinearVelocity::X] = true;
       flags[core::StateIndex::LinearVelocity::Y] = true;
       flags[core::StateIndex::LinearVelocity::Z] = true;
       flags[core::StateIndex::AngularVelocity::X] = true;
       flags[core::StateIndex::AngularVelocity::Y] = true;
       flags[core::StateIndex::AngularVelocity::Z] = true;

       return flags;
     }

     StateFlags InitializeState(
         const DynamicVector& measurement,
         const StateFlags&,
         StateVector& state,
         StateCovariance& covariance) const override {
       ValidateMeasurementSize(measurement);

       StateFlags initialized_states = StateFlags::Zero();

       // Extract measurement components in sensor frame
       const Eigen::Vector3d sensor_lin_vel = measurement.segment<3>(0);
       const Eigen::Vector3d sensor_ang_vel = measurement.segment<3>(3);

         // Extract the sensor-to-body transform components
         const Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();
         const Eigen::Matrix3d rot_b_s = sensor_pose_in_body_frame_.rotation();

         // Calculate the skew-symmetric matrix for the cross product: ω × r
         auto skew_matrix = [](const Eigen::Vector3d& v) -> Eigen::Matrix3d {
           Eigen::Matrix3d skew;
           skew << 0, -v(2), v(1),
                   v(2), 0, -v(0),
                   -v(1), v(0), 0;
           return skew;
         };

         // First, compute body angular velocity
         const Eigen::Vector3d body_ang_vel = rot_b_s * sensor_ang_vel;

         // Convert sensor linear velocity to body frame
         const Eigen::Vector3d sensor_lin_vel_in_body = rot_b_s * sensor_lin_vel;

         // Account for the lever arm effect
         const Eigen::Vector3d lever_arm_effect = skew_matrix(body_ang_vel) * trans_b_s;

         // Compute body linear velocity
         const Eigen::Vector3d body_lin_vel = sensor_lin_vel_in_body - lever_arm_effect;

         // Initialize state vector
         state.segment<3>(core::StateIndex::LinearVelocity::Begin()) = body_lin_vel;
         state.segment<3>(core::StateIndex::AngularVelocity::Begin()) = body_ang_vel;

         // Set all velocity states as initialized
         initialized_states[core::StateIndex::LinearVelocity::X] = true;
         initialized_states[core::StateIndex::LinearVelocity::Y] = true;
         initialized_states[core::StateIndex::LinearVelocity::Z] = true;
         initialized_states[core::StateIndex::AngularVelocity::X] = true;
         initialized_states[core::StateIndex::AngularVelocity::Y] = true;
         initialized_states[core::StateIndex::AngularVelocity::Z] = true;

         // Transform covariance from sensor to body frame
         // For angular velocity (straightforward rotation)
         Eigen::Matrix3d ang_vel_cov = rot_b_s *
             measurement_covariance_.block<3, 3>(3, 3) *
             rot_b_s.transpose();

         // For linear velocity (approximate - ignoring cross-terms with angular velocity)
         Eigen::Matrix3d lin_vel_cov = rot_b_s *
             measurement_covariance_.block<3, 3>(0, 0) *
             rot_b_s.transpose();

         // Set covariance blocks in state covariance
         covariance.block<3, 3>(
             core::StateIndex::LinearVelocity::Begin(),
             core::StateIndex::LinearVelocity::Begin()) = lin_vel_cov;

         covariance.block<3, 3>(
             core::StateIndex::AngularVelocity::Begin(),
             core::StateIndex::AngularVelocity::Begin()) = ang_vel_cov;


       return initialized_states;
     }


   };

   } // namespace sensors
   } // namespace kinematic_arbiter
