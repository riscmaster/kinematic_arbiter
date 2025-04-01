
.. _program_listing_file_include_kinematic_arbiter_models_rigid_body_state_model.hpp:

Program Listing for File rigid_body_state_model.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_models_rigid_body_state_model.hpp>` (``include/kinematic_arbiter/models/rigid_body_state_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include "kinematic_arbiter/core/state_model_interface.hpp"
   #include "kinematic_arbiter/core/state_index.hpp"
   #include <Eigen/Dense>
   #include <Eigen/Geometry>
   #include <cmath>
   #include <limits>

   namespace kinematic_arbiter {
   namespace models {
   namespace {
     const double kLambda = 46.05;
   }


   class RigidBodyStateModel : public core::StateModelInterface {
   public:
     // Use convenient matrix/vector aliases for 3D/4D operations
     using Matrix3d = Eigen::Matrix<double, 3, 3>;
     using Vector3d = Eigen::Vector3d;
     using Vector4d = Eigen::Vector4d;
     using Matrix4d = Eigen::Matrix<double, 4, 4>;

     // Type alias for cleaner access to state indices
     using SIdx = core::StateIndex;

     explicit RigidBodyStateModel(const Params& params = Params())
       : core::StateModelInterface(params) {}

       StateVector PredictState(const StateVector& current_state, double time_step) const override {
       StateVector new_states = current_state;

       if (fabs(time_step) <= std::numeric_limits<double>::epsilon()) {
         return new_states;
       }

       // Extract quaternion and normalize
       Eigen::Quaterniond orientation(
         current_state[SIdx::Quaternion::W],
         current_state[SIdx::Quaternion::X],
         current_state[SIdx::Quaternion::Y],
         current_state[SIdx::Quaternion::Z]
       );
       orientation.normalize();

       // Get rotation matrix
       Matrix3d rotation_matrix_b_to_w = orientation.toRotationMatrix();

       // Extract velocity and acceleration components
       const Vector3d linear_velocity =
         current_state.segment<3>(SIdx::LinearVelocity::Begin());

       const Vector3d angular_velocity =
         current_state.segment<3>(SIdx::AngularVelocity::Begin());

       const Vector3d linear_acceleration =
         current_state.segment<3>(SIdx::LinearAcceleration::Begin());

       const Vector3d angular_acceleration =
         current_state.segment<3>(SIdx::AngularAcceleration::Begin());

       // Linear XYZ Position Prediction Model
       new_states.segment<3>(SIdx::Position::Begin()) +=
         rotation_matrix_b_to_w * ((linear_velocity + time_step * 0.5 * linear_acceleration) * time_step);

       // Angular Position (Quaternion) Prediction Model
       // Use quaternion kinematics with angular acceleration: q(t+dt) = exp(0.5*ω*dt + 0.25*ω̇*dt²) * q(t)
       Eigen::Quaterniond delta_q;

       // Calculate combined angular motion vector v = 0.5*ω*dt + 0.25*ω̇*dt²
       // This is the same vector used in the quaternion update: q(t+dt) = exp(v) * q(t)
       double t = time_step;
       Vector3d combined_angular_motion = angular_velocity * t + 0.5 * angular_acceleration * t * t;
       double motion_norm = combined_angular_motion.norm();

       if (motion_norm > std::numeric_limits<double>::min()) {
         Vector3d axis = combined_angular_motion / motion_norm;
         double angle = motion_norm;
         delta_q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
       } else {
         delta_q = Eigen::Quaterniond::Identity();
       }

       Eigen::Quaterniond new_quaternion = delta_q * orientation;
       new_quaternion.normalize();

       new_states[SIdx::Quaternion::W] = new_quaternion.w();
       new_states[SIdx::Quaternion::X] = new_quaternion.x();
       new_states[SIdx::Quaternion::Y] = new_quaternion.y();
       new_states[SIdx::Quaternion::Z] = new_quaternion.z();

       // Velocity prediction (integrate acceleration)
       new_states.segment<3>(SIdx::LinearVelocity::Begin()) += time_step * linear_acceleration;
       new_states.segment<3>(SIdx::AngularVelocity::Begin()) += time_step * angular_acceleration;
       // Acceleration is modelled as exponential decay: a(t+dt) = a(t) * exp(-lambda * dt)
       // With lambda = 46.05, acceleration decays to near zero within 0.1s
       new_states.segment<3>(SIdx::LinearAcceleration::Begin()) =
           linear_acceleration * std::exp(-kLambda * time_step);
       new_states.segment<3>(SIdx::AngularAcceleration::Begin()) =
           angular_acceleration * std::exp(-kLambda * time_step);

       return new_states;
     }

     StateMatrix GetTransitionMatrix(const StateVector& current_state, double time_step) const override {
       if (fabs(time_step) <= std::numeric_limits<double>::epsilon()) {
         return StateMatrix::Identity();
       }

       StateMatrix jacobian = StateMatrix::Zero();

       // Extract quaternion and normalize
       Eigen::Quaterniond orientation(
         current_state[SIdx::Quaternion::W],
         current_state[SIdx::Quaternion::X],
         current_state[SIdx::Quaternion::Y],
         current_state[SIdx::Quaternion::Z]
       );
       orientation.normalize();

       // Get rotation matrix (world to body frame)
       Matrix3d rotation_matrix_w_to_b = orientation.toRotationMatrix().transpose();

       // Position block - identity for position elements
       jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::Position::Begin()) =
         Matrix3d::Identity();

       // Position wrt linear velocity
       jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::LinearVelocity::Begin()) =
         rotation_matrix_w_to_b * (time_step * Matrix3d::Identity());

       // Position wrt linear acceleration
       jacobian.block<3,3>(SIdx::Position::Begin(), SIdx::LinearAcceleration::Begin()) =
         rotation_matrix_w_to_b * (time_step * time_step * 0.5 * Matrix3d::Identity());

       // Remove coupling between linear and angular position in the jacobian by
       // setting to zero. Derivation of partial of position with respect to the
       // rotation in quaternion form can be found here:
       // http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
       jacobian.block<3,4>(SIdx::Position::Begin(), SIdx::Quaternion::Begin()) =
           Eigen::Matrix<double, 3, 4>::Zero();

       // Quaternion self-propagation
       jacobian.block<4,4>(SIdx::Quaternion::Begin(), SIdx::Quaternion::Begin()) =
           Matrix4d::Identity();

       // Extract angular velocity and acceleration components
       const Vector3d angular_velocity =
         current_state.segment<3>(SIdx::AngularVelocity::Begin());

       const Vector3d angular_acceleration =
         current_state.segment<3>(SIdx::AngularAcceleration::Begin());

       // Calculate combined angular motion vector v = 0.5*ω*dt + 0.25*ω̇*dt²
       // This is the same vector used in the quaternion update: q(t+dt) = exp(v) * q(t)
       double t = time_step;
       Vector3d combined_angular_motion = angular_velocity * t + 0.5 * angular_acceleration * t * t;
       double motion_norm = combined_angular_motion.norm();

       // Jacobian terms for quaternion derivatives with respect to angular velocity
       // ∂q/∂ω ≈ (t/2)*I - higher order correction terms
       Matrix3d velocity_term = (t/2.0) * Matrix3d::Identity();

       // Jacobian terms for quaternion derivatives with respect to angular acceleration
       // ∂q/∂ω̇ ≈ (t²/4)*I - higher order correction terms
       Matrix3d accel_term = (t*t/4.0) * Matrix3d::Identity();

       // Apply correction terms when motion is non-zero
       if (motion_norm > std::numeric_limits<double>::min()) {
         // Compute outer product correction based on combined motion
         // This comes from the derivative of the exponential map
         Matrix3d outer_product_correction = (combined_angular_motion * combined_angular_motion.transpose()) /
                                            (motion_norm * motion_norm);

         // Scale for velocity: (t³*||v||)/16 * (v*v^T/||v||²)
         Matrix3d velocity_correction = (t*t*t * motion_norm / 16.0) * outer_product_correction;

         // Scale for acceleration: (t⁴*||v||)/32 * (v*v^T/||v||²)
         Matrix3d accel_correction = (t*t*t*t * motion_norm / 32.0) * outer_product_correction;

         velocity_term -= velocity_correction;
         accel_term -= accel_correction;
       }

       // Create quaternion derivatives for angular velocity components
       // Format: [scalar_part, vector_part_x, vector_part_y, vector_part_z]
       Vector4d dq_dx(
           -0.25 * t * combined_angular_motion[0],  // Scalar part derivative
           velocity_term(0,0), velocity_term(0,1), velocity_term(0,2)  // Vector part derivatives
       );
       Vector4d dq_dy(
           -0.25 * t * combined_angular_motion[1],
           velocity_term(1,0), velocity_term(1,1), velocity_term(1,2)
       );
       Vector4d dq_dz(
           -0.25 * t * combined_angular_motion[2],
           velocity_term(2,0), velocity_term(2,1), velocity_term(2,2)
       );

       // Set the Jacobian block for quaternion wrt angular velocity
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::X) = dq_dx;
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::Y) = dq_dy;
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularVelocity::Z) = dq_dz;

       // Create quaternion derivatives for angular acceleration components
       // Format: [scalar_part, vector_part_x, vector_part_y, vector_part_z]
       Vector4d dq_dax(
           -0.125 * t * t * combined_angular_motion[0],  // Scalar part derivative
           accel_term(0,0), accel_term(0,1), accel_term(0,2)  // Vector part derivatives
       );
       Vector4d dq_day(
           -0.125 * t * t * combined_angular_motion[1],
           accel_term(1,0), accel_term(1,1), accel_term(1,2)
       );
       Vector4d dq_daz(
           -0.125 * t * t * combined_angular_motion[2],
           accel_term(2,0), accel_term(2,1), accel_term(2,2)
       );

       // Set the Jacobian block for quaternion wrt angular acceleration
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularAcceleration::X) = dq_dax;
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularAcceleration::Y) = dq_day;
       jacobian.block<4,1>(SIdx::Quaternion::Begin(), SIdx::AngularAcceleration::Z) = dq_daz;

       // Velocity blocks - identity for velocity elements
       jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearVelocity::Begin()) =
         Matrix3d::Identity();

       // Linear velocity wrt linear acceleration
       jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearAcceleration::Begin()) =
         time_step * Matrix3d::Identity();

       // Angular velocity blocks - identity for angular velocity elements
       jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularVelocity::Begin()) =
         Matrix3d::Identity();

       // Angular velocity wrt angular acceleration
       jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularAcceleration::Begin()) =
         time_step * Matrix3d::Identity();

       // Acceleration blocks are modelled as zero (no change)

       // Velocity blocks - identity for velocity elements
       jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearVelocity::Begin()) =
         Matrix3d::Identity();

       // Linear velocity wrt linear acceleration
       jacobian.block<3,3>(SIdx::LinearVelocity::Begin(), SIdx::LinearAcceleration::Begin()) =
         time_step * Matrix3d::Identity();

       // Angular velocity blocks - identity for angular velocity elements
       jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularVelocity::Begin()) =
         Matrix3d::Identity();

       // Angular velocity wrt angular acceleration
       jacobian.block<3,3>(SIdx::AngularVelocity::Begin(), SIdx::AngularAcceleration::Begin()) =
         time_step * Matrix3d::Identity();

       jacobian.block<3,3>(SIdx::AngularAcceleration::Begin(), SIdx::AngularAcceleration::Begin()) =
         std::exp(-kLambda * time_step) * Matrix3d::Identity();

       jacobian.block<3,3>(SIdx::LinearAcceleration::Begin(), SIdx::LinearAcceleration::Begin()) =
         std::exp(-kLambda * time_step) * Matrix3d::Identity();

       return jacobian;
     }
   };

   } // namespace models
   } // namespace kinematic_arbiter
