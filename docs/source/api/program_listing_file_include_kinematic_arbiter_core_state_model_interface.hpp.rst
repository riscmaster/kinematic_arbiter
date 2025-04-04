
.. _program_listing_file_include_kinematic_arbiter_core_state_model_interface.hpp:

Program Listing for File state_model_interface.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_core_state_model_interface.hpp>` (``include/kinematic_arbiter/core/state_model_interface.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <Eigen/Dense>
   #include <cmath>
   #include "kinematic_arbiter/core/state_index.hpp"

   namespace kinematic_arbiter {
   namespace core {

   class StateModelInterface {
   public:
     // Type definitions
     static constexpr int StateSize = StateIndex::kFullStateSize;
     using StateVector = Eigen::Matrix<double, StateSize, 1>;
     using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
     using Vector3d = Eigen::Vector3d;
     struct Params {
       // Sample window size for process noise estimation (n)
       size_t process_noise_window;

       // Initial process noise values for different state components
       double position_uncertainty_per_second;
       double orientation_uncertainty_per_second;
       double linear_velocity_uncertainty_per_second;
       double angular_velocity_uncertainty_per_second;
       double linear_acceleration_uncertainty_per_second;
       double angular_acceleration_uncertainty_per_second;

       // Default constructor with initialization
       Params()
         : process_noise_window(40),
           position_uncertainty_per_second(0.01),
           orientation_uncertainty_per_second(0.01),
           linear_velocity_uncertainty_per_second(0.1),
           angular_velocity_uncertainty_per_second(0.1),
           linear_acceleration_uncertainty_per_second(1.0),
           angular_acceleration_uncertainty_per_second(1.0) {}
     };

     void reset() {
       process_noise_ = StateMatrix::Identity();
     }

     explicit StateModelInterface(const Params& params = Params())
       : params_(params),
         state_initialized_(false),
         time_since_reset_(0.0) {

       // Initialize process noise with diagonal elements based on parameters
       process_noise_ = StateMatrix::Zero();

       // Set position uncertainty (states 0-2)
       process_noise_.block<3, 3>(0, 0) =
           params_.position_uncertainty_per_second* params_.position_uncertainty_per_second* Eigen::Matrix3d::Identity() / 9.0;

       // Set orientation uncertainty (states 3-6, quaternion)
       process_noise_.block<4, 4>(3, 3) =
           params_.orientation_uncertainty_per_second * params_.orientation_uncertainty_per_second * Eigen::Matrix4d::Identity() / (16.0 * 9.0);

       // Set linear velocity uncertainty (states 7-9)
       process_noise_.block<3, 3>(7, 7) =
           params_.linear_velocity_uncertainty_per_second * params_.linear_velocity_uncertainty_per_second * Eigen::Matrix3d::Identity() / (9.0);

       // Set angular velocity uncertainty (states 10-12)
       process_noise_.block<3, 3>(10, 10) =
           params_.angular_velocity_uncertainty_per_second * params_.angular_velocity_uncertainty_per_second * Eigen::Matrix3d::Identity() / (9.0);

       // Set linear acceleration uncertainty (states 13-15)
       process_noise_.block<3, 3>(13, 13) =
           params_.linear_acceleration_uncertainty_per_second * params_.linear_acceleration_uncertainty_per_second * Eigen::Matrix3d::Identity() / (9.0);

       // Set angular acceleration uncertainty (states 16-18)
       process_noise_.block<3, 3>(16, 16) =
           params_.angular_acceleration_uncertainty_per_second * params_.angular_acceleration_uncertainty_per_second * Eigen::Matrix3d::Identity() / (9.0);
     }

     virtual ~StateModelInterface() = default;

     virtual StateVector PredictState(const StateVector& state, double dt) const = 0;


     virtual StateMatrix GetTransitionMatrix(const StateVector& state, double dt) const = 0;

     virtual StateVector PredictStateWithInputAccelerations(const StateVector& state, double dt, const Vector3d& linear_acceleration=Vector3d::Zero(), const Vector3d& angular_acceleration=Vector3d::Zero()) const
     {
       StateVector new_state = state;
       new_state.segment<3>(StateIndex::LinearAcceleration::Begin()) = linear_acceleration;
       new_state.segment<3>(StateIndex::AngularAcceleration::Begin()) = angular_acceleration;
       return PredictState(new_state, dt);
     };

     virtual StateMatrix GetProcessNoiseCovariance(double dt) const {
       return process_noise_ * fabs(dt);
     }

     void UpdateProcessNoise(
         const StateVector& a_priori_state,
         const StateVector& a_posteriori_state,
         double process_to_measurement_ratio,
         double dt) {
       // Compute state correction
       StateVector tmp_state_diff = sqrt(process_to_measurement_ratio) * (a_priori_state - a_posteriori_state).array().abs() / fabs(dt);
       StateVector bounded_diff = (tmp_state_diff.array() < kMinStateDiff).select(0.0, tmp_state_diff.array()).min(kMaxStateDiff);

       // Apply maximum bound to state differences
       state_diff_ += (bounded_diff - state_diff_) / params_.process_noise_window;

       if( (state_diff_.array() > kMinStateDiff).all() ) {

       process_noise_ += (state_diff_ * state_diff_.transpose() - process_noise_) / (params_.process_noise_window);
       }
     }

     const StateMatrix& GetProcessNoise() const {
       return process_noise_;
     }

     const Params& GetParams() const { return params_; }

   protected:
     // Threshold for numerical comparisons
     static constexpr double kMinStateDiff = 1e-6;
     static constexpr double kMaxStateDiff = 1e6;

     StateVector state_diff_ = StateVector::Zero();

     // State model parameters
     Params params_;

     // Process noise covariance
     StateMatrix process_noise_ = StateMatrix::Identity();

     // State model initialization flags
     bool state_initialized_ = false;
     double time_since_reset_ = 0.0;
   };

   } // namespace core
   } // namespace kinematic_arbiter
