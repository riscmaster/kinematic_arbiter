
.. _program_listing_file_include_kinematic_arbiter_sensors_imu_sensor_model.hpp:

Program Listing for File imu_sensor_model.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_sensors_imu_sensor_model.hpp>` (``include/kinematic_arbiter/sensors/imu_sensor_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <Eigen/Core>
   #include <Eigen/Geometry>

   #include "kinematic_arbiter/core/measurement_model_interface.hpp"
   #include "kinematic_arbiter/core/state_index.hpp"
   #include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"
   #include "kinematic_arbiter/core/sensor_types.hpp"

   namespace kinematic_arbiter {
   namespace sensors {
   namespace test {
   class ImuStationaryTest;
   }  // namespace test

   // Move Config struct outside the class
   struct ImuSensorConfig {
     uint32_t bias_estimation_window_size = 1000;
     bool calibration_enabled = false;
     double stationary_confidence_threshold = 0.01;
   };

   class ImuSensorModel : public core::MeasurementModelInterface {
   public:
     // Type definitions for clarity
     using Base = core::MeasurementModelInterface;
     static constexpr int kMeasurementDimension = core::MeasurementDimension<core::SensorType::Imu>::value;
     using StateVector = typename Base::StateVector;
     using StateCovariance = typename Base::StateCovariance;
     using Vector = Eigen::Matrix<double, kMeasurementDimension, 1>;
     using Jacobian = Eigen::Matrix<double, kMeasurementDimension, core::StateIndex::kFullStateSize>;
     using Covariance = Eigen::Matrix<double, kMeasurementDimension, kMeasurementDimension>;
     using StateFlags = typename Base::StateFlags;

     struct MeasurementIndex {
       // Gyroscope indices
       static constexpr int GX = 0;
       static constexpr int GY = 1;
       static constexpr int GZ = 2;

       // Accelerometer indices
       static constexpr int AX = 3;
       static constexpr int AY = 4;
       static constexpr int AZ = 5;
     };

     explicit ImuSensorModel(
         const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
         const ImuSensorConfig& config = ImuSensorConfig(),
         const ValidationParams& params = ValidationParams())
       : Base(core::SensorType::Imu, sensor_pose_in_body_frame, params, Covariance::Identity()),
         bias_estimator_(config.bias_estimation_window_size),
         config_(config) {
           this->can_predict_input_accelerations_ = false;
         }

       void reset() override {
         measurement_covariance_ = Covariance::Identity();
       }

     DynamicVector PredictMeasurement(const StateVector& state) const override;

     bool UpdateBiasEstimates(
         const StateVector& state,
         const Eigen::MatrixXd& state_covariance,
         const Vector& raw_measurement);

     DynamicJacobian GetMeasurementJacobian(const StateVector& state) const override;

     Eigen::Matrix<double, 6, 1> GetPredictionModelInputs(
         const StateVector& state_before_prediction,
         const StateCovariance& state_covariance_before_prediction,
         const DynamicVector& measurement_after_prediction,
         double dt) const override;

     void EnableCalibration(bool enable) {
       config_.calibration_enabled = enable;
     }

     void SetConfig(const ImuSensorConfig& config) {
       config_ = config;
     }

     double GetGravity() const {
       return kGravity;
     }

     StateFlags GetInitializableStates() const override;

     StateFlags InitializeState(
         const DynamicVector& measurement,
         const StateFlags&,
         StateVector& state,
         StateCovariance& covariance) const override;

     friend class kinematic_arbiter::sensors::test::ImuStationaryTest;

   private:
     ImuBiasEstimator bias_estimator_;
     ImuSensorConfig config_;

     // Gravity constants moved here from global scope
     static constexpr double kGravity = 9.80665;
     static constexpr double kGravityVariance = 0.012 * 0.012;  // Variance in gravity estimate

     bool IsStationary(
         const StateVector& state,
         const StateCovariance& state_covariance,
         const Vector& measurement) const;
   };

   } // namespace sensors
   } // namespace kinematic_arbiter
