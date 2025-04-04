
.. _program_listing_file_include_kinematic_arbiter_sensors_imu_bias_estimator.hpp:

Program Listing for File imu_bias_estimator.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_sensors_imu_bias_estimator.hpp>` (``include/kinematic_arbiter/sensors/imu_bias_estimator.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #pragma once

   #include <Eigen/Core>
   #include <Eigen/Geometry>

   namespace kinematic_arbiter {
   namespace sensors {

   class ImuBiasEstimator {
   public:

     explicit ImuBiasEstimator(uint32_t window_size = 100)
         : window_size_(window_size),
           gyro_bias_(Eigen::Vector3d::Zero()),
           accel_bias_(Eigen::Vector3d::Zero()) {}

     void EstimateBiases(
         const Eigen::Vector3d& measured_gyro,
         const Eigen::Vector3d& measured_accel,
         const Eigen::Vector3d& predicted_gyro,
         const Eigen::Vector3d& predicted_accel);

     void ResetCalibration() {
       gyro_bias_.setZero();
       accel_bias_.setZero();
     }

     const Eigen::Vector3d& GetGyroBias() const { return gyro_bias_; }

     const Eigen::Vector3d& GetAccelBias() const { return accel_bias_; }


   private:
     uint32_t window_size_;
     Eigen::Vector3d gyro_bias_;
     Eigen::Vector3d accel_bias_;
   };

   } // namespace sensors
   } // namespace kinematic_arbiter
