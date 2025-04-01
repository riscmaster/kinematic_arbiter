
.. _program_listing_file_include_kinematic_arbiter_core_measurement_model_interface.hpp:

Program Listing for File measurement_model_interface.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file_include_kinematic_arbiter_core_measurement_model_interface.hpp>` (``include/kinematic_arbiter/core/measurement_model_interface.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   #ifndef KINEMATIC_ARBITER_CORE_MEASUREMENT_MODEL_INTERFACE_HPP_
   #define KINEMATIC_ARBITER_CORE_MEASUREMENT_MODEL_INTERFACE_HPP_

   #include <Eigen/Dense>
   #include <Eigen/Geometry>
   #include <iostream>
   #include <stdexcept>
   #include "kinematic_arbiter/core/state_index.hpp"
   #include "kinematic_arbiter/core/mediation_types.hpp"
   #include "kinematic_arbiter/core/statistical_utils.hpp"
   #include "kinematic_arbiter/core/sensor_types.hpp"

   namespace kinematic_arbiter {
   namespace core {

     namespace {
       constexpr int kMaxMeasurementDim = 7;  // Maximum measurement dimension (for Pose type)
     }

   class MeasurementModelInterface {
   public:
     // Type definitions
     static constexpr int StateSize = StateIndex::kFullStateSize;

     using StateVector = Eigen::Matrix<double, StateSize, 1>;                     // State vector x
     using StateCovariance = Eigen::Matrix<double, StateSize, StateSize>;
     using DynamicVector = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, kMaxMeasurementDim, 1>;          // Measurement vector y_k
     using DynamicCovariance = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, kMaxMeasurementDim, kMaxMeasurementDim>; // Measurement covariance R
     using DynamicJacobian = Eigen::Matrix<double, Eigen::Dynamic, StateSize, Eigen::ColMajor, kMaxMeasurementDim, StateSize>; // Measurement Jacobian C_k                        // Innovation covariance S

     // More general name for boolean state flags
     using StateFlags = Eigen::Array<bool, StateSize, 1>;

     struct ValidationParams {
       // Sample window size for adaptive estimation of R
       size_t noise_sample_window;

       // Confidence level assumption validation (0-1)
       double confidence_level;

       // Process to measurement noise ratio
       double process_to_measurement_noise_ratio;

       // Mediation action to take if validation fails
       MediationAction mediation_action;

       // Constructor with default values
       ValidationParams()
         : noise_sample_window(40),
           confidence_level(0.95),
           process_to_measurement_noise_ratio(2.0),
           mediation_action(MediationAction::ForceAccept) {}
     };

     struct MeasurementAuxData {
       DynamicVector innovation;                // y_k - C_k x_k
       DynamicJacobian jacobian;                // C_k
       DynamicCovariance innovation_covariance;  // S = C_k P_k C_k^T + R

       // Constructors
       MeasurementAuxData() = default;
       MeasurementAuxData(
           const DynamicVector& inn,
           const DynamicJacobian& jac,
           const DynamicCovariance& inn_cov)
         : innovation(inn), jacobian(jac), innovation_covariance(inn_cov) {}
     };

     explicit MeasurementModelInterface(
         SensorType type,
         const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
         const ValidationParams& params = ValidationParams(),
         const DynamicCovariance& measurement_covariance = Eigen::Matrix<double, kMaxMeasurementDim, kMaxMeasurementDim>::Identity())
       : sensor_pose_in_body_frame_(sensor_pose_in_body_frame),
         body_to_sensor_transform_(sensor_pose_in_body_frame.inverse()),
         measurement_covariance_(measurement_covariance),
         validation_params_(params),
         type_(type) {}

     virtual void reset() = 0;

     virtual ~MeasurementModelInterface() = default;

     virtual DynamicVector PredictMeasurement(const StateVector& state) const = 0;

     virtual DynamicJacobian GetMeasurementJacobian(const StateVector& state) const = 0;

     virtual Eigen::Matrix<double, 6, 1> GetPredictionModelInputs(const StateVector& , const StateCovariance& , const DynamicVector& , double) const {return Eigen::Matrix<double, 6, 1>::Zero();};

     bool CanPredictInputAccelerations() const {return can_predict_input_accelerations_;}

     const DynamicCovariance& GetMeasurementCovariance() const {
       return measurement_covariance_;
     }

     void SetValidationParams(const ValidationParams& params) {
       validation_params_ = params;
     }

     const ValidationParams& GetValidationParams() const {
       return validation_params_;
     }

     MeasurementAuxData ComputeAuxiliaryData(
         const StateVector& state,
         const StateCovariance& state_covariance,
         const DynamicVector& measurement) const {
       ValidateMeasurementSize(measurement);

       // Calculate innovation: ν = y_k - h(x_k)
       DynamicVector innovation = measurement - PredictMeasurement(state);

       // Get measurement Jacobian: C_k
       DynamicJacobian jacobian = GetMeasurementJacobian(state);

       // Calculate innovation covariance: S = C_k P_k C_k^T + R
       DynamicCovariance innovation_covariance =
           jacobian * state_covariance * jacobian.transpose() + measurement_covariance_;

       return MeasurementAuxData(innovation, jacobian, innovation_covariance);
     }

     bool ValidateAuxiliaryData(const MeasurementAuxData& aux_data) const {
       // Check innovation and jacobian for NaN/Inf
       if (!aux_data.innovation.allFinite() || !aux_data.jacobian.allFinite()) {
         std::cerr << "Invalid innovation or jacobian from " << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       // Check innovation covariance for NaN/Inf
       if (!aux_data.innovation_covariance.allFinite()) {
         std::cerr << "Innovation covariance contains NaN/Inf from " << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       // Check if innovation covariance is well-conditioned
       Eigen::JacobiSVD<DynamicCovariance> svd(aux_data.innovation_covariance);
       double condition_number = svd.singularValues()(0) /
                                std::max(svd.singularValues()(svd.singularValues().size() - 1), 1e-12);

       const double kConditionThreshold = 1e12;
       if (condition_number > kConditionThreshold) {
         std::cerr << "Ill-conditioned innovation covariance (cond=" << condition_number
                   << ") from " << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       // Check if LDLT decomposition succeeds (positive definite check)
       Eigen::LDLT<DynamicCovariance> ldlt(aux_data.innovation_covariance);
       if (ldlt.info() != Eigen::Success) {
         std::cerr << "Innovation covariance not positive definite from "
                   << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       // All checks passed
       return true;
     }

     bool ValidateAndMediate(
         const StateVector& state,
         const StateCovariance& state_covariance,
         const double& measurement_timestamp,
         const DynamicVector& measurement,
         MeasurementAuxData& aux_data) {

       previous_measurement_data_ = MeasurementData(
         measurement_timestamp,
         measurement,
         measurement_covariance_
       );

       // Compute auxiliary data
       aux_data = ComputeAuxiliaryData(state, state_covariance, measurement);

       // Validate auxiliary data
       if (!ValidateAuxiliaryData(aux_data)) {
         return false;
       }

       // Skip innovation test for forced accept
       if (validation_params_.mediation_action == MediationAction::ForceAccept) {
         UpdateCovariance(aux_data.innovation);
         previous_measurement_data_.covariance = measurement_covariance_;
         return true;
       }

       // Mahalanobis distance: d = ν^T S^-1 ν
       double chi_squared_term = aux_data.innovation.transpose() *
                           aux_data.innovation_covariance.ldlt().solve(
                           aux_data.innovation);

       // Determine threshold for chi-squared test
       double threshold = utils::CalculateChiSquareCriticalValueNDof(
             aux_data.innovation.rows()-1, validation_params_.confidence_level);

       // Check if measurement passes validation
       if (chi_squared_term < threshold) {
         UpdateCovariance(aux_data.innovation);
         previous_measurement_data_.covariance = measurement_covariance_;
         return true;
       }

       // Measurement Apply Mediation Action
       if (validation_params_.mediation_action == MediationAction::AdjustCovariance) {
         // Scale covariance to make chi-squared test pass
         double scale_factor = chi_squared_term / threshold;
         aux_data.innovation_covariance *= scale_factor;
         measurement_covariance_ = aux_data.innovation_covariance - aux_data.jacobian * state_covariance * aux_data.jacobian.transpose();
         previous_measurement_data_.covariance = measurement_covariance_;
         return true;
       }

       // Log warning with metadata when validation fails
       std::cerr << "WARNING: Measurement validation failed for sensor type "
                 << SensorTypeToString(type_) << std::endl
                 << "  Chi-squared value: " << chi_squared_term
                 << ", Threshold: " << threshold << std::endl
                 << "  Innovation norm: " << aux_data.innovation.norm()
                 << ", Measurement timestamp: " << measurement_timestamp << std::endl;

       // Always return false if validation fails
       return false;
     }

     bool GetSensorPoseInBodyFrame(Eigen::Isometry3d& pose) const {
       pose = sensor_pose_in_body_frame_;
       return true;
     }

     bool GetBodyToSensorTransform(Eigen::Isometry3d& pose) const {
       pose = body_to_sensor_transform_;
       return true;
     }

     bool SetSensorPoseInBodyFrame(const Eigen::Isometry3d& pose) {
       sensor_pose_in_body_frame_ = pose;
       body_to_sensor_transform_ = pose.inverse();
       return true;
     }

     virtual StateFlags GetInitializableStates() const = 0;

     virtual StateFlags InitializeState(
         const DynamicVector& measurement,
         const StateFlags& valid_states,
         StateVector& state,
         StateCovariance& covariance) const = 0;

     SensorType GetModelType() const {
       return type_;
     }

     bool ValidateMeasurementAndTime(
         const DynamicVector& measurement,
         double timestamp,
         double reference_time,
         double max_delay_window) const {

       // Check for NaN/Inf in measurement
       if (!measurement.allFinite()) {
         std::cerr << "Invalid measurement from " << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       // Check measurement size
       ValidateMeasurementSize(measurement);

       // Reference time not yet set (first measurement)
       if (reference_time == std::numeric_limits<double>::lowest()) {
         return true;
       }

       // Check timestamp against delay window
       if (timestamp < reference_time - max_delay_window ||
           timestamp > reference_time + max_delay_window) {
         std::cerr << "Timestamp outside acceptable window for " << SensorTypeToString(type_) << " sensor" << std::endl;
         return false;
       }

       return true;
     }

     protected:
     void ValidateMeasurementSize(const DynamicVector& measurement) const {
           // Validate sensor type is known
       if (type_ == SensorType::Unknown) {
         throw std::invalid_argument("Cannot process measurement with Unknown sensor type");
       }

       if (measurement.size() != GetMeasurementDimension(type_)) {
         throw std::invalid_argument(
             "Measurement size mismatch: expected " +
             std::to_string(GetMeasurementDimension(type_)) +
             ", got " + std::to_string(measurement.size()));
       }
     }

   private:
     struct MeasurementData {
       double timestamp = 0.0;
       DynamicVector value;
       DynamicCovariance covariance;

       MeasurementData() = default;
       MeasurementData(double t, const DynamicVector& v, const DynamicCovariance& c)
         : timestamp(t), value(v), covariance(c) {}
     };

     void UpdateCovariance(const DynamicVector& innovation) {
       // Define bounds as named constants for clarity
       static const double kMinInnovation = 1e-6;
       static const double kMaxInnovation = 1e6;

       // 1. Clip large values (positive and negative)
       auto clipped = innovation.array().cwiseMax(-kMaxInnovation).cwiseMin(kMaxInnovation);

       // 2. Apply minimum magnitude while preserving sign
       // Formula: sign(x) * max(|x|, kMinInnovation)
       DynamicVector bounded_innovation =
           clipped.sign() * clipped.abs().cwiseMax(kMinInnovation);

       // Update covariance with the adaptive formula
       measurement_covariance_ += (bounded_innovation * bounded_innovation.transpose() -
                                  measurement_covariance_) / validation_params_.noise_sample_window;
     }

   protected:
     Eigen::Isometry3d sensor_pose_in_body_frame_ = Eigen::Isometry3d::Identity();
     Eigen::Isometry3d body_to_sensor_transform_ = Eigen::Isometry3d::Identity();
     DynamicCovariance measurement_covariance_;
     ValidationParams validation_params_;
     bool can_predict_input_accelerations_ = false;
     MeasurementData previous_measurement_data_;
     SensorType type_ = SensorType::Unknown;
   };

   } // namespace core
   } // namespace kinematic_arbiter

   #endif // KINEMATIC_ARBITER_CORE_MEASUREMENT_MODEL_INTERFACE_HPP_
