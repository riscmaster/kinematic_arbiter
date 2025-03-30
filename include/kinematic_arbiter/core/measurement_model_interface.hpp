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

/**
 * @brief Interface for measurement models with assumption validation
 *
 * Provides methods for predicting measurements, computing Jacobians,
 * and validating filter assumptions using chi-squared testing.
 *
 * @tparam Type Sensor type that defines the measurement vector dimensions
 */
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

  /**
   * @brief Parameters for assumption validation
   */
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

  /**
   * @brief Core measurement auxiliary data that can be reused across algorithms
   *
   * Contains expensive-to-compute elements that are needed for validation,
   * updating, retrodiction, and other filter operations.
   */
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

  /**
   * @brief Constructor
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
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

  /**
   * @brief Reset the measurement model to initial state
   */
  virtual void reset() = 0;

  /**
   * @brief Virtual destructor
   */
  virtual ~MeasurementModelInterface() = default;

  /**
   * @brief Predict expected measurement h(x) from state
   *
   * @param state Current state estimate x_k
   * @return Expected measurement h(x)
   */
  virtual DynamicVector PredictMeasurement(const StateVector& state) const = 0;

  /**
   * @brief Compute the measurement Jacobian H = ∂h/∂x
   *
   * @param state Current state estimate x_k
   * @return Measurement Jacobian H (C_k in paper notation)
   */
  virtual DynamicJacobian GetMeasurementJacobian(const StateVector& state) const = 0;

  /**
   * @brief Get the inputs to the prediction model
   *
   * @param state_before_prediction Current state estimate x_k
   * @param state_covariance_before_prediction Current state covariance P_k
   * @param measurement_after_prediction Actual measurement y_k after prediction of dt
   * @param dt Time step in seconds
   * @return Linear and angular acceleration as inputs to the prediction model
   */
  virtual Eigen::Matrix<double, 6, 1> GetPredictionModelInputs(const StateVector& , const StateCovariance& , const DynamicVector& , double) const {return Eigen::Matrix<double, 6, 1>::Zero();};

  /**
   * @brief Get whether the prediction model can predict input accelerations
   * @return True if the prediction model can predict input accelerations
   */
  bool CanPredictInputAccelerations() const {return can_predict_input_accelerations_;}

  /**
   * @brief Get current measurement covariance
   * @return Measurement noise covariance R
   */
  const DynamicCovariance& GetMeasurementCovariance() const {
    return measurement_covariance_;
  }

  /**
   * @brief Set the validation parameters
   * @param params New validation parameters
   */
  void SetValidationParams(const ValidationParams& params) {
    validation_params_ = params;
  }

  /**
   * @brief Get the current validation parameters
   * @return Current validation parameters
   */
  const ValidationParams& GetValidationParams() const {
    return validation_params_;
  }

  /**
   * @brief Compute auxiliary measurement data for use in various algorithms
   *
   * Calculates innovation, Jacobian, and innovation covariance once
   * for reuse in validation, update, retrodiction, etc.
   *
   * @param state Current state estimate x_k
   * @param state_covariance Current state covariance P_k
   * @param measurement Actual measurement y_k
   * @return Auxiliary measurement data
   */
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

  /**
   * @brief Validate auxiliary measurement data (innovation, jacobian, covariance)
   *
   * @param aux_data Auxiliary measurement data
   * @return true if validation passes, false otherwise
   */
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

  /**
   * @brief Perform the validation and mediation process
   *
   * @param state Current state estimate x_k
   * @param state_covariance Current state covariance P_k
   * @param measurement_timestamp Measurement timestamp
   * @param measurement Actual measurement y_k
   * @param aux_data Output parameter for auxiliary measurement data
   * @return Whether filter assumptions hold for this measurement
   */
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
      measurement_covariance_ *= scale_factor;
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

  /**
   * @brief Get sensor pose in body frame
   * @return Sensor-to-body transform
   */
  bool GetSensorPoseInBodyFrame(Eigen::Isometry3d& pose) const {
    pose = sensor_pose_in_body_frame_;
    return true;
  }

  /**
   * @brief Get body to sensor transform
   * @return Body-to-sensor transform
   */
  bool GetBodyToSensorTransform(Eigen::Isometry3d& pose) const {
    pose = body_to_sensor_transform_;
    return true;
  }

  /**
   * @brief Set sensor pose in body frame
   * @param pose New sensor-to-body transform
   */
  bool SetSensorPoseInBodyFrame(const Eigen::Isometry3d& pose) {
    sensor_pose_in_body_frame_ = pose;
    body_to_sensor_transform_ = pose.inverse();
    return true;
  }

  /**
   * @brief Get the states that this sensor can directly initialize
   *
   * Returns an array of boolean flags where each element corresponds to a state component.
   * A true value indicates that the sensor can directly initialize that state.
   *
   * @return Flags indicating initializable states
   */
  virtual StateFlags GetInitializableStates() const = 0;

  /**
   * @brief Initialize state components from a measurement
   *
   * This method initializes state components that this sensor can directly observe.
   * Updates both the state vector and covariance matrix for initialized components.
   *
   * @param measurement The measurement to use for initialization
   * @param valid_states Flags indicating which states are valid for use in initialization
   * @param state [in/out] The state vector to be initialized
   * @param covariance [in/out] The state covariance to be initialized
   * @return Flags indicating which states were initialized
   */
  virtual StateFlags InitializeState(
      const DynamicVector& measurement,
      const StateFlags& valid_states,
      StateVector& state,
      StateCovariance& covariance) const = 0;

  /**
   * @brief Get the type of this sensor model (non-virtual)
   * @return The sensor type enumeration value
   */
  SensorType GetModelType() const {
    return type_;
  }

  /**
   * @brief Validate measurement and timestamp before any expensive operations
   *
   * @param measurement Raw measurement vector
   * @param timestamp Measurement timestamp
   * @param reference_time Current filter reference time
   * @param max_delay_window Maximum delay window for measurements
   * @return bool Whether measurement passed basic validation
   */
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
  /**
   * @brief Validate measurement size based on sensor type
   * @param measurement Measurement vector to validate
   */
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

  /**
   * @brief Update measurement covariance with bounded innovation
   *
   * @param innovation Measurement innovation
   */
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
