#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/mediation_types.hpp"
#include "kinematic_arbiter/core/statistical_utils.hpp"

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Interface for measurement models with assumption validation
 *
 * Provides methods for predicting measurements, computing Jacobians,
 * and validating filter assumptions using chi-squared testing.
 *
 * @tparam MeasurementVectorType Type defining the measurement vector dimensions
 */
template<typename MeasurementVectorType>
class MeasurementModelInterface {
public:
  // Type definitions
  static constexpr int StateSize = StateIndex::kFullStateSize;
  using StateVector = Eigen::Matrix<double, StateSize, 1>;                     // State vector x
  using StateCovariance = Eigen::Matrix<double, StateSize, StateSize>;
  using MeasurementVector = MeasurementVectorType;                             // Measurement vector y_k
  using MeasurementCovariance = Eigen::Matrix<double,
    MeasurementVectorType::RowsAtCompileTime,
    MeasurementVectorType::RowsAtCompileTime>;                               // Measurement covariance R
  using MeasurementJacobian = Eigen::Matrix<double,
    MeasurementVectorType::RowsAtCompileTime, StateSize>;                    // Measurement Jacobian C_k
  using InnovationCovariance = MeasurementCovariance;                          // Innovation covariance S

  // More general name for boolean state flags
  using StateFlags = Eigen::Array<bool, StateSize, 1>;

  /**
   * @brief Core measurement auxiliary data that can be reused across algorithms
   *
   * Contains expensive-to-compute elements that are needed for validation,
   * updating, retrodiction, and other filter operations.
   */
  struct MeasurementAuxData {
    MeasurementVector innovation;                // y_k - C_k x_k
    MeasurementJacobian jacobian;                // C_k
    InnovationCovariance innovation_covariance;  // S = C_k P_k C_k^T + R

    // Constructors
    MeasurementAuxData() = default;
    MeasurementAuxData(
        const MeasurementVector& inn,
        const MeasurementJacobian& jac,
        const InnovationCovariance& inn_cov)
      : innovation(inn), jacobian(jac), innovation_covariance(inn_cov) {}
  };


  /**
   * @brief Parameters for assumption validation
   */
  struct ValidationParams {
    // Sample window size for adaptive estimation of R
    size_t noise_sample_window = 40;

    // Confidence level assumption validation (0-1)
    double confidence_level = 0.95;

    // Process to measurement noise ratio
    double process_to_measurement_noise_ratio = 2.0;

    // Mediation action to take if validation fails
    MediationAction mediation_action = MediationAction::AdjustCovariance;
  };

  /**
   * @brief Constructor
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit MeasurementModelInterface(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : sensor_pose_in_body_frame_(sensor_pose_in_body_frame),
      body_to_sensor_transform_(sensor_pose_in_body_frame.inverse()),
      measurement_covariance_(MeasurementCovariance::Identity()),
      validation_params_(params) {}

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
  virtual MeasurementVector PredictMeasurement(const StateVector& state) const = 0;

  /**
   * @brief Compute the measurement Jacobian H = ∂h/∂x
   *
   * @param state Current state estimate x_k
   * @return Measurement Jacobian H (C_k in paper notation)
   */
  virtual MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const = 0;

  /**
   * @brief Get the inputs to the prediction model
   *
   * @param state_before_prediction Current state estimate x_k
   * @param state_covariance_before_prediction Current state covariance P_k
   * @param measurement_after_prediction Actual measurement y_k after prediction of dt
   * @param dt Time step in seconds
   * @return Linear and angular acceleration as inputs to the prediction model
   */
  virtual Eigen::Matrix<double, 6, 1> GetPredictionModelInputs(const StateVector& , const StateCovariance& , const MeasurementVector& , double) const {return Eigen::Matrix<double, 6, 1>::Zero();};

  /**
   * @brief Get whether the prediction model can predict input accelerations
   * @return True if the prediction model can predict input accelerations
   */
  bool CanPredictInputAccelerations() const {return can_predict_input_accelerations_;}

  /**
   * @brief Get current measurement covariance
   * @return Measurement noise covariance R
   */
  const MeasurementCovariance& GetMeasurementCovariance() const {
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
      const MeasurementVector& measurement) const {

    // Calculate innovation: ν = y_k - h(x_k)
    MeasurementVector innovation = measurement - PredictMeasurement(state);

    // Get measurement Jacobian: C_k
    MeasurementJacobian jacobian = GetMeasurementJacobian(state);

    // Calculate innovation covariance: S = C_k P_k C_k^T + R
    InnovationCovariance innovation_covariance =
        jacobian * state_covariance * jacobian.transpose() + measurement_covariance_;

    return MeasurementAuxData(innovation, jacobian, innovation_covariance);
  }

  /**
   * @brief Perform the validation and mediation process
   *
   * @param state Current state estimate x_k
   * @param state_covariance Current state covariance P_k
   * @param measurement Actual measurement y_k
   * @return Whether filter assumptions hold for this measurement
   */
  bool ValidateAndMediate(
      const StateVector& state,
      const StateCovariance& state_covariance,
      const MeasurementVector& measurement) {

    // Compute auxiliary data once for all operations
    MeasurementAuxData aux_data = ComputeAuxiliaryData(
        state, state_covariance, measurement);

    // Mahalanobis distance: d = ν^T S^-1 ν
    double chi_squared_term = aux_data.innovation.transpose() *
                         aux_data.innovation_covariance.llt().solve(
                         aux_data.innovation);

    // Determine threshold for chi-squared test
    double threshold = utils::CalculateChiSquareCriticalValueNDof(
          aux_data.innovation.rows()-1, validation_params_.confidence_level);

    // Check if measurement passes validation
    if (chi_squared_term < threshold) {
      UpdateCovariance(aux_data.innovation);
      return true;
    }

    // Measurement Apply Mediation Action
    if (validation_params_.mediation_action == MediationAction::AdjustCovariance) {
      // Scale covariance to make chi-squared test pass
      double scale_factor = chi_squared_term / threshold;
      measurement_covariance_ *= scale_factor;
    }

    // Always return false if validation fails
    return false;
  }

  /**
   * @brief Get sensor pose in body frame
   * @return Sensor-to-body transform
   */
  const Eigen::Isometry3d& GetSensorPoseInBodyFrame() const {
    return sensor_pose_in_body_frame_;
  }

  /**
   * @brief Get body to sensor transform
   * @return Body-to-sensor transform
   */
  const Eigen::Isometry3d& GetBodyToSensorTransform() const {
    return body_to_sensor_transform_;
  }

  /**
   * @brief Set sensor pose in body frame
   * @param pose New sensor-to-body transform
   */
  void SetSensorPoseInBodyFrame(const Eigen::Isometry3d& pose) {
    sensor_pose_in_body_frame_ = pose;
    body_to_sensor_transform_ = pose.inverse();
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
      const MeasurementVector& measurement,
      const StateFlags& valid_states,
      StateVector& state,
      StateCovariance& covariance) const = 0;

private:

  /**
   * @brief Update measurement covariance with bounded innovation
   *
   * @param innovation Measurement innovation
   */
  void UpdateCovariance(const MeasurementVector& innovation) {
    // Define bounds as named constants for clarity
    static const double kMinInnovation = 1e-6;
    static const double kMaxInnovation = 1e6;

    // 1. Clip large values (positive and negative)
    auto clipped = innovation.array().cwiseMax(-kMaxInnovation).cwiseMin(kMaxInnovation);

    // 2. Apply minimum magnitude while preserving sign
    // Formula: sign(x) * max(|x|, kMinInnovation)
    MeasurementVector bounded_innovation =
        clipped.sign() * clipped.abs().cwiseMax(kMinInnovation);

    // Update covariance with the adaptive formula
    measurement_covariance_ += (bounded_innovation * bounded_innovation.transpose() -
                               measurement_covariance_) / validation_params_.noise_sample_window;
  }

protected:
  Eigen::Isometry3d sensor_pose_in_body_frame_;      // Sensor-to-body transform
  Eigen::Isometry3d body_to_sensor_transform_;       // Body-to-sensor transform
  MeasurementCovariance measurement_covariance_;     // Measurement noise covariance R
  ValidationParams validation_params_;               // Parameters for validation
  bool can_predict_input_accelerations_ = false;
};

} // namespace core
} // namespace kinematic_arbiter
