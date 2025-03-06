#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/mediation_types.hpp"

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
  using StateCovariance = Eigen::Matrix<double, StateSize, StateSize>;         // State covariance P
  using MeasurementVector = MeasurementVectorType;                             // Measurement vector y_k
  using MeasurementCovariance = Eigen::Matrix<double,
    MeasurementVectorType::RowsAtCompileTime,
    MeasurementVectorType::RowsAtCompileTime>;                               // Measurement covariance R
  using MeasurementJacobian = Eigen::Matrix<double,
    MeasurementVectorType::RowsAtCompileTime, StateSize>;                    // Measurement Jacobian C_k
  using InnovationCovariance = MeasurementCovariance;                          // Innovation covariance S

  /**
   * @brief Core measurement auxiliary data that can be reused across algorithms
   *
   * Contains expensive-to-compute elements that are needed for validation,
   * updating, retrodiction, and other filter operations.
   */
  struct MeasurementAuxData {
    MeasurementVector innovation;                // y_k - h(x_k)
    MeasurementJacobian jacobian;                // C_k or H
    InnovationCovariance innovation_covariance;  // S = C_k P_k C_k^T + R
    Eigen::MatrixXd innovation_covariance_inv;   // S^-1 (cached for efficiency)

    // Constructors
    MeasurementAuxData() = default;
    MeasurementAuxData(
        const MeasurementVector& inn,
        const MeasurementJacobian& jac,
        const InnovationCovariance& inn_cov)
      : innovation(inn), jacobian(jac), innovation_covariance(inn_cov) {
      innovation_covariance_inv = innovation_covariance.inverse();
    }
  };

  /**
   * @brief Validation result with chi-squared test details
   */
  struct ValidationResult {
    bool assumptions_valid = false;    // Whether filter assumptions hold
    double chi_squared = 0.0;          // Computed chi-squared value
    double threshold = 0.0;            // Threshold used for the test
  };

  /**
   * @brief Parameters for assumption validation
   */
  struct ValidationParams {
    // Sample window size for adaptive estimation of R
    size_t noise_sample_window = 30;

    // Chi-squared threshold for measurement validation
    // If set to 0.0, will be auto-computed based on confidence level
    double chi_squared_threshold = 0.0;

    // Confidence level for auto-computing chi-squared threshold (0-1)
    double confidence_level = 0.95;
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
   * @brief Validate filter assumptions using chi-squared test
   *
   * Tests if measurement is consistent with model expectations.
   *
   * @param aux_data Auxiliary measurement data
   * @return Result of chi-squared validation
   */
  ValidationResult ValidateAssumptions(const MeasurementAuxData& aux_data) const {
    ValidationResult result;

    // Mahalanobis distance: d = ν^T S^-1 ν
    result.chi_squared = aux_data.innovation.transpose() *
                         aux_data.innovation_covariance_inv *
                         aux_data.innovation;

    // Determine threshold for chi-squared test
    result.threshold = validation_params_.chi_squared_threshold;
    if (result.threshold <= 0.0) {
      result.threshold = GetChiSquaredThreshold(
          validation_params_.confidence_level,
          aux_data.innovation.rows());
    }

    // Check if filter assumptions hold
    result.assumptions_valid = (result.chi_squared < result.threshold);

    return result;
  }

  /**
   * @brief Apply mediation action if validation fails
   *
   * If assumptions are violated and action is AdjustCovariance,
   * updates the covariance matrix to make test pass.
   *
   * @param validation_result Result from ValidateAssumptions
   * @param action Action to take if validation fails
   * @param[out] adjusted_covariance Output for potentially adjusted covariance
   * @return Whether mediation was applied
   */
  bool ApplyMediation(
      const ValidationResult& validation_result,
      MediationAction action,
      MeasurementCovariance& adjusted_covariance) const {

    // Initialize output to current covariance
    adjusted_covariance = measurement_covariance_;

    // If assumptions hold, no mediation needed
    if (validation_result.assumptions_valid) {
      return false;
    }

    // Apply mediation according to selected action
    if (action == MediationAction::AdjustCovariance) {
      // Scale covariance to make chi-squared test pass
      double scale_factor = validation_result.chi_squared / validation_result.threshold;
      adjusted_covariance = measurement_covariance_ * scale_factor;
    }

    // Mediation was applied
    return true;
  }

  /**
   * @brief Update measurement covariance based on innovation
   *
   * Implements the formula:
   * R̂_k = R̂_{k-1} + ((y_k - h(x_k))(y_k - h(x_k))^T - R̂_{k-1})/n
   *
   * @param aux_data Auxiliary measurement data
   */
  void UpdateCovariance(const MeasurementAuxData& aux_data) {
    measurement_covariance_ = measurement_covariance_ +
        (aux_data.innovation * aux_data.innovation.transpose() - measurement_covariance_) /
        validation_params_.noise_sample_window;
  }

  /**
   * @brief Perform the entire validation and mediation process
   *
   * Convenience method that combines ComputeAuxiliaryData, ValidateAssumptions,
   * ApplyMediation, and optionally UpdateCovariance in one call.
   *
   * @param state Current state estimate x_k
   * @param state_covariance Current state covariance P_k
   * @param measurement Actual measurement y_k
   * @param action Action to take if validation fails
   * @param[out] adjusted_covariance Output for potentially adjusted covariance
   * @param update_covariance Whether to update internal covariance with this measurement
   * @return Whether filter assumptions hold for this measurement
   */
  bool ValidateAndMediate(
      const StateVector& state,
      const StateCovariance& state_covariance,
      const MeasurementVector& measurement,
      MediationAction action,
      MeasurementCovariance& adjusted_covariance,
      bool update_covariance = false) {

    // Compute auxiliary data once for all operations
    MeasurementAuxData aux_data = ComputeAuxiliaryData(
        state, state_covariance, measurement);

    // Validate assumptions
    ValidationResult result = ValidateAssumptions(aux_data);

    // Apply mediation if needed
    bool mediation_applied = ApplyMediation(result, action, adjusted_covariance);

    // Update covariance if requested and appropriate
    if (update_covariance && (result.assumptions_valid ||
                             action == MediationAction::ForceAccept)) {
      UpdateCovariance(aux_data);
    }

    return result.assumptions_valid;
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

protected:
  /**
   * @brief Get chi-squared threshold for a given confidence level and DOF
   *
   * @param confidence Confidence level (0-1)
   * @param dof Degrees of freedom (measurement dimension)
   * @return Chi-squared threshold value
   */
  double GetChiSquaredThreshold(double confidence, int dof) const {
    // Common values for 95% confidence:
    switch (dof) {
      case 1: return 3.84;
      case 2: return 5.99;
      case 3: return 7.81;
      case 4: return 9.49;
      case 5: return 11.07;
      case 6: return 12.59;
      default: return dof + 1.8 * std::sqrt(dof); // Approximation
    }
  }

  Eigen::Isometry3d sensor_pose_in_body_frame_;      // Sensor-to-body transform
  Eigen::Isometry3d body_to_sensor_transform_;       // Body-to-sensor transform
  MeasurementCovariance measurement_covariance_;     // Measurement noise covariance R
  ValidationParams validation_params_;               // Parameters for validation
};

} // namespace core
} // namespace kinematic_arbiter
