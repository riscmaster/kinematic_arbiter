#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Interface for measurement models used in Kalman filtering
 *
 * This interface defines the common functionality for measurement models
 * that map the system state x to expected measurements h(x).
 *
 * @tparam StateType Type of state vector x
 * @tparam MeasurementType Type of measurement vector z
 */
template<typename StateType, typename MeasurementType>
class MeasurementModelInterface {
public:
  // Type definitions
  using StateVector = StateType;                  // State vector x
  using MeasurementVector = MeasurementType;      // Measurement vector z
  using MeasurementMatrix = Eigen::Matrix<double, MeasurementType::RowsAtCompileTime, MeasurementType::RowsAtCompileTime>;  // Measurement covariance R
  using MeasurementJacobian = Eigen::Matrix<double, MeasurementType::RowsAtCompileTime, StateType::RowsAtCompileTime>;     // Measurement Jacobian H = ∂h/∂x

  /**
   * @brief Parameters for measurement model mediation
   */
  struct Params {
    // Sample window size for adaptive measurement noise estimation
    size_t noise_sample_window = 40;

    // Process noise modifier (ζ). Lower values increase confidence in the process model
    // Range: [0, inf)
    double process_to_measurement_noise_ratio = 2.0;

    // Confidence level for measurement validation (used for chi-squared test)
    // Range: (0, 1)
    double confidence_value = 0.95;

    // Initial measurement noise covariance (σ²)
    double initial_noise_uncertainty = 1.0;

    // Pose of the sensor in the body frame (T_B^S)
    // This defines where the sensor is located and oriented relative to the body
    Eigen::Isometry3d sensor_pose_in_body_frame = Eigen::Isometry3d::Identity();
  };

  /**
   * @brief Constructor with parameters
   *
   * @param params Parameters for this measurement model
   */
  explicit MeasurementModelInterface(const Params& params = Params())
    : params_(params),
      covariance_(MeasurementMatrix::Identity() * params.initial_noise_uncertainty),
      body_to_sensor_transform_(params.sensor_pose_in_body_frame.inverse()) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~MeasurementModelInterface() = default;

  /**
   * @brief Predict expected measurement from state: z = h(x)
   *
   * Maps the current state estimate to the expected measurement.
   *
   * @param state Current state estimate x
   * @return Expected measurement h(x)
   */
  virtual MeasurementVector PredictMeasurement(const StateVector& state) const = 0;

  /**
   * @brief Compute the Jacobian of the measurement model: H = ∂h/∂x
   *
   * This matrix linearizes the measurement function around the current state.
   *
   * @param state Current state estimate x
   * @return Jacobian matrix H = ∂h/∂x
   */
  virtual MeasurementJacobian GetMeasurementJacobian(const StateVector& state) const = 0;

  /**
   * @brief Get the current measurement covariance matrix R
   *
   * @return Measurement covariance matrix R
   */
  const MeasurementMatrix& GetCovariance() const { return covariance_; }

  /**
   * @brief Get the measurement model parameters
   *
   * @return Const reference to parameters
   */
  const Params& GetParams() const { return params_; }

  /**
   * @brief Update the measurement covariance based on innovation
   *
   * Implements an adaptive estimation of R using the measurement innovation:
   * R_k = R_{k-1} + (ν_k·ν_k^T - R_{k-1})/N
   * where ν_k is the innovation and N is the sample window size.
   *
   * @param innovation The measurement innovation ν = z - h(x)
   */
  void UpdateCovariance(const MeasurementVector& innovation) {
    // R_k = R_{k-1} + (ν_k·ν_k^T - R_{k-1})/N
    covariance_ = covariance_ +
        (innovation * innovation.transpose() - covariance_) / params_.noise_sample_window;
  }

  /**
   * @brief Get the pose of the sensor in the body frame (T_B^S)
   *
   * @return The sensor pose in body frame
   */
  const Eigen::Isometry3d& GetSensorPoseInBodyFrame() const {
    return params_.sensor_pose_in_body_frame;
  }

  /**
   * @brief Get the transform from body frame to sensor frame (T_S^B)
   *
   * @return The body-to-sensor transform
   */
  const Eigen::Isometry3d& GetBodyToSensorTransform() const {
    return body_to_sensor_transform_;
  }

  /**
   * @brief Set the pose of the sensor in the body frame (T_B^S)
   *
   * Updates both the sensor pose and the inverse transform.
   *
   * @param pose The new sensor pose in body frame
   */
  void SetSensorPoseInBodyFrame(const Eigen::Isometry3d& pose) {
    params_.sensor_pose_in_body_frame = pose;
    body_to_sensor_transform_ = pose.inverse();
  }

protected:
  Params params_;
  MeasurementMatrix covariance_;    // Measurement covariance R
  Eigen::Isometry3d body_to_sensor_transform_;  // T_S^B (transform from body to sensor)
};

} // namespace core
} // namespace kinematic_arbiter
