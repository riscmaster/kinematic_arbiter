#pragma once

#include <stdexcept>
#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <optional>
#include <type_traits>
#include <Eigen/Dense>
#include <boost/math/distributions/chi_squared.hpp>

#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/mediation_types.hpp"

namespace kinematic_arbiter {
namespace core {

/**
 * @brief MediatedKalmanFilter implements a Kalman filter with chi-squared validation
 *
 * Leverages StateModelInterface for prediction and ProcessNoise management
 * and MeasurementModelInterface for measurement processing, validation and mediation.
 *
 * @tparam StateSize Dimension of state vector
 * @tparam ProcessModel Type implementing StateModelInterface
 */
template<int StateSize, typename ProcessModel>
class MediatedKalmanFilter {
public:
  // Type aliases
  using StateVector = Eigen::Matrix<double, StateSize, 1>;
  using StateMatrix = Eigen::Matrix<double, StateSize, StateSize>;
  using StateFlags = Eigen::Array<bool, StateSize, 1>;

  /**
   * @brief Constructor with just process model
   */
  explicit MediatedKalmanFilter(std::shared_ptr<ProcessModel> process_model)
    : process_model_(process_model),
      reference_time_(0.0),
      max_delay_window_(1.0),
      initialized_states_(StateFlags::Zero()),
      reference_state_(StateVector::Zero()),
      reference_covariance_(StateMatrix::Identity()) {}

  /**
   * @brief Add a sensor model
   *
   * @tparam Type Sensor type (from SensorType enum)
   * @param sensor_model Shared pointer to measurement model interface
   * @return size_t Index of the newly added sensor
   */
  size_t AddSensor(std::shared_ptr<MeasurementModelInterface> sensor_model) {
    size_t sensor_index = sensors_types_.size();

    // Store sensor model and its type tag
    sensors_.push_back(sensor_model);

    return sensor_index;
  }

  /**
   * @brief Get sensor pose in body frame
   * @param sensor_index Index of the sensor
   * @param[out] pose The sensor-to-body transform
   * @return bool True if the operation was successful
   */
  bool GetSensorPoseInBodyFrameByIndex(size_t sensor_index, Eigen::Isometry3d& pose) const {
    if (sensor_index >= sensors_.size()) {return false;}
    return sensors_[sensor_index]->GetSensorPoseInBodyFrame(pose);
  }

  /**
   * @brief Set sensor pose in body frame
   * @param sensor_index Index of the sensor
   * @param pose New sensor-to-body transform
   * @return bool True if the operation was successful
   */
  bool SetSensorPoseInBodyFrameByIndex(size_t sensor_index, const Eigen::Isometry3d& pose) {
    if (sensor_index >= sensors_.size()) {return false;}
    return sensors_[sensor_index]->SetSensorPoseInBodyFrame(pose);
  }

  /**
   * @brief Process measurement for a sensor by index
   *
   * @tparam Type The sensor type enum value
   * @param sensor_index Index of the sensor to use
   * @param measurement Measurement vector
   * @param timestamp Timestamp of the measurement
   * @return true if measurement was processed successfully
   */
  bool ProcessMeasurementByIndex(size_t sensor_index,
                                const MeasurementModelInterface::MeasurementVector& measurement,
                                double timestamp) {
    if (sensor_index >= sensors_.size()) {return false;}

    auto sensor = sensors_[sensor_index];

    // ---- Phase 1: Process the measurement ----

    // Try to initialize any states this sensor can provide
    StateFlags initializable = sensor->GetInitializableStates();
    StateFlags not_initialized = initialized_states_.select(StateFlags::Zero(), StateFlags::Ones());
    StateFlags uninit_states = initializable.cwiseProduct(not_initialized);

    // Only attempt initialization for states we haven't initialized yet
    if (uninit_states.any()) {
      StateFlags new_states = sensor->InitializeState(
          measurement, initialized_states_, reference_state_, reference_covariance_);
      initialized_states_ = initialized_states_.cwiseMax(new_states);
    }

    // Reject too-old measurements
    if (timestamp < reference_time_ - max_delay_window_) {
      return false;
    }

    double dt = timestamp - reference_time_;

    // Predict state to measurement time
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateVector state_at_sensor_time;

    if (sensor->CanPredictInputAccelerations()) {
      // Get acceleration inputs from the sensor
      Eigen::Matrix<double, 6, 1> inputs = sensor->GetPredictionModelInputs(
          reference_state_, reference_covariance_, measurement, dt);

      // Extract components with explicit template argument
      Eigen::Vector3d linear_accel = inputs.template segment<3>(0);
      Eigen::Vector3d angular_accel = inputs.template segment<3>(3);

      // Predict with accelerations
      state_at_sensor_time = process_model_->PredictStateWithInputAccelerations(
          reference_state_, dt, linear_accel, angular_accel);
    } else {
      // Default prediction
      state_at_sensor_time = process_model_->PredictState(reference_state_, dt);
    }

    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    StateMatrix covariance_at_sensor_time = A * reference_covariance_ * A.transpose() + Q;

    // ---- Phase 2: Apply the measurement ----

    // Compute auxiliary data (innovation, Jacobian, innovation covariance)
    auto aux_data = sensor->ComputeAuxiliaryData(
        state_at_sensor_time, covariance_at_sensor_time, measurement);

    // Validate measurement
    bool measurement_valid = sensor->ValidateAndMediate(
        state_at_sensor_time, covariance_at_sensor_time, timestamp, measurement);

    // If invalid and set to reject, return early
    if (!measurement_valid &&
        sensor->GetValidationParams().mediation_action == MediationAction::Reject) {
      return false;
    }

    // Kalman update - compute PHt
    auto PHt = covariance_at_sensor_time * aux_data.jacobian.transpose();
    auto S = aux_data.innovation_covariance;

    // Use LDLT decomposition for solving
    Eigen::MatrixXd K_transpose = S.ldlt().solve(PHt.transpose());
    auto kalman_gain = K_transpose.transpose();

    // Update state estimate
    StateVector updated_state = state_at_sensor_time + kalman_gain * aux_data.innovation;

    // Joseph form covariance update for stability
    auto I_KH = StateMatrix::Identity() - kalman_gain * aux_data.jacobian;
    StateMatrix updated_covariance = I_KH * covariance_at_sensor_time * I_KH.transpose() +
        kalman_gain * sensor->GetMeasurementCovariance() * kalman_gain.transpose();

    // Update process noise
    process_model_->UpdateProcessNoise(
        state_at_sensor_time,
        updated_state,
        sensor->GetValidationParams().process_to_measurement_noise_ratio,
        dt
    );

    // Update reference state based on timestamp
    if (timestamp > reference_time_) {
      // For newer measurements, update directly
      reference_state_ = updated_state;
      reference_covariance_ = updated_covariance;
      reference_time_ = timestamp;
    } else {
      // For older measurements, propagate forward
      double forward_dt = reference_time_ - timestamp;
      reference_state_ = process_model_->PredictState(updated_state, forward_dt);
      StateMatrix A_forward = process_model_->GetTransitionMatrix(updated_state, forward_dt);
      StateMatrix Q_forward = process_model_->GetProcessNoiseCovariance(forward_dt);
      reference_covariance_ = A_forward * updated_covariance * A_forward.transpose() + Q_forward;
    }

    return measurement_valid;
  }


  /**
   * @brief Get state estimate at given time
   */
  StateVector GetStateEstimate(double timestamp = -1.0) const {
    if (timestamp < 0.0 || timestamp == reference_time_) {
      return reference_state_;
    }

    // Predict state to requested time without changing reference
    double dt = timestamp - reference_time_;
    return process_model_->PredictState(reference_state_, dt);
  }

  /**
   * @brief Get state covariance at given time
   */
  StateMatrix GetStateCovariance(double timestamp = -1.0) const {
    if (timestamp < 0.0 || timestamp == reference_time_) {
      return reference_covariance_;
    }

    // Predict covariance to requested time without changing reference
    double dt = timestamp - reference_time_;
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    return A * reference_covariance_ * A.transpose() + Q;
  }

  /**
   * @brief Set state estimate
   */
  void SetStateEstimate(const StateVector& state, double timestamp, const StateMatrix& covariance=StateMatrix::Identity()) {
    reference_state_ = state;
    initialized_states_ = StateFlags::Ones();
    reference_time_ = timestamp;
    reference_covariance_ = covariance;
  }

  /**
   * @brief Get current time
   */
  double GetCurrentTime() const { return reference_time_; }

  /**
   * @brief Set max delay window
   */
  void SetMaxDelayWindow(double window) { max_delay_window_ = window; }


  // Update IsInitialized to check if any states are initialized
  bool IsInitialized() const { return initialized_states_.any(); }

  /**
   * @brief Predict and update reference state to a new timestamp (for dead reckoning)
   *
   * This method explicitly advances the filter's reference state and covariance
   * using only the process model prediction, without incorporating any measurements.
   *
   * @param timestamp The new reference timestamp
   */
  void PredictNewReference(double timestamp) {
    double dt = timestamp - reference_time_;
    reference_state_ = process_model_->PredictState(reference_state_, dt);
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    reference_covariance_ = A * reference_covariance_ * A.transpose() + Q;
    reference_time_ = timestamp;
  }

  /**
   * @brief Get sensor's measurement covariance matrix
   * @param sensor_index Index of the sensor
   * @param[out] covariance The measurement covariance matrix
   * @return bool True if the operation was successful
   */
  bool GetSensorCovarianceByIndex(size_t sensor_index,
                                 MeasurementModelInterface::MeasurementCovariance& covariance) const {
    if (sensor_index >= sensors_.size()) {return false;}
    return sensors_[sensor_index]->GetMeasurementCovariance(covariance);
  }

  /**
   * @brief Get expected measurement from a sensor using current reference state
   * @param sensor_index Index of the sensor
   * @param[out] expected_measurement The predicted measurement
   * @return bool True if the operation was successful
   */
  bool GetExpectedMeasurementByIndex(size_t sensor_index,
                                   MeasurementModelInterface::MeasurementVector& expected_measurement) const {
    if (sensor_index >= sensors_.size()) {return false;}
    expected_measurement = sensors_[sensor_index]->PredictMeasurement(reference_state_);
    return true;
  }

private:

  // Private data members
  std::shared_ptr<ProcessModel> process_model_;
  std::vector<std::shared_ptr<MeasurementModelInterface>> sensors_;
  double reference_time_;
  double max_delay_window_;
  StateFlags initialized_states_;
  StateVector reference_state_;
  StateMatrix reference_covariance_;
};

} // namespace core
} // namespace kinematic_arbiter
