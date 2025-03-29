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
      reference_time_(std::numeric_limits<double>::lowest()),
      max_delay_window_(1.0),
      initialized_states_(StateFlags::Zero()),
      reference_state_(StateVector::Zero()),
      reference_covariance_(StateMatrix::Identity()) {}

  /**
   * @brief Reset the filter to initial state
   */
  void reset() {
    reference_time_ = std::numeric_limits<double>::lowest();
    initialized_states_ = StateFlags::Zero();
    reference_state_ = StateVector::Zero();
    reference_covariance_ = StateMatrix::Identity();
    for (auto& sensor : sensors_) {
      sensor->reset();
    }
    process_model_->reset();
  }

  /**
   * @brief Add a sensor model
   *
   * @tparam Type Sensor type (from SensorType enum)
   * @param sensor_model Shared pointer to measurement model interface
   * @return size_t Index of the newly added sensor
   */
  size_t AddSensor(std::shared_ptr<MeasurementModelInterface> sensor_model) {
    if (!sensor_model) {
      throw std::invalid_argument("Cannot add null sensor model to filter");
    }

    size_t sensor_index = sensors_.size();

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
                                const MeasurementModelInterface::DynamicVector& measurement,
                                double timestamp) {
    if (sensor_index >= sensors_.size()) {
      std::cerr << "Warning: ProcessMeasurementByIndex called with invalid sensor index: " << sensor_index << std::endl;
      return false;
    }

    auto sensor = sensors_[sensor_index];

    // Set reference time if not initialized
    if (reference_time_ == std::numeric_limits<double>::lowest()) {
      reference_time_ = timestamp;
    }

    // Quick validation of measurement and timestamp before any expensive operations
    if (!sensor->ValidateMeasurementAndTime(measurement, timestamp, reference_time_, max_delay_window_)) {
      return false;
    }

    // Try to initialize uninitiated states
    StateFlags initializable = sensor->GetInitializableStates();
    StateFlags not_initialized = initialized_states_.select(StateFlags::Zero(), StateFlags::Ones());
    StateFlags uninit_states = initializable.cwiseProduct(not_initialized);

    // Only attempt initialization for states we haven't initialized yet
    if (uninit_states.any()) {
      StateFlags new_states = sensor->InitializeState(
          measurement, initialized_states_, reference_state_, reference_covariance_);
      initialized_states_ = initialized_states_.cwiseMax(new_states);
    }

    double dt = timestamp - reference_time_;

    // Predict state to measurement time
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateVector state_at_sensor_time;


    state_at_sensor_time = process_model_->PredictState(reference_state_, dt);


    // Validate predicted state
    if (!state_at_sensor_time.allFinite()) {
      std::cerr << "Predicted state contains NaN/Inf values at time " << timestamp << std::endl;
      return false;
    }

    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    StateMatrix covariance_at_sensor_time = A * reference_covariance_ * A.transpose() + Q;
    covariance_at_sensor_time = 0.5 * (covariance_at_sensor_time + covariance_at_sensor_time.transpose());

    // Compute auxiliary data
    MeasurementModelInterface::MeasurementAuxData aux_data;

    // Validate and mediate the measurement
    bool measurement_valid = sensor->ValidateAndMediate(
        state_at_sensor_time, covariance_at_sensor_time, timestamp, measurement, aux_data);

    if (!measurement_valid) {
      return false;
    }

    Eigen::MatrixXd PHt = covariance_at_sensor_time * aux_data.jacobian.transpose();
    Eigen::MatrixXd K_transpose = aux_data.innovation_covariance.ldlt().solve(PHt.transpose());
    Eigen::MatrixXd kalman_gain = K_transpose.transpose();

    // Update state estimate
    StateVector updated_state = state_at_sensor_time + kalman_gain * aux_data.innovation;

    // Update covariance (Joseph form for numerical stability)
    StateMatrix I_KH = StateMatrix::Identity() - kalman_gain * aux_data.jacobian;
    StateMatrix updated_covariance = I_KH * covariance_at_sensor_time * I_KH.transpose() +
        kalman_gain * sensor->GetMeasurementCovariance() * kalman_gain.transpose();

    updated_covariance = 0.5 * (updated_covariance + updated_covariance.transpose());

    if (!updated_covariance.allFinite()) {
      std::cerr << "Updated covariance contains NaN/Inf values for at time " << timestamp << std::endl;
      return false;
    }

    // Update process noise
    process_model_->UpdateProcessNoise(
        state_at_sensor_time,
        updated_state,
        sensor->GetValidationParams().process_to_measurement_noise_ratio,
        dt
    );

    // Update reference state based on timestamp
    if (timestamp > reference_time_) {
      reference_state_ = updated_state;
      reference_covariance_ = updated_covariance;
      reference_time_ = timestamp;
    } else {
      double forward_dt = reference_time_ - timestamp;
      reference_state_ = process_model_->PredictState(updated_state, forward_dt);

      if (!reference_state_.allFinite()) {
        std::cerr << "Forward propagation produced invalid state for at time " << timestamp << std::endl;
        return false;
      }

      StateMatrix A_forward = process_model_->GetTransitionMatrix(updated_state, forward_dt);
      StateMatrix Q_forward = process_model_->GetProcessNoiseCovariance(forward_dt);
      reference_covariance_ = A_forward * updated_covariance * A_forward.transpose() + Q_forward;
      reference_covariance_ = 0.5 * (reference_covariance_ + reference_covariance_.transpose());
    }

    return true;
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
                                 MeasurementModelInterface::DynamicCovariance& covariance) const {
    if (sensor_index >= sensors_.size()) {return false;}
    covariance = sensors_[sensor_index]->GetMeasurementCovariance();
    return true;
  }

  /**
   * @brief Get expected measurement from a sensor using current reference state
   * @param sensor_index Index of the sensor
   * @param[out] expected_measurement The predicted measurement
   * @param[in] state_at_sensor_time The state at the sensor time
   * @return bool True if the operation was successful
   */
  bool GetExpectedMeasurementByIndex(size_t sensor_index,
                                   MeasurementModelInterface::DynamicVector& expected_measurement) const {
    if (sensor_index >= sensors_.size()) {return false;}
    expected_measurement = sensors_[sensor_index]->PredictMeasurement(reference_state_);
    return true;
  }

  /**
   * @brief Get expected measurement from a sensor using a specific state
   * @param sensor_index Index of the sensor
   * @param[out] expected_measurement The predicted measurement
   * @param[in] state_at_sensor_time The state at the sensor time
   * @return bool True if the operation was successful
   */
  bool GetExpectedMeasurementByIndex(size_t sensor_index,
                                   MeasurementModelInterface::DynamicVector& expected_measurement,
                                   const StateVector& state_at_sensor_time) const {
    if (sensor_index >= sensors_.size()) {return false;}
    expected_measurement = sensors_[sensor_index]->PredictMeasurement(state_at_sensor_time);
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
