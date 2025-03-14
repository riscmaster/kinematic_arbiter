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
 * @brief Exception class for filter-related errors
 */
class FilterException : public std::runtime_error {
public:
  explicit FilterException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Generic measurement container with timestamp
 *
 * @tparam SensorType The sensor model type
 */
template<typename SensorType>
struct Measurement {
  using ValueType = typename SensorType::MeasurementVector;

  double timestamp;                                      // Measurement timestamp
  ValueType value;                                       // Measurement value
  std::optional<typename SensorType::MeasurementCovariance> noise_override; // Optional override for noise

  Measurement(double t, const ValueType& v)
    : timestamp(t), value(v), noise_override(std::nullopt) {}

  Measurement(double t, const ValueType& v,
              const typename SensorType::MeasurementCovariance& cov)
    : timestamp(t), value(v), noise_override(cov) {}
};

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
      max_delay_window_(10.0),
      initialized_states_(StateFlags::Zero()),
      reference_state_(StateVector::Zero()),
      reference_covariance_(StateMatrix::Identity()) {}

  /**
   * @brief Add a new sensor model
   *
   * @tparam SensorModelType Type of sensor model implementing MeasurementModelInterface
   * @param sensor_model Shared pointer to the sensor model
   * @return size_t Index of the newly added sensor
   */
  template<typename SensorModelType>
  size_t AddSensor(std::shared_ptr<SensorModelType> sensor_model) {
    size_t sensor_index = sensors_.size();

    // Create a wrapper that stores both the sensor model and its type
    auto wrapper = std::make_shared<SensorWrapper<SensorModelType>>(sensor_model);
    sensors_.push_back(wrapper);

    return sensor_index;
  }

  /**
   * @brief Process measurement for a sensor by index
   *
   * @tparam MeasurementType Type of measurement
   * @param sensor_index Index of the sensor to use
   * @param timestamp Measurement timestamp
   * @param measurement Measurement data
   * @return true if measurement was processed successfully
   */
  template<typename MeasurementType>
  bool ProcessMeasurementByIndex(size_t sensor_index, double timestamp, const MeasurementType& measurement) {
    if (sensor_index >= sensors_.size()) {
      return false; // Invalid sensor index
    }

    auto& sensor_wrapper = sensors_[sensor_index];
    return sensor_wrapper->ProcessMeasurement(*this, timestamp, measurement);
  }

  /**
   * @brief Get sensor model by index
   *
   * @tparam SensorModelType Expected type of the sensor model
   * @param sensor_index Index of the sensor
   * @return std::shared_ptr<SensorModelType> The sensor model or nullptr if types don't match
   */
  template<typename SensorModelType>
  std::shared_ptr<SensorModelType> GetSensorByIndex(size_t sensor_index) const {
    if (sensor_index >= sensors_.size()) {
      return nullptr;
    }

    auto& wrapper = sensors_[sensor_index];
    auto* typed_wrapper = dynamic_cast<SensorWrapper<SensorModelType>*>(wrapper.get());

    if (!typed_wrapper) {
      return nullptr;
    }

    return typed_wrapper->sensor_model;
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
  void SetStateEstimate(const StateVector& state) {
    reference_state_ = state;
    initialized_states_ = StateFlags::Ones();
  }

  /**
   * @brief Set state covariance
   */
  void SetStateCovariance(const StateMatrix& covariance) {
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

  /**
   * @brief Get process model
   */
  std::shared_ptr<ProcessModel> GetProcessModel() const {
    return process_model_;
  }

  // Update IsInitialized to check if any states are initialized
  bool IsInitialized() const { return initialized_states_.any(); }

  /**
   * @brief Predict and update reference state to a new timestamp (for dead reckoning)
   *
   * This method explicitly advances the filter's reference state and covariance
   * using only the process model prediction, without incorporating any measurements.
   *
   * @param timestamp The new reference timestamp
   * @return true if prediction was successful
   */
  void PredictNewReference(double timestamp) {
    double dt = timestamp - reference_time_;
    reference_state_ = process_model_->PredictState(reference_state_, dt);
    StateMatrix A = process_model_->GetTransitionMatrix(reference_state_, dt);
    StateMatrix Q = process_model_->GetProcessNoiseCovariance(dt);
    reference_covariance_ = A * reference_covariance_ * A.transpose() + Q;
    reference_time_ = timestamp;
  }

private:
  // Base type-erased sensor wrapper
  class SensorWrapperBase {
  public:
    virtual ~SensorWrapperBase() = default;

    template<typename MeasurementType>
    bool ProcessMeasurement(MediatedKalmanFilter& filter, double timestamp, const MeasurementType& measurement) {
      return DoProcessMeasurement(filter, timestamp, &measurement);
    }

  protected:
    virtual bool DoProcessMeasurement(MediatedKalmanFilter& filter, double timestamp, const void* measurement) = 0;
  };

  // Type-specific sensor wrapper implementation
  template<typename SensorModelType>
  class SensorWrapper : public SensorWrapperBase {
  public:
    explicit SensorWrapper(std::shared_ptr<SensorModelType> model)
      : sensor_model(model) {}

    std::shared_ptr<SensorModelType> sensor_model;

  protected:
    using MeasurementType = typename SensorModelType::MeasurementVector;

    bool DoProcessMeasurement(MediatedKalmanFilter& filter, double timestamp, const void* measurement) override {
      const MeasurementType* typed_measurement = static_cast<const MeasurementType*>(measurement);

      auto sensor_model = this->sensor_model;

      // Try to initialize any states this sensor can provide
      StateFlags initializable = sensor_model->GetInitializableStates();

      // Create "not initialized" mask using select
      StateFlags not_initialized = filter.initialized_states_.select(StateFlags::Zero(), StateFlags::Ones());

      // States that can be initialized and aren't yet
      StateFlags uninit_states = initializable.cwiseProduct(not_initialized);

      // Only attempt initialization for states we haven't initialized yet
      if (uninit_states.any()) {
        StateFlags new_states = sensor_model->InitializeState(
            *typed_measurement, filter.initialized_states_, filter.reference_state_, filter.reference_covariance_);

        // Update initialized states
        filter.initialized_states_ = filter.initialized_states_.cwiseMax(new_states);
      }

      // Reject too-old measurements
      if (timestamp < filter.reference_time_ - filter.max_delay_window_) {
        return false;
      }

      double dt = timestamp - filter.reference_time_;



      // Use the inputs for the transition matrix - note we only pass the required parameters
      StateMatrix A = filter.process_model_->GetTransitionMatrix(
          filter.reference_state_, dt);
      StateVector state_at_sensor_time = filter.reference_state_;
      if (sensor_model->CanPredictInputAccelerations()) {
        // Use the fully qualified type name to avoid scope issues
      Eigen::Matrix<double, 6, 1> inputs = sensor_model->GetPredictionModelInputs(
          filter.reference_state_, filter.reference_covariance_, *typed_measurement, dt);

      // Extract the linear and angular acceleration components
      Eigen::Vector3d linear_accel = inputs.template segment<3>(0);
      Eigen::Vector3d angular_accel = inputs.template segment<3>(3);
      // Predict the state using the optional acceleration inputs
      state_at_sensor_time = filter.process_model_->PredictStateWithInputAccelerations(
          filter.reference_state_, dt, linear_accel, angular_accel);
      } else {
        // Use the default prediction model
      state_at_sensor_time = filter.process_model_->PredictState(
          filter.reference_state_, dt);
      }

      StateMatrix Q = filter.process_model_->GetProcessNoiseCovariance(dt);
      StateMatrix covariance_at_sensor_time = A * filter.reference_covariance_ * A.transpose() + Q;

      // Apply measurement directly to the filter's reference state
      bool measurement_accepted = this->ApplyMeasurement(
          filter,
          state_at_sensor_time,
          covariance_at_sensor_time,
          *typed_measurement,
          timestamp,
          dt
      );

      // Return false only when the measurement is explicitly rejected
      return measurement_accepted || sensor_model->GetValidationParams().mediation_action != MediationAction::Reject;
    }

    bool ApplyMeasurement(MediatedKalmanFilter& filter,
                         const StateVector& predicted_state,
                         const StateMatrix& predicted_covariance,
                         const MeasurementType& measurement,
                         double measurement_timestamp,
                         double dt = 0.0) {
      auto& sensor_model = this->sensor_model;

      // Compute auxiliary data (innovation, Jacobian, innovation covariance)
      auto aux_data = sensor_model->ComputeAuxiliaryData(predicted_state, predicted_covariance, measurement);

      // Validate measurement
      bool measurement_valid = sensor_model->ValidateAndMediate(
          predicted_state,        // Current state
          predicted_covariance,   // Current covariance
          measurement           // Measurement to validate
      );

      // If measurement is invalid and we're set to reject, return early
      if (!measurement_valid && sensor_model->GetValidationParams().mediation_action == MediationAction::Reject) {
        return false;  // Measurement rejected without updating filter
      }

      // Proceed with Kalman update - compute PHt
      auto PHt = predicted_covariance * aux_data.jacobian.transpose();

      // Simple regularization of innovation covariance to ensure it's invertible
      auto S = aux_data.innovation_covariance;

      // Direct calculation of Kalman gain
      // Solve S*K' = PHt' for K'
      Eigen::MatrixXd K_transpose = S.ldlt().solve(PHt.transpose());
      auto kalman_gain = K_transpose.transpose();

      // Update state estimate (x = x + K*innovation)
      StateVector updated_state = predicted_state + kalman_gain * aux_data.innovation;

      // Joseph form update for better numerical stability in the covariance
      auto I_KH = StateMatrix::Identity() - kalman_gain * aux_data.jacobian;
      StateMatrix updated_covariance = I_KH * predicted_covariance * I_KH.transpose() +
                                       kalman_gain * sensor_model->GetMeasurementCovariance() * kalman_gain.transpose();

      // Update process noise using the proper UpdateProcessNoise interface
      filter.process_model_->UpdateProcessNoise(
          predicted_state,   // a priori state (before update)
          updated_state,     // a posteriori state (after update)
          sensor_model->GetValidationParams().process_to_measurement_noise_ratio,
          dt
      );

      // Directly update the filter's reference values
      if (measurement_timestamp > filter.reference_time_) {
        // For newer measurements, update the reference time and state
        filter.reference_state_ = updated_state;
        filter.reference_covariance_ = updated_covariance;
        filter.reference_time_ = measurement_timestamp;
      } else {
        // For older measurements, propagate the updated state forward to the reference time
        double forward_dt = filter.reference_time_ - measurement_timestamp;
        filter.reference_state_ = filter.process_model_->PredictState(updated_state, forward_dt);
        StateMatrix A_forward = filter.process_model_->GetTransitionMatrix(updated_state, forward_dt);
        StateMatrix Q_forward = filter.process_model_->GetProcessNoiseCovariance(forward_dt);
        filter.reference_covariance_ = A_forward * updated_covariance * A_forward.transpose() + Q_forward;
      }

      return measurement_valid;
    }
  };

  // Private data members
  std::shared_ptr<ProcessModel> process_model_;
  std::vector<std::shared_ptr<SensorWrapperBase>> sensors_;

  // Core filter state data (made accessible to SensorWrapper)
  friend class SensorWrapperBase;
  template<typename SensorModelType> friend class SensorWrapper;

  double reference_time_;
  double max_delay_window_;
  StateFlags initialized_states_;
  StateVector reference_state_;
  StateMatrix reference_covariance_;
};

} // namespace core
} // namespace kinematic_arbiter
