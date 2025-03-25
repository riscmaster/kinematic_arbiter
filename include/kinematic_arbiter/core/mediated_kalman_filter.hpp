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
   * @brief Add a sensor model (only cares about the interface, not specific types)
   *
   * @param sensor_model Any sensor model implementing MeasurementModelInterface
   * @return size_t Index of the newly added sensor
   */
  size_t AddSensor(std::shared_ptr<MeasurementModelInterface> sensor_model) {
    size_t sensor_index = sensors_.size();
    sensors_.push_back(sensor_model);
    return sensor_index;
  }

  /**
   * @brief Get sensor model by index (returns base interface pointer)
   *
   * @param sensor_index Index of the sensor
   * @return std::shared_ptr<MeasurementModelInterface> The sensor model or nullptr if invalid index
   */
  std::shared_ptr<MeasurementModelInterface> GetSensorByIndex(size_t sensor_index) const {
    if (sensor_index >= sensors_.size()) {
      return nullptr;
    }
    return sensors_[sensor_index];
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

    auto& sensor_model = sensors_[sensor_index];
    return sensor_model->ProcessMeasurement(*this, timestamp, measurement);
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
  // Store sensors directly as measurement model interfaces
  std::vector<std::shared_ptr<MeasurementModelInterface>> sensors_;

  // Process measurement functionality can be implemented without
  // needing to know the specific sensor types

  // Private data members
  std::shared_ptr<ProcessModel> process_model_;
  double reference_time_;
  double max_delay_window_;
  StateFlags initialized_states_;
  StateVector reference_state_;
  StateMatrix reference_covariance_;
};

} // namespace core
} // namespace kinematic_arbiter
