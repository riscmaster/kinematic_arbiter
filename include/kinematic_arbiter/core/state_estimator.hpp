#pragma once

#include <memory>
#include <string>
#include <Eigen/Dense>

#include "kinematic_arbiter/core/state_index.hpp"

namespace kinematic_arbiter {
namespace core {

// Forward declarations to minimize includes
class StateModelInterface;
template<typename T> class MeasurementModelInterface;

/**
 * @brief Minimal state estimation interface
 */
class StateEstimator {
public:
  using StateVector = Eigen::Matrix<double, StateIndex::kFullStateSize, 1>;
  using StateCovariance = Eigen::Matrix<double, StateIndex::kFullStateSize, StateIndex::kFullStateSize>;

  /**
   * @brief Constructor
   *
   * @param process_model Process model instance
   */
  explicit StateEstimator(std::shared_ptr<StateModelInterface> process_model);

  /**
   * @brief Destructor
   */
  ~StateEstimator();

  /**
   * @brief Register a sensor
   *
   * @param sensor_id Unique identifier for the sensor
   * @param sensor_model Sensor model instance
   * @return Success status
   */
  template<typename SensorType>
  bool AddSensor(const std::string& sensor_id, std::shared_ptr<SensorType> sensor_model);

  /**
   * @brief Add a measurement
   *
   * @param sensor_id Sensor identifier
   * @param timestamp Measurement timestamp
   * @param value Measurement value
   * @return Success status
   */
  template<typename MeasurementType>
  bool AddMeasurement(const std::string& sensor_id, double timestamp, const MeasurementType& value);

  /**
   * @brief Get state estimate
   *
   * @param timestamp Desired timestamp (0 = latest)
   * @return State vector at requested time
   */
  StateVector GetState(double timestamp = 0);

  /**
   * @brief Get state covariance
   *
   * @param timestamp Desired timestamp (0 = latest)
   * @return Covariance matrix at requested time
   */
  StateCovariance GetCovariance(double timestamp = 0);

  /**
   * @brief Initialize the filter
   *
   * @param initial_state Initial state
   * @param initial_covariance Initial covariance
   * @param timestamp Initial timestamp
   */
  void Initialize(const StateVector& initial_state,
                  const StateCovariance& initial_covariance,
                  double timestamp);

  /**
   * @brief Get sensor model by ID
   *
   * @param sensor_id Sensor identifier
   * @return Pointer to sensor model or nullptr
   */
  template<typename SensorType>
  std::shared_ptr<SensorType> GetSensor(const std::string& sensor_id);

  /**
   * @brief Get process model
   *
   * @return Pointer to process model
   */
  std::shared_ptr<StateModelInterface> GetProcessModel();

  /**
   * @brief Get sensor measurement covariance by ID
   *
   * @param sensor_id Sensor identifier
   * @return Measurement covariance matrix for the sensor
   */
  template<typename SensorType>
  typename SensorType::MeasurementCovariance GetSensorCovariance(const std::string& sensor_id) const;

  /**
   * @brief Get current filter time
   *
   * @return Current filter time
   */
  double GetCurrentTime() const;

private:
  // Implementation details hidden
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace core
} // namespace kinematic_arbiter
