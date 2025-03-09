#include "kinematic_arbiter/core/state_estimator.hpp"
#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"
#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/measurement_model_interface.hpp"

#include <unordered_map>
#include <typeindex>
#include <stdexcept>

namespace kinematic_arbiter {
namespace core {

// Implementation class for StateEstimator
class StateEstimator::Impl {
public:
  // Constructor with process model
  explicit Impl(std::shared_ptr<StateModelInterface> process_model)
    : process_model_(process_model), initialized_(false), current_time_(0.0) {}

  // Type-erased sensor model storage
  struct SensorModelEntry {
    std::type_index type_id;
    std::shared_ptr<void> model;

    template<typename SensorType>
    std::shared_ptr<SensorType> get() {
      return std::static_pointer_cast<SensorType>(model);
    }
  };

  // Type-erased generic filter implementation
  struct FilterBase {
    virtual ~FilterBase() = default;
    virtual bool ProcessMeasurement(const std::string& sensor_id, double timestamp,
                                    const void* measurement, size_t measure_type_id) = 0;
    virtual StateVector Predict(double timestamp) = 0;
    virtual const StateVector& GetStateEstimate() const = 0;
    virtual const StateMatrix& GetStateCovariance() const = 0;
    virtual double GetCurrentTime() const = 0;
    virtual void Initialize(const StateVector& state, const StateMatrix& covariance, double timestamp) = 0;
  };

  // Concrete filter implementation with known sensor types
  template<typename... SensorTypes>
  struct ConcreteFilter : public FilterBase {
    using FilterType = MediatedKalmanFilter<StateIndex::kFullStateSize,
                                          StateModelInterface, SensorTypes...>;

    ConcreteFilter(std::shared_ptr<StateModelInterface> process_model,
                   std::shared_ptr<SensorTypes>... sensor_models)
      : filter_(process_model, sensor_models...) {}

    bool ProcessMeasurement(const std::string& sensor_id, double timestamp,
                            const void* measurement, size_t measure_type_id) override {
      // This is where we'd use a tuple of indexes or other technique to
      // match the sensor_id with the correct type and dispatch the measurement
      // For a real implementation, this requires a bit more template metaprogramming
      return false; // Simplified for example
    }

    StateVector Predict(double timestamp) override {
      return filter_.Predict(timestamp);
    }

    const StateVector& GetStateEstimate() const override {
      return filter_.GetStateEstimate();
    }

    const StateMatrix& GetStateCovariance() const override {
      return filter_.GetStateCovariance();
    }

    double GetCurrentTime() const override {
      return filter_.GetCurrentTime();
    }

    void Initialize(const StateVector& state, const StateMatrix& covariance, double timestamp) override {
      filter_.Initialize(state, covariance, timestamp);
    }

    FilterType filter_;
  };

  // Storage for registered sensors
  std::unordered_map<std::string, SensorModelEntry> sensors_;

  // Process model
  std::shared_ptr<StateModelInterface> process_model_;

  // Filter implementation (created once all sensors are registered)
  std::unique_ptr<FilterBase> filter_;

  // State tracking
  bool initialized_;
  double current_time_;
  StateVector current_state_;
  StateMatrix current_covariance_;

  // Helper to register a sensor
  template<typename SensorType>
  bool RegisterSensor(const std::string& sensor_id, std::shared_ptr<SensorType> sensor_model) {
    // Check if sensor ID already exists
    if (sensors_.find(sensor_id) != sensors_.end()) {
      return false;
    }

    // Store sensor model
    SensorModelEntry entry;
    entry.type_id = std::type_index(typeid(SensorType));
    entry.model = sensor_model;
    sensors_[sensor_id] = entry;

    // Invalidate filter - will be recreated on next operation
    filter_.reset();

    return true;
  }

  // Helper to process a measurement
  template<typename MeasurementType>
  bool ProcessMeasurement(const std::string& sensor_id, double timestamp, const MeasurementType& value) {
    // Find sensor
    auto it = sensors_.find(sensor_id);
    if (it == sensors_.end()) {
      return false;
    }

    // Ensure filter exists
    if (!filter_) {
      CreateFilter();
    }

    // Process measurement through filter
    // In a real implementation, we'd need to ensure type compatibility and dispatch correctly
    return filter_->ProcessMeasurement(sensor_id, timestamp, &value, typeid(MeasurementType).hash_code());
  }

  // Create filter instance with registered sensors
  void CreateFilter() {
    // In a real implementation, this would use template metaprogramming to create
    // the correct filter type with all registered sensor models
    // For example purpose, we'll just create a simplified placeholder
    filter_ = std::make_unique<ConcreteFilter<>>(process_model_);
  }
};

// StateEstimator implementation delegates to Impl

StateEstimator::StateEstimator(std::shared_ptr<StateModelInterface> process_model)
  : impl_(std::make_unique<Impl>(process_model)) {}

StateEstimator::~StateEstimator() = default;

template<typename SensorType>
bool StateEstimator::AddSensor(const std::string& sensor_id, std::shared_ptr<SensorType> sensor_model) {
  return impl_->RegisterSensor(sensor_id, sensor_model);
}

template<typename MeasurementType>
bool StateEstimator::AddMeasurement(const std::string& sensor_id, double timestamp, const MeasurementType& value) {
  return impl_->ProcessMeasurement(sensor_id, timestamp, value);
}

StateEstimator::StateVector StateEstimator::GetState(double timestamp) {
  if (timestamp <= 0.0) {
    return impl_->current_state_;
  }

  // Create filter if needed
  if (!impl_->filter_) {
    impl_->CreateFilter();
  }

  // Predict state at requested timestamp
  return impl_->filter_->Predict(timestamp);
}

StateEstimator::StateCovariance StateEstimator::GetCovariance(double timestamp) {
  if (timestamp <= 0.0) {
    return impl_->current_covariance_;
  }

  // In a real implementation, we'd predict the covariance as well
  // This is a simplified placeholder
  return impl_->current_covariance_;
}

void StateEstimator::Initialize(const StateVector& initial_state,
                               const StateCovariance& initial_covariance,
                               double timestamp) {
  impl_->current_state_ = initial_state;
  impl_->current_covariance_ = initial_covariance;
  impl_->current_time_ = timestamp;
  impl_->initialized_ = true;

  // Initialize filter if it exists
  if (impl_->filter_) {
    impl_->filter_->Initialize(initial_state, initial_covariance, timestamp);
  }
}

template<typename SensorType>
std::shared_ptr<SensorType> StateEstimator::GetSensor(const std::string& sensor_id) {
  auto it = impl_->sensors_.find(sensor_id);
  if (it == impl_->sensors_.end()) {
    return nullptr;
  }

  return it->second.get<SensorType>();
}

std::shared_ptr<StateModelInterface> StateEstimator::GetProcessModel() {
  return impl_->process_model_;
}

// Explicit template instantiations for common sensor types
// These would be expanded based on your actual sensor models
// For example:
// template bool StateEstimator::AddSensor<sensors::ImuSensorModel>(const std::string&, std::shared_ptr<sensors::ImuSensorModel>);
// template bool StateEstimator::AddMeasurement<ImuMeasurement>(const std::string&, double, const ImuMeasurement&);

} // namespace core
} // namespace kinematic_arbiter
