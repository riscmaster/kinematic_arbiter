#include "kinematic_arbiter/core/mediated_kalman_filter.hpp"

namespace kinematic_arbiter {
namespace core {

// This file can contain non-template implementations or specializations if needed,
// but currently the implementation is fully in the .tpp file as it's template-based.
//
// Potential additions to this file:
// 1. Factory functions for common filter configurations
// 2. Specializations for common state/measurement combinations
// 3. Debug and logging utilities

// Example convenience factory function for creating standard filter configurations
// This would need to be extended with actual implementations

/*
template<typename StateVector, typename... SensorTypes>
auto CreateStandardFilter(double history_window = 1.0) {
    // Create process model with typical parameters
    auto process_model = CreateStandardProcessModel<StateVector>();

    // Create measurement models with typical parameters
    auto measurement_models = std::make_tuple(CreateStandardSensorModel<SensorTypes>()...);

    // Create and initialize filter
    auto filter = MediatedKalmanFilter(std::move(process_model), std::move(measurement_models)...);

    // Set filter parameters
    MediatedKalmanFilter::Params params;
    params.max_history_window = history_window;
    filter.Initialize(params);

    return filter;
}
*/

} // namespace core
} // namespace kinematic_arbiter
