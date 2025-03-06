#pragma once

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Possible corrective actions when filter assumptions are violated
 *
 * These actions define how to respond when a measurement fails validation.
 */
enum class MediationAction {
  ForceAccept,     // Proceed with measurement despite validation failure
  Reject,          // Reject the measurement entirely
  AdjustCovariance // Adjust covariance to make measurement valid
};

} // namespace core
} // namespace kinematic_arbiter
