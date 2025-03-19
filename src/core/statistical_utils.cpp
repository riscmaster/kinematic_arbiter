#include "kinematic_arbiter/core/statistical_utils.hpp"
#include <stdexcept>
#include <string>

namespace kinematic_arbiter {
namespace utils {

double InterpolatedCriticalValue(double confidence_level,
                                 ChiSquareCriticalValue lower,
                                 ChiSquareCriticalValue upper) {
  double confidence_spread = upper.confidence - lower.confidence;
  if (std::fabs(confidence_spread) < kFuzzyMoreThanZero) {
    return upper.critical_value;
  }
  double portion = (confidence_level - lower.confidence) / confidence_spread;
  return lower.critical_value +
         portion * (upper.critical_value - lower.critical_value);
}

double CalculateChiSquareCriticalValue1Dof(double confidence_level) {
  // Index 0 corresponds to 1 DOF
  return CalculateChiSquareCriticalValueNDof(0, confidence_level);
}

double CalculateChiSquareCriticalValueNDof(size_t dof_index,
                                           double confidence_level) {
  // Check if the requested degrees of freedom is supported
  if (dof_index >= kMaxChiSquareDof) {
    throw std::invalid_argument("Unsupported degrees of freedom: " +
                               std::to_string(dof_index) +
                               ". Maximum supported DoF is " +
                               std::to_string(kMaxChiSquareDof));
  }
  if (confidence_level <
      kChiSquareCriticalValues[dof_index].front().confidence) {
    return kChiSquareCriticalValues[dof_index].front().critical_value;
  }

  // Binary search considered but array is small
  for (size_t index = 1U; index < kChiSquareCriticalValues[dof_index].size();
       ++index) {
    if (confidence_level <
        kChiSquareCriticalValues[dof_index][index].confidence) {
      return InterpolatedCriticalValue(
          confidence_level, kChiSquareCriticalValues[dof_index][index - 1U],
          kChiSquareCriticalValues[dof_index][index]);
    }
  }
  return kChiSquareCriticalValues[dof_index].back().critical_value;
}

} // namespace utils
} // namespace kinematic_arbiter
