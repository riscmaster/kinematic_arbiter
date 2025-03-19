#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <cstddef>
#include <Eigen/Dense>

namespace kinematic_arbiter {
namespace utils {

/**
 * @brief Maximum degrees of freedom for chi-square distribution
 */
constexpr size_t kMaxChiSquareDof = 7;

/**
 * @brief Critical value for chi-square test at a specific confidence level
 */
struct ChiSquareCriticalValue {
  double confidence;
  double critical_value;
};

// Constants
constexpr double kFuzzyMoreThanZero = 1e-10;
constexpr std::size_t kNumChiSquareCriticalValues = 13U;

/**
 * @brief Chi-square critical values table
 *
 * These values are generated using the integration of the chi square
 * distribution from 0 to the critical value resulting a given confidence
 * level. Reference: https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
 */
constexpr std::array<
    std::array<ChiSquareCriticalValue, kNumChiSquareCriticalValues>,
    kMaxChiSquareDof>
    kChiSquareCriticalValues{{
        // 1 DOF
        {{
            {0.01, 0.0002},
            {0.025, 0.001},
            {0.05, 0.0039},
            {0.10, 0.0158},
            {0.25, 0.102},
            {0.50, 0.455},
            {0.75, 1.33},
            {0.90, 2.71},
            {0.95, 3.84},
            {0.975, 5.02},
            {0.99, 6.63},
            {0.999, 10.83},
            {0.9999, 15.0}
        }},
        // 2 DOF
        {{
            {0.01, 0.0201},
            {0.025, 0.0506},
            {0.05, 0.103},
            {0.10, 0.211},
            {0.25, 0.575},
            {0.50, 1.39},
            {0.75, 2.77},
            {0.90, 4.61},
            {0.95, 5.99},
            {0.975, 7.38},
            {0.99, 9.21},
            {0.999, 13.8},
            {0.9999, 18.5}
        }},
        // 3 DOF
        {{
            {0.01, 0.115},
            {0.025, 0.216},
            {0.05, 0.35},
            {0.10, 0.584},
            {0.25, 1.21},
            {0.50, 2.37},
            {0.75, 4.11},
            {0.90, 6.25},
            {0.95, 7.82},
            {0.975, 9.35},
            {0.99, 11.3},
            {0.999, 16.3},
            {0.9999, 21}
        }},
        // 4 DOF
        {{
            {0.01, 0.297},
            {0.025, 0.484},
            {0.05, 0.711},
            {0.10, 1.06},
            {0.25, 1.92},
            {0.50, 3.36},
            {0.75, 5.39},
            {0.90, 7.78},
            {0.95, 9.49},
            {0.975, 11.1},
            {0.99, 13.3},
            {0.999, 18.5},
            {0.9999, 23.5}
        }},
        // 5 DOF
        {{
            {0.01, 0.554},
            {0.025, 0.831},
            {0.05, 1.15},
            {0.10, 1.61},
            {0.25, 2.68},
            {0.50, 4.35},
            {0.75, 6.63},
            {0.90, 9.24},
            {0.95, 11.1},
            {0.975, 12.8},
            {0.99, 15.1},
            {0.999, 20.5},
            {0.9999, 25.8}
        }},
        // 6 DOF
        {{
            {0.01, 0.872},
            {0.025, 1.24},
            {0.05, 1.64},
            {0.10, 2.2},
            {0.25, 3.46},
            {0.50, 5.35},
            {0.75, 7.84},
            {0.90, 10.7},
            {0.95, 12.6},
            {0.975, 14.5},
            {0.99, 16.8},
            {0.999, 22.4},
            {0.9999, 28}
        }},
        // 7 DOF
        {{
            {0.01, 1.24},
            {0.025, 1.69},
            {0.05, 2.17},
            {0.10, 2.83},
            {0.25, 4.26},
            {0.50, 6.35},
            {0.75, 9.04},
            {0.90, 12},
            {0.95, 14.1},
            {0.975, 16},
            {0.99, 18.5},
            {0.999, 24.3},
            {0.9999, 30}
        }}
    }};

/**
 * @brief Interpolate between two critical values
 *
 * @param confidence_level Target confidence level
 * @param lower Lower critical value data point
 * @param upper Upper critical value data point
 * @return Interpolated critical value
 */
double InterpolatedCriticalValue(double confidence_level,
                                 ChiSquareCriticalValue lower,
                                 ChiSquareCriticalValue upper);

/**
 * @brief Calculate chi-square critical value for 1 degree of freedom
 *
 * @param confidence_level Confidence level (0-1)
 * @return Chi-square critical value
 */
double CalculateChiSquareCriticalValue1Dof(double confidence_level);

/**
 * @brief Calculate chi-square critical value for N degrees of freedom
 *
 * @param dof_index Index of degrees of freedom (0-6)
 * @param confidence_level Confidence level (0-1)
 * @return Chi-square critical value
 */
double CalculateChiSquareCriticalValueNDof(size_t dof_index,
                                           double confidence_level);

} // namespace utils
} // namespace kinematic_arbiter
