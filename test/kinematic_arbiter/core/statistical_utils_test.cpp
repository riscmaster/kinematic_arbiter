#include "kinematic_arbiter/core/statistical_utils.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace kinematic_arbiter {
namespace core {
namespace test {

class StatisticalUtilsTest : public ::testing::Test {
protected:
  // Tolerance for floating point comparisons
  static constexpr double kEpsilon = 1e-10;
};

TEST_F(StatisticalUtilsTest, InterpolatedCriticalValue_MidPoint) {
  // Replace designated initializers with regular struct initialization
  ChiSquareCriticalValue lower{0.90, 10.0};
  ChiSquareCriticalValue upper{0.95, 20.0};

  // Test midpoint interpolation
  double result = InterpolatedCriticalValue(0.925, lower, upper);
  EXPECT_NEAR(result, 15.0, kEpsilon);
}

TEST_F(StatisticalUtilsTest, InterpolatedCriticalValue_AtLowerBound) {
  ChiSquareCriticalValue lower{0.90, 10.0};
  ChiSquareCriticalValue upper{0.95, 20.0};

  // Test at lower bound
  double result = InterpolatedCriticalValue(0.90, lower, upper);
  EXPECT_NEAR(result, 10.0, kEpsilon);
}

TEST_F(StatisticalUtilsTest, InterpolatedCriticalValue_AtUpperBound) {
  ChiSquareCriticalValue lower{0.90, 10.0};
  ChiSquareCriticalValue upper{0.95, 20.0};

  // Test at upper bound
  double result = InterpolatedCriticalValue(0.95, lower, upper);
  EXPECT_NEAR(result, 20.0, kEpsilon);
}

TEST_F(StatisticalUtilsTest, InterpolatedCriticalValue_ZeroSpread) {
  ChiSquareCriticalValue lower{0.90, 10.0};
  ChiSquareCriticalValue upper{0.90, 20.0}; // Same confidence

  // Test with zero spread
  double result = InterpolatedCriticalValue(0.90, lower, upper);
  EXPECT_NEAR(result, 20.0, kEpsilon); // Should return upper value
}

TEST_F(StatisticalUtilsTest, InterpolatedCriticalValue_NearZeroSpread) {
  ChiSquareCriticalValue lower{0.90, 10.0};
  ChiSquareCriticalValue upper{0.90 + kFuzzyMoreThanZero/2, 20.0};

  // Test with near-zero spread (below threshold)
  double result = InterpolatedCriticalValue(0.90, lower, upper);
  EXPECT_NEAR(result, 20.0, kEpsilon); // Should return upper value
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValue1Dof_KnownValues) {
  // Test known values for 1 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValue1Dof(0.95), 3.84, 0.01);
  EXPECT_NEAR(CalculateChiSquareCriticalValue1Dof(0.99), 6.63, 0.01);
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValueNDof_ExactTableValues) {
  // Test exact values from the table
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(0, 0.95), 3.84, 0.01); // 1 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(1, 0.95), 5.99, 0.01); // 2 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(2, 0.95), 7.82, 0.01); // 3 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(3, 0.95), 9.49, 0.01); // 4 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(4, 0.95), 11.1, 0.01); // 5 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(5, 0.95), 12.6, 0.01); // 6 DOF
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(6, 0.95), 14.1, 0.01); // 7 DOF
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValueNDof_Interpolation) {
  // Test interpolation between table values
  // For 2 DOF: 0.90 -> 4.61, 0.95 -> 5.99
  double midpoint_confidence = 0.925;
  double expected = 4.61 + (5.99 - 4.61) * 0.5; // Linear interpolation
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(1, midpoint_confidence), expected, 0.01);
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValueNDof_BelowMinConfidence) {
  // Test below minimum confidence in table
  double min_confidence = 0.01;
  double below_min = min_confidence / 2;

  // Should return value for minimum confidence
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(0, below_min),
              CalculateChiSquareCriticalValueNDof(0, min_confidence), kEpsilon);
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValueNDof_AboveMaxConfidence) {
  // Test above maximum confidence in table
  double max_confidence = 0.9999;
  double above_max = max_confidence + 0.0001;

  // Should return value for maximum confidence
  EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(0, above_max),
              CalculateChiSquareCriticalValueNDof(0, max_confidence), kEpsilon);
}

TEST_F(StatisticalUtilsTest, CalculateChiSquareCriticalValueNDof_InvalidDof) {
  // Test unsupported DOF
  EXPECT_THROW(CalculateChiSquareCriticalValueNDof(7, 0.95), std::invalid_argument);
  EXPECT_THROW(CalculateChiSquareCriticalValueNDof(100, 0.95), std::invalid_argument);
}

// Cross-check against statistical tables or external libraries
TEST_F(StatisticalUtilsTest, ValidationAgainstStatisticalReferences) {
  // Values from a well-established chi-squared table
  // Source: NIST/SEMATECH e-Handbook of Statistical Methods
  const std::vector<std::pair<int, double>> reference_values = {
    {1, 3.84}, // 1 DOF, 95% confidence
    {2, 5.99}, // 2 DOF, 95% confidence
    {3, 7.81}, // 3 DOF, 95% confidence
    {4, 9.49}, // 4 DOF, 95% confidence
    {5, 11.07}, // 5 DOF, 95% confidence
    {6, 12.59}, // 6 DOF, 95% confidence
    {7, 14.07}  // 7 DOF, 95% confidence
  };

  for (const auto& [dof, expected_value] : reference_values) {
    EXPECT_NEAR(CalculateChiSquareCriticalValueNDof(dof-1, 0.95), expected_value, 0.05)
        << "Failed for DOF: " << dof;
  }
}

// Test consistency between 1 DOF function and N DOF function
TEST_F(StatisticalUtilsTest, ConsistencyBetween1DofAndNDof) {
  for (double confidence = 0.01; confidence <= 0.99; confidence += 0.05) {
    EXPECT_NEAR(CalculateChiSquareCriticalValue1Dof(confidence),
                CalculateChiSquareCriticalValueNDof(0, confidence), kEpsilon);
  }
}

} // namespace test
} // namespace core
} // namespace kinematic_arbiter

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
