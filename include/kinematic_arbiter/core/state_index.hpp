#pragma once

#include <cstddef>

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Base state indices for common state vector elements
 *
 * This provides a common framework for accessing state vector elements
 * that can be used across different state model implementations.
 */
struct StateIndex {
  // Position indices (3D)
  struct Position {
    static constexpr int X = 0;
    static constexpr int Y = 1;
    static constexpr int Z = 2;

    static constexpr int Begin() { return X; }
    static constexpr int Size() { return 3; }
  };

  // Orientation indices (quaternion)
  struct Quaternion {
    static constexpr int W = 3;
    static constexpr int X = 4;
    static constexpr int Y = 5;
    static constexpr int Z = 6;

    static constexpr int Begin() { return W; }
    static constexpr int Size() { return 4; }
  };

  // Linear velocity indices
  struct LinearVelocity {
    static constexpr int X = 7;
    static constexpr int Y = 8;
    static constexpr int Z = 9;

    static constexpr int Begin() { return X; }
    static constexpr int Size() { return 3; }
  };

  // Angular velocity indices
  struct AngularVelocity {
    static constexpr int X = 10;
    static constexpr int Y = 11;
    static constexpr int Z = 12;

    static constexpr int Begin() { return X; }
    static constexpr int Size() { return 3; }
  };

  // Linear acceleration indices
  struct LinearAcceleration {
    static constexpr int X = 13;
    static constexpr int Y = 14;
    static constexpr int Z = 15;

    static constexpr int Begin() { return X; }
    static constexpr int Size() { return 3; }
  };

  // Angular acceleration indices
  struct AngularAcceleration {
    static constexpr int X = 16;
    static constexpr int Y = 17;
    static constexpr int Z = 18;

    static constexpr int Begin() { return X; }
    static constexpr int Size() { return 3; }
  };

  // Total size of the state vector for a full rigid body state
  static constexpr std::size_t kFullStateSize = 19;
};

} // namespace core
} // namespace kinematic_arbiter
