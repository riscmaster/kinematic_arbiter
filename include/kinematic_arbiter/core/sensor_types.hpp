#ifndef KINEMATIC_ARBITER_CORE_SENSOR_TYPES_HPP_
#define KINEMATIC_ARBITER_CORE_SENSOR_TYPES_HPP_

#include <string>
#include <Eigen/Dense>

namespace kinematic_arbiter {
namespace core {

/**
 * @brief Enumeration of sensor types with built-in dimension information
 */
enum class SensorType {
  Position,     // 3D position (x, y, z)
  Pose,         // 7D pose (position + quaternion)
  BodyVelocity, // 6D velocity (linear + angular)
  Imu,          // 6D IMU (acceleration + angular velocity)
  Unknown       // Unknown type
};

/**
 * @brief Template specialization to get measurement dimension at compile time
 */
template<SensorType Type>
struct MeasurementDimension;

template<> struct MeasurementDimension<SensorType::Position>     { static constexpr int value = 3; };
template<> struct MeasurementDimension<SensorType::Pose>         { static constexpr int value = 7; };
template<> struct MeasurementDimension<SensorType::BodyVelocity> { static constexpr int value = 6; };
template<> struct MeasurementDimension<SensorType::Imu>          { static constexpr int value = 6; };
template<> struct MeasurementDimension<SensorType::Unknown>      { static constexpr int value = -1; };

/**
 * @brief Get measurement vector type for a sensor type
 */
template<SensorType Type>
using MeasurementVector = Eigen::Matrix<double, MeasurementDimension<Type>::value, 1>;

/**
 * @brief Convert sensor type to string
 */
inline std::string SensorTypeToString(SensorType type) {
  switch (type) {
    case SensorType::Position:     return "Position";
    case SensorType::Pose:         return "Pose";
    case SensorType::BodyVelocity: return "BodyVelocity";
    case SensorType::Imu:          return "Imu";
    case SensorType::Unknown:      return "Unknown";
  }
  return "Unknown";
}

} // namespace core
} // namespace kinematic_arbiter

#endif // KINEMATIC_ARBITER_CORE_SENSOR_TYPES_HPP_
