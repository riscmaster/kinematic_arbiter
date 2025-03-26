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
 * @brief Runtime function to get measurement dimension from sensor type
 *
 * This function provides the same information as the MeasurementDimension template
 * but can be used at runtime with non-constexpr types.
 *
 * @param type The sensor type to query
 * @return int The measurement dimension for the given sensor type
 */
inline int GetMeasurementDimension(SensorType type) {
  switch (type) {
    case SensorType::Position:     return MeasurementDimension<SensorType::Position>::value;
    case SensorType::Pose:         return MeasurementDimension<SensorType::Pose>::value;
    case SensorType::BodyVelocity: return MeasurementDimension<SensorType::BodyVelocity>::value;
    case SensorType::Imu:          return MeasurementDimension<SensorType::Imu>::value;
    case SensorType::Unknown:      return MeasurementDimension<SensorType::Unknown>::value;
  }
  return -1; // Unknown type
}

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
