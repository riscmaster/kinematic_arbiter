# Kinematic Arbiter

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Test Status](https://github.com/acfr/kinematic_arbiter/actions/workflows/ci_actions.yml/badge.svg)](https://github.com/acfr/kinematic_arbiter/actions/workflows/ci_actions.yml)

## About

The Kinematic Arbiter is a ROS 2 package that implements a sophisticated sensor fusion algorithm for state estimation. It uses a mediated Kalman filter approach to combine multiple sensor inputs and provide robust state estimates.

## Algorithm Overview

The Kinematic Arbiter implements a mediated Kalman filter that addresses two key challenges in state estimation:
1. Maintaining the validity of fundamental Kalman filter assumptions in practice
2. Simplifying the tuning process for non-expert users

### Mediated Kalman Filter
The algorithm extends the traditional Kalman filter by adding a mediation layer that:
- Actively validates measurement consistency using chi-square testing
- Prevents filter divergence by detecting and handling assumption violations
- Dynamically adjusts measurement noise estimates based on observed data
- Maintains a conservative estimate of process noise linked to measurement updates

### Key Features

#### Measurement Validation
- Uses chi-square testing to verify that measurements align with filter predictions
- Detects when filter assumptions break down
- Enables appropriate handling of recoverable and non-recoverable failures
- Maintains filter reliability during challenging conditions

#### Simplified Tuning
Instead of requiring complex manual tuning of multiple parameters, the filter provides:
- Dynamic estimation of measurement noise based on observed data
- Automated process noise estimation linked to measurement updates
- A single scalar parameter (ζ) to adjust the relative weighting between measurement and process noise
- Intuitive tuning focused on measurement confidence levels

### Applications
This approach is particularly useful when:
- Sensor measurements may be unreliable or contain unmodeled disturbances
- Expert tuning knowledge is not available
- Robust state estimation is required
- Filter assumptions need active validation

The algorithm can be applied to both linear and nonlinear systems through extended Kalman filter implementations.

## Prerequisites

### System Requirements
- Ubuntu 22.04
- Python 3.10+
- ROS 2 Humble

### Install ROS 2
Follow the official ROS 2 Humble installation instructions:
[ROS 2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

Make sure to install the version that corresponds to the branch you require, or slight modifications may be required to compile the code.

### Additional Dependencies
```bash
# Install pip
sudo apt install python3-pip

# Install development tools
pip install pre-commit
```

## Installation

### Build from Source
```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone the repository
git clone https://github.com/acfr/kinematic_arbiter.git
cd ..

# Install dependencies
rosdep install -y --from-paths src --ignore-src --rosdistro humble

# Build the package
colcon build --packages-select kinematic_arbiter
```

## Demonstrations

### Simplified 1D Demo
The package includes a simplified one-dimensional implementation that demonstrates:
- Basic filter operation
- Measurement validation
- Parameter tuning effects
- Real-time visualization

Run the demo with:
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the demo
ros2 launch kinematic_arbiter simplified_demo_launch.py
```

## Usage

### Basic Usage
1. Source your ROS 2 workspace:
```bash
source ~/ros2_ws/install/setup.bash
```

2. Launch the demo:
```bash
ros2 launch kinematic_arbiter simplified_demo_launch.py
```

### Configuration
The filter can be configured through:
- ROS 2 parameters
- Launch file arguments
- Real-time parameter updates

### Future Development
- 2D implementation for planar motion
- 3D implementation for full pose estimation
- C++ implementation for performance-critical applications
- Additional sensor type support

## Contributing
Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License
[License Type] - See [LICENSE](LICENSE) for details.

## References
- [Mediated Kalman Filter](doc/mediated_kalman_filter.pdf)
