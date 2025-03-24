# Kinematic Arbiter

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Test Status](https://github.com/riscmaster/kinematic_arbiter/actions/workflows/ci_actions.yml/badge.svg)](https://github.com/riscmaster/kinematic_arbiter/actions/workflows/ci_actions.yml)

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

> **Note**: Our C++ implementation uses optimal filtering techniques to handle out-of-sequence measurements and other practical challenges common in robotics applications. This approach enables robust state estimation even when measurements arrive late or sensor data is temporarily unreliable. For more details on this approach, see [Optimal Update with Out-of-Sequence Measurements](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=546f758d8981b8c8420a0b630f90cd2cce3606b4), which presents algorithms based on BLUE (best linear unbiased estimation) fusion that are optimal for the information available at the time of update.

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

## Demonstrations

### Standalone Python Demo (No ROS 2 Required)

For those who want to explore the Mediated Kalman Filter without ROS 2 dependencies, we provide a standalone Python demo:

```bash
# Navigate to the package source directory
cd src/single_dof_demo

# Run the standalone demo
python3 demo.py
```

This interactive demo provides:
- Side-by-side comparison of standard and mediated Kalman filters
- Real-time parameter tuning via sliders
- Visualization of measurement validation and mediation effects
- Different mediation strategies (state adjustment, measurement adjustment, rejection)

![Standalone Demo](https://github.com/riscmaster/kinematic_arbiter/blob/master/doc/StandaloneDemo.gif)

The standalone demo is an excellent way to understand the core concepts of the Mediated Kalman Filter before integrating it into a ROS 2 system.

### ROS 2 Single DOF Demo

The package includes a comprehensive single degree of freedom (DOF) demonstration that showcases:
- Signal generation with configurable parameters
- Kalman filtering with dynamic parameter adjustment
- Foxglove integration for visualization

Run the demo with:
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the demo
ros2 launch kinematic_arbiter single_dof_demo.launch.py
```

To also launch Foxglove Studio (if installed):
```bash
ros2 launch kinematic_arbiter single_dof_demo.launch.py use_foxglove_studio:=true
```

#### Demo Components

The Single DOF Demo consists of three main nodes:

1. **Signal Generator Node**: Generates synthetic signals with configurable frequency, amplitude, and noise
   - Adjustable parameters: publishing rate, max frequency, max amplitude, noise level, number of signals
   - Services: reset generator, reset parameters

2. **Kalman Filter Node**: Processes measurements using a standard Kalman filter
   - Adjustable parameters: process noise, measurement noise, model frequency, model amplitude
   - Services: reset filter, reset parameters
   - Publishes: state estimate, state bounds, measurement bounds, diagnostics

3. **Mediated Filter Node**: Processes measurements using the Mediated Kalman filter
   - Adjustable parameters: process measurement ratio (ζ), sample window (n), mediation mode
   - Services: reset filter, reset parameters
   - Publishes: state estimate, state bounds, measurement bounds, mediation points, diagnostics

#### Visualization with Foxglove

The demo includes Foxglove Bridge integration for advanced visualization:
1. Connect to Foxglove Studio at `ws://localhost:8765`
2. Import the provided layout from `config/single_dof_demo.json`
3. Observe real-time signal generation, filtering, and parameter effects

![Foxglove Studio Visualization](https://github.com/riscmaster/kinematic_arbiter/blob/master/doc/SingleDofDemo.gif)

## Prerequisites

### System Requirements
- Ubuntu 22.04
- Python 3.10+
- ROS 2 Humble (for ROS 2 demos only)

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

# Install Foxglove Bridge (for visualization)
sudo apt install ros-humble-foxglove-bridge

# For C++ implementation, ensure you have Eigen
sudo apt install libeigen3-dev
```

## Installation

### Full Installation (Python + C++)

To build the complete package with both Python and C++ components:

```bash
# Clone the repository
git clone https://github.com/riscmaster/kinematic_arbiter.git

# Navigate to your ROS 2 workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select kinematic_arbiter

# Source the workspace
source install/setup.bash
```

### Python-Only Installation

If you're only interested in the Python demonstrations and don't need the C++ implementation:

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws

# Build only the Python components
colcon build --packages-select kinematic_arbiter --cmake-args -DBUILD_PYTHON_ONLY=ON

# Source the workspace
source install/setup.bash
```

This option is useful if:
- You want to quickly try out the demos
- You're primarily interested in the educational aspects of the package
- You're developing or testing only the Python components

## Usage

### Adjusting Parameters

You can adjust the filter parameters at runtime using either of these methods:

#### Method 1: Command Line

```bash
# Adjust the process to measurement noise ratio (ζ) for the mediated filter
ros2 param set /mediated_filter_node process_measurement_ratio 2.0

# Change the mediation mode (0=ADJUST_STATE, 1=ADJUST_MEASUREMENT, 2=REJECT_MEASUREMENT, 3=NO_ACTION)
ros2 param set /mediated_filter_node mediation_mode 0
```

#### Method 2: Foxglove Studio

1. Connect to Foxglove Studio at `ws://localhost:8765`
2. Import the provided layout from `config/single_dof_demo.json`
3. Adjust the parameters in the layout and press enter to apply the changes

### Troubleshooting

#### Foxglove Studio Connection Issues
If you're having trouble connecting to Foxglove Studio:
- Ensure the Foxglove Bridge is running (`ros2 node list` should show `/foxglove_bridge`)
- Check that port 8765 is not blocked by a firewall
- Try restarting the bridge: `ros2 node kill /foxglove_bridge` and relaunch

#### Filter Performance Issues
- If the filter is too responsive to noise, increase the `process_measurement_ratio`
- If the filter is too slow to respond to changes, decrease the `process_measurement_ratio`
- Try different mediation modes to see which works best for your data

## Technical Documentation

The Kinematic Arbiter package includes detailed technical documentation covering the key components:

### Mediated Kalman Filter

The Mediated Kalman Filter addresses two key challenges in practical Kalman filter implementations:

1. **Maintaining linear Gaussian assumptions**: The filter uses a chi-square test to validate measurements against the filter's current state and uncertainty. When measurements violate these assumptions, the filter can:
   - Adjust the state covariance
   - Adjust the measurement covariance
   - Reject the measurement entirely

2. **Simplified tuning**: The filter dynamically estimates both measurement and process noise, requiring only a single tuning parameter (ζ) that represents the ratio between process and measurement noise. This methodology dramatically reduces the degrees of freedom in tuning parameters from scaling quadratically with the number of states and sensor degrees of freedom (as in traditional Kalman filters) to a single scalar value.

The recursive noise estimation follows these equations:

- Measurement noise update: R̂_k = R̂_{k-1} + ((y_k - C_k x̂_k)(y_k - C_k x̂_k)^T - R̂_{k-1}) / n
- Process noise update: Q̂_k = Q̂_{k-1} + (ζ (x̌_k - x̂_k)(x̌_k - x̂_k)^T - Q̂_{k-1}) / n

Where n is the sample window size for noise estimation.

For more details, see the [mediated Kalman filter documentation](doc/mediated_kalman_filter.pdf).

### Rigid Body State Model

The package implements a quaternion-based rigid body state model for accurate 3D dynamics:

- 19-dimensional state vector (position, orientation, velocity, acceleration)
- Quaternion kinematics for robust rotation modeling
- Exponential decay model for acceleration stability
- Derived Jacobian matrices for linearized state updates

This comprehensive approach ensures numerical stability and physical accuracy for state estimation.

For more details, see the [motion model documentation](doc/motion_model.pdf).

### IMU Measurement Model

The IMU model accounts for:

- Sensor offset and misalignment
- Bias estimation using sliding window averaging
- Stationary detection logic for improved reliability
- Compensation for rotational effects, lever arm displacement, and gravity

The model provides detailed Jacobian derivations and practical testing guidelines to ensure robust performance.

For more details, see the [IMU model documentation](doc/iimu_model.pdf).

### Figure-8 Trajectory Generator

For testing purposes, the package includes a configurable Figure-8 trajectory generator:

- Parameterized by maximum velocity, length, width, and angular scale
- Dynamically adjusted orientation using quaternions
- Accurate angular dynamics with Coriolis effects
- Smooth yet challenging motion profile for evaluation purposes

For more details, see the [trajectory generator documentation](doc/figure8_trajectory.pdf).

## Package Structure

```
kinematic_arbiter/
├── config/                  # Configuration files
│   ├── filter_params.yaml   # Filter parameters
│   └── single_dof_demo.json # Foxglove Studio layout
├── doc/                     # Documentation
│   └── mediated_kalman_filter.pdf  # Technical paper
├── launch/                  # Launch files
│   └── single_dof_demo.launch.py   # Demo launcher
├── src/
│   └── single_dof_demo/     # Single DOF demo implementation
│       ├── core/            # Core filter implementations
│       │   ├── filter_output.py       # Output data structures
│       │   ├── kalman_filter.py       # Standard Kalman filter
│       │   ├── mediated_kalman_filter.py  # Mediated filter
│       │   └── signal_generator.py    # Signal generation
│       ├── demo.py          # Standalone Python demo
│       └── ros2/
│           └── nodes/       # ROS 2 nodes
│               ├── kalman_filter_node.py    # Standard filter node
│               ├── mediated_filter_node.py  # Mediated filter node
│               └── signal_generator_node.py # Signal generator node
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This package is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Mediated Kalman Filter](doc/mediated_kalman_filter.pdf)

These additions would make the README more practical and user-friendly, providing concrete guidance for users at different levels of expertise.
