#!/usr/bin/env python3

# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Signal generator node for the simplified demo."""

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.node import Node

# Update imports to match installed package structure
from kinematic_arbiter.single_dof_demo.core.generate_signals import (
    SignalParams,
    generate_signals,
)
from kinematic_arbiter.single_dof_demo.ros2.domain_models import State
from kinematic_arbiter.single_dof_demo.ros2.message_converters import (
    state_to_pose_with_covariance_stamped,
)


class SignalGeneratorNode(Node):
    """Signal generator node for the simplified demo."""

    def __init__(self):
        """Initialize the signal generator node."""
        super().__init__("signal_generator")

        # Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "frequency",
                    10.0,
                    self._create_float_descriptor(
                        0.1, 100.0, "Signal generation frequency in Hz"
                    ),
                ),
                (
                    "amplitude",
                    1.0,
                    self._create_float_descriptor(
                        0.0, 10.0, "Signal amplitude"
                    ),
                ),
                (
                    "noise_level",
                    0.1,
                    self._create_float_descriptor(0.0, 1.0, "Noise level"),
                ),
            ],
        )

        # Publisher for noisy measurements
        self.measurement_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "raw_measurement", 10
        )

        # Initialize signal generator
        self.signal_params = SignalParams()
        self.signal_params.sample_frequency = self.get_parameter(
            "frequency"
        ).value
        self.signal_params.max_amplitude = self.get_parameter(
            "amplitude"
        ).value
        self.signal_params.noise_amplitude = self.get_parameter(
            "noise_level"
        ).value

        self.signal, self.noisy_signal, self.time = generate_signals(
            self.signal_params
        )
        self.current_index = 0

        # Create timer for signal publishing
        timer_period = 1.0 / self.signal_params.sample_frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Signal generator node initialized")

    def _create_float_descriptor(self, min_val, max_val, description):
        """Create a float parameter descriptor."""
        return ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=min_val, to_value=max_val)
            ],
            description=description,
        )

    def timer_callback(self):
        """Publish noisy measurements."""
        if self.current_index >= len(self.time):
            self.current_index = 0

        # Get current signal values
        noisy_value = self.noisy_signal[self.current_index]

        # Create measurement state
        measurement = State(
            value=noisy_value, variance=self.signal_params.max_amplitude
        )

        # Get current ROS time for message timestamp
        current_stamp = self.get_clock().now().to_msg()

        # Convert to ROS message and publish
        measurement_msg = state_to_pose_with_covariance_stamped(
            measurement, current_stamp
        )
        self.measurement_publisher.publish(measurement_msg)

        self.current_index += 1


def main(args=None):
    """Run the signal generator node."""
    rclpy.init(args=args)
    node = SignalGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
