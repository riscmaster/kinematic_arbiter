#!/usr/bin/env python3

# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Signal generator node for the simplified demo."""

import rclpy
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.node import Node

from kinematic_arbiter.single_dof_demo.core.signal_generator import (
    SignalParams,
    SingleDofSignalGenerator,
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
                    "publishing_rate",
                    20.0,
                    self._create_float_descriptor(
                        0.1, 100.0, "Rate at which signals are published (Hz)"
                    ),
                ),
                (
                    "max_frequency",
                    1.0,
                    self._create_float_descriptor(
                        0.1, 100.0, "Maximum frequency for signal components"
                    ),
                ),
                (
                    "max_amplitude",
                    1.0,
                    self._create_float_descriptor(
                        0.0, 10.0, "Maximum amplitude for signal components"
                    ),
                ),
                (
                    "number_of_signals",
                    10,
                    ParameterDescriptor(
                        description="Number of sinusoidal components"
                    ),
                ),
            ],
        )

        # Publishers
        self.noisy_publisher = self.create_publisher(
            PointStamped, "raw_measurements", 10
        )
        self.clean_publisher = self.create_publisher(
            PointStamped, "true_signal", 10
        )

        # Initialize signal generator parameters
        self.signal_params = SignalParams()
        self.signal_params.max_frequency = self.get_parameter(
            "max_frequency"
        ).value
        self.signal_params.max_amplitude = self.get_parameter(
            "max_amplitude"
        ).value
        self.signal_params.number_of_signals = self.get_parameter(
            "number_of_signals"
        ).value

        # Create signal generator
        self.signal_generator = SingleDofSignalGenerator(self.signal_params)

        # Initialize time
        self.initial_time = self.get_clock().now()

        # Create timer for signal publishing
        publishing_rate = self.get_parameter("publishing_rate").value
        timer_period = 1.0 / publishing_rate
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
        """Publish clean and noisy measurements."""
        # Update current time
        current_time = self.get_clock().now() - self.initial_time

        # Generate signals
        clean_signal, noisy_signal = self.signal_generator.generate_signal(
            current_time.nanoseconds * 1e-9
        )

        # Get current ROS time for message timestamp
        current_stamp = self.get_clock().now().to_msg()

        # Create and publish signal messages
        clean_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id="world"),
            point=Point(x=clean_signal, y=0.0, z=0.0),
        )
        self.clean_publisher.publish(clean_msg)

        noisy_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id="world"),
            point=Point(x=noisy_signal, y=0.0, z=0.0),
        )
        self.noisy_publisher.publish(noisy_msg)


def main(args=None):
    """Run the signal generator node."""
    rclpy.init(args=args)
    node = SignalGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
