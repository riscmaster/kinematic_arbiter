# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Kinematic Arbiter package component."""

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from ..core.generate_signals import SignalParams, generate_signals
from ..core.mediated_kalman_filter import MediatedKalmanFilter
from .domain_models import FilterParameters, MediationMode, State
from .message_converters import (
    state_to_pose_with_covariance_stamped,
)


class SimplifiedDemoNode(Node):
    """Node that demonstrates a simplified kinematic filter implementation."""

    def __init__(self):
        """Initialize simplified demo node with publishers and parameters."""
        super().__init__("simplified_demo")

        # Use ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Publishers
        self.filtered_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "filtered_state", 10
        )
        self.measurement_publisher = self.create_publisher(
            PoseWithCovarianceStamped, "processed_measurement", 10
        )

        # Parameter descriptors
        pm_ratio_descriptor = ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=0.0, to_value=100.0)
            ],
            description="Process to measurement ratio for the filter",
        )

        window_descriptor = ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=0.0, to_value=1.0)
            ],
            description="Time window for filter adaptation",
        )

        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("process_measurement_ratio", 0.25, pm_ratio_descriptor),
                ("window_time", 0.01, window_descriptor),
                ("mediation_mode", MediationMode.ADJUST_STATE.value),
            ],
        )

        # Initialize filter with parameters
        filter_params = FilterParameters(
            process_measurement_ratio=self.get_parameter(
                "process_measurement_ratio"
            ).value,
            window_time=self.get_parameter("window_time").value,
        )

        self.filter = MediatedKalmanFilter(
            process_to_measurement_ratio=(
                filter_params.process_measurement_ratio
            ),
            window_time=filter_params.window_time,
            mediation=self.get_parameter("mediation_mode").value,
        )

        # Initialize signal generator
        self.signal_params = SignalParams()
        self.signal, self.noisy_signal, self.time = generate_signals(
            self.signal_params
        )
        self.current_index = 0

        # Create timer for signal publishing
        timer_period = 1.0 / self.signal_params.sample_frequency
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """Generate and publish signal data at regular intervals."""
        if self.current_index >= len(self.time):
            self.current_index = 0

        # Get current signal values
        current_time = self.time[self.current_index]
        noisy_value = self.noisy_signal[self.current_index]

        # Create measurement state
        measurement = State(
            value=noisy_value, variance=self.signal_params.max_amplitude
        )

        # Update filter
        output = self.filter.update(
            measurement=measurement.value, t=current_time
        )

        # Get current ROS time for message timestamps
        current_stamp = self.get_clock().now().to_msg()

        # Convert filter output to ROS messages and publish
        filtered_state = State(
            value=output.final.state.value, variance=self.filter.state_variance
        )
        filtered_msg = state_to_pose_with_covariance_stamped(
            filtered_state, current_stamp
        )
        self.filtered_publisher.publish(filtered_msg)

        processed_measurement = State(
            value=output.final.measurement.value,
            variance=self.filter.measurement_variance,
        )
        measurement_msg = state_to_pose_with_covariance_stamped(
            processed_measurement, current_stamp
        )
        self.measurement_publisher.publish(measurement_msg)

        self.current_index += 1


def main(args=None):
    """Start the simplified demo node and begin processing."""
    rclpy.init(args=args)
    node = SimplifiedDemoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
