#!/usr/bin/env python3

# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Filter node for the simplified demo."""

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from kinematic_arbiter.single_dof_demo.core.kalman_filter import KalmanFilter


class FilterNode(Node):
    """Filter node for the simplified demo."""

    def __init__(self):
        """Initialize the filter node."""
        super().__init__("filter_node")

        # Use ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "process_noise",
                    0.01,
                    self._create_float_descriptor(
                        0.0, 1.0, "Process noise variance"
                    ),
                ),
                (
                    "measurement_noise",
                    0.1,
                    self._create_float_descriptor(
                        0.0, 1.0, "Measurement noise variance"
                    ),
                ),
            ],
        )

        # Initialize filter
        self._init_filter()

        # Subscribers
        self.measurement_sub = self.create_subscription(
            PointStamped,
            "raw_measurements",
            self.measurement_callback,
            10,
            callback_group=self.callback_group,
        )

        # Publishers
        self.state_pub = self.create_publisher(
            PointStamped, "kalman_state_estimate", 10
        )
        self.state_upper_bound_pub = self.create_publisher(
            PointStamped, "kalman_state_upper_bound", 10
        )
        self.state_lower_bound_pub = self.create_publisher(
            PointStamped, "kalman_state_lower_bound", 10
        )
        self.measurement_upper_bound_pub = self.create_publisher(
            PointStamped, "kalman_measurement_upper_bound", 10
        )
        self.measurement_lower_bound_pub = self.create_publisher(
            PointStamped, "kalman_measurement_lower_bound", 10
        )
        self.diagnostics_pub = self.create_publisher(
            DiagnosticStatus, "filter_status", 10
        )

        # Services
        self.reset_service = self.create_service(
            Trigger,
            "reset_filter",
            self.handle_reset,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Filter node initialized")

    def _create_float_descriptor(self, min_val, max_val, description):
        """Create a float parameter descriptor."""
        return ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=min_val, to_value=max_val)
            ],
            description=description,
        )

    def _init_filter(self):
        """Initialize the Kalman filter."""
        self.filter = KalmanFilter(
            process_noise=self.get_parameter("process_noise").value,
            measurement_noise=self.get_parameter("measurement_noise").value,
        )

    def measurement_callback(self, msg):
        """Process incoming measurement messages and update the filter."""
        # Extract measurement from message
        measurement = msg.point.x

        # Update filter
        output = self.filter.update(measurement=measurement)

        # Get state and bounds
        state_value = output.final.state.value
        state_bound = output.final.state.bound
        measurement_bound = output.final.measurement.bound

        # Get current timestamp and frame_id from input message
        current_stamp = msg.header.stamp
        frame_id = msg.header.frame_id

        # Publish state
        state_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id=frame_id),
            point=Point(x=state_value, y=0.0, z=0.0),
        )
        self.state_pub.publish(state_msg)

        # Publish state bounds
        state_upper_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id=frame_id),
            point=Point(x=state_value + state_bound, y=0.0, z=0.0),
        )
        self.state_upper_bound_pub.publish(state_upper_msg)

        state_lower_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id=frame_id),
            point=Point(x=state_value - state_bound, y=0.0, z=0.0),
        )
        self.state_lower_bound_pub.publish(state_lower_msg)

        # Publish measurement bounds
        measurement_upper_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id=frame_id),
            point=Point(x=measurement + measurement_bound, y=0.0, z=0.0),
        )
        self.measurement_upper_bound_pub.publish(measurement_upper_msg)

        measurement_lower_msg = PointStamped(
            header=Header(stamp=current_stamp, frame_id=frame_id),
            point=Point(x=measurement - measurement_bound, y=0.0, z=0.0),
        )
        self.measurement_lower_bound_pub.publish(measurement_lower_msg)

        # Publish diagnostics
        self._publish_diagnostics()

    def handle_reset(self, request, response):
        """Handle filter reset requests."""
        self._init_filter()
        response.success = True
        response.message = "Filter reset successful"
        return response

    def _publish_diagnostics(self):
        """Publish diagnostics for the filter."""
        msg = DiagnosticStatus()
        msg.level = DiagnosticStatus.OK
        msg.name = "Kalman Filter"
        msg.message = "Filter running normally"
        msg.values = [
            KeyValue(
                key="state_variance", value=str(self.filter.state_variance)
            ),
            KeyValue(
                key="measurement_variance",
                value=str(self.filter.measurement_variance),
            ),
        ]
        self.diagnostics_pub.publish(msg)


def main(args=None):
    """Start the filter node and begin processing."""
    rclpy.init(args=args)
    node = FilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
