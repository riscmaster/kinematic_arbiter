#!/usr/bin/env python3

# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Kalman Filter node for the single DOF demo with dynamic parameter adjustment."""

import math
import rclpy
from rclpy.parameter import Parameter
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from rcl_interfaces.msg import (
    FloatingPointRange,
    ParameterDescriptor,
    SetParametersResult,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

# Now import using absolute imports - should work with ROS 2 environment
from single_dof_demo.core.kalman_filter import KalmanFilter


class KalmanFilterNode(Node):
    """Kalman Filter node for the single DOF demo with dynamic parameter adjustment."""

    def __init__(self):
        """Initialize the Kalman Filter node."""
        super().__init__("kalman_filter_node")

        # Use ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Default parameter values
        self.default_params = {
            "process_noise": 1.0,
            "measurement_noise": 5.0,
            "model_frequency": 0.0,
            "model_amplitude": 0.0,
        }

        # Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "process_noise",
                    self.default_params["process_noise"],
                    self._create_float_descriptor(
                        0.0, 100.0, "Process noise variance"
                    ),
                ),
                (
                    "measurement_noise",
                    self.default_params["measurement_noise"],
                    self._create_float_descriptor(
                        0.0, 100.0, "Measurement noise variance"
                    ),
                ),
                (
                    "model_frequency",
                    self.default_params["model_frequency"],
                    self._create_float_descriptor(
                        0.0, 20.0, "Frequency of the model in Hz"
                    ),
                ),
                (
                    "model_amplitude",
                    self.default_params["model_amplitude"],
                    self._create_float_descriptor(
                        0.0, 10.0, "Amplitude of the model"
                    ),
                ),
            ],
        )

        # Add parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

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
            "~/reset_filter",
            self.handle_reset,
            callback_group=self.callback_group,
        )

        self.reset_params_service = self.create_service(
            Trigger,
            "~/reset_parameters",
            self.handle_reset_parameters,
            callback_group=self.callback_group,
        )

        # Initialize time tracking
        self.initial_time = self.get_clock().now()

        self.get_logger().info("Kalman Filter node initialized")

    def _create_float_descriptor(self, min_val, max_val, description):
        """Create a float parameter descriptor."""
        return ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=min_val, to_value=max_val)
            ],
            description=description,
        )

    def parameters_callback(self, params):
        """Handle parameter changes."""
        result = SetParametersResult(successful=True)

        for param in params:
            try:
                if param.name == "process_noise":
                    if param.value < 0.0:
                        raise ValueError("Process noise must be non-negative")
                    self.filter.set_process_noise(param.value)
                    self.get_logger().info(
                        f"Updated process noise to {param.value}"
                    )
                elif param.name == "measurement_noise":
                    if param.value < 0.0:
                        raise ValueError(
                            "Measurement noise must be non-negative"
                        )
                    self.filter.set_measurement_noise(param.value)
                    self.get_logger().info(
                        f"Updated measurement noise to {param.value}"
                    )
                elif param.name == "model_frequency":
                    self.filter.set_frequency(param.value)
                    self.get_logger().info(
                        f"Updated model frequency to {param.value}"
                    )
                elif param.name == "model_amplitude":
                    self.filter.set_amplitude(param.value)
                    self.get_logger().info(
                        f"Updated model amplitude to {param.value}"
                    )
            except Exception as e:
                self.get_logger().error(
                    f"Error setting parameter {param.name}: {str(e)}"
                )
                result.successful = False
                result.reason = str(e)
                return result

        return result

    def _init_filter(self):
        """Initialize the Kalman filter."""
        self.filter = KalmanFilter(
            process_noise=self.get_parameter("process_noise").value,
            measurement_noise=self.get_parameter("measurement_noise").value,
            frequency=self.get_parameter("model_frequency").value,
            amplitude=self.get_parameter("model_amplitude").value,
        )

    def measurement_callback(self, msg):
        """Process incoming measurement messages and update the filter."""
        # Extract measurement from message
        measurement = msg.point.x

        # Skip invalid measurements
        if math.isnan(measurement):
            self.get_logger().warn("Received NaN measurement, skipping update")
            return

        # Calculate current time in seconds
        current_time = self.get_clock().now() - self.initial_time
        time_secs = current_time.nanoseconds * 1e-9

        # Update filter
        output = self.filter.update(
            measurement=measurement, time_secs=time_secs
        )

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
        """Handle Kalman Filter reset requests."""
        self._init_filter()
        self.initial_time = self.get_clock().now()
        response.success = True
        response.message = "Kalman Filter reset successful"
        return response

    def handle_reset_parameters(self, request, response):
        """Reset all parameters to their default values."""
        try:
            # Set parameters directly
            parameters = []
            for name, value in self.default_params.items():
                parameters.append(
                    Parameter(name, Parameter.Type.DOUBLE, value)
                )

            self.set_parameters(parameters)

            for name, value in self.default_params.items():
                self.get_logger().info(f"Reset {name} to {value}")

            # Update filter with default values
            self.filter.set_process_noise(self.default_params["process_noise"])
            self.filter.set_measurement_noise(
                self.default_params["measurement_noise"]
            )
            self.filter.set_frequency(self.default_params["model_frequency"])
            self.filter.set_amplitude(self.default_params["model_amplitude"])

            self.get_logger().info("All parameters reset to default values")
            response.success = True
            response.message = "Parameters reset successful"
        except Exception as e:
            self.get_logger().error(f"Error resetting parameters: {str(e)}")
            response.success = False
            response.message = f"Error: {str(e)}"

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
            KeyValue(
                key="process_variance",
                value=str(self.filter.process_variance),
            ),
            KeyValue(
                key="model_frequency",
                value=str(self.filter.frequency),
            ),
            KeyValue(
                key="model_amplitude",
                value=str(self.filter.amplitude),
            ),
        ]
        self.diagnostics_pub.publish(msg)


def main(args=None):
    """Start the Kalman Filter node and begin processing."""
    rclpy.init(args=args)
    node = KalmanFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
