#!/usr/bin/env python3

# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Mediated Kalman Filter node for the single DOF demo with dynamic parameter adjustment."""

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

from single_dof_demo.core.mediated_kalman_filter import (
    MediatedKalmanFilter,
    Mediation,
)


class MediatedFilterNode(Node):
    """Mediated Kalman Filter node for the single DOF demo with dynamic parameter adjustment."""

    def __init__(self):
        """Initialize the Mediated Kalman Filter node."""
        super().__init__("mediated_filter_node")

        # Use ReentrantCallbackGroup to allow concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Default parameter values
        self.default_params = {
            "process_measurement_ratio": 1.8,
            "sample_window": 20,
            "mediation_mode": Mediation.ADJUST_STATE,
            "model_frequency": 0.0,
            "model_amplitude": 0.0,
        }

        # Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    "process_measurement_ratio",
                    self.default_params["process_measurement_ratio"],
                    self._create_float_descriptor(
                        0.0, 100.0, "Process to measurement noise ratio (ζ)"
                    ),
                ),
                (
                    "sample_window",
                    self.default_params["sample_window"],
                    ParameterDescriptor(
                        description="Number of samples for noise estimation (n)"
                    ),
                ),
                (
                    "mediation_mode",
                    self.default_params["mediation_mode"],
                    ParameterDescriptor(
                        description="Mediation behavior (0=ADJUST_STATE, 1=ADJUST_MEASUREMENT, 2=REJECT_MEASUREMENT, 3=NO_ACTION)"
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
            PointStamped, "mediated_state_estimate", 10
        )
        self.state_upper_bound_pub = self.create_publisher(
            PointStamped, "mediated_state_upper_bound", 10
        )
        self.state_lower_bound_pub = self.create_publisher(
            PointStamped, "mediated_state_lower_bound", 10
        )
        self.measurement_upper_bound_pub = self.create_publisher(
            PointStamped, "mediated_measurement_upper_bound", 10
        )
        self.measurement_lower_bound_pub = self.create_publisher(
            PointStamped, "mediated_measurement_lower_bound", 10
        )
        self.diagnostics_pub = self.create_publisher(
            DiagnosticStatus, "mediated_filter_status", 10
        )
        self.mediation_point_pub = self.create_publisher(
            PointStamped, "mediated_mediation_point", 10
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

        # Initialize filter
        self._init_filter()
        self.initial_time = self.get_clock().now()

        self.get_logger().info("Mediated Kalman Filter node initialized")

    def _create_float_descriptor(self, min_val, max_val, description):
        """Create a float parameter descriptor."""
        return ParameterDescriptor(
            floating_point_range=[
                FloatingPointRange(from_value=min_val, to_value=max_val)
            ],
            description=description,
        )

    def parameters_callback(self, params):
        """Handle parameter updates."""
        result = SetParametersResult(successful=True)

        for param in params:
            try:
                if param.name == "process_measurement_ratio":
                    if param.value < 0.0:
                        raise ValueError(
                            "Process measurement ratio must be non-negative"
                        )
                    self.filter.set_process_measurement_ratio(param.value)
                    self.get_logger().info(
                        f"Updated process measurement ratio to {param.value}"
                    )
                elif param.name == "sample_window":
                    if param.value <= 0:
                        raise ValueError("Sample window must be positive")
                    self.filter.set_sample_window(param.value)
                    self.get_logger().info(
                        f"Updated sample window to {param.value}"
                    )
                elif param.name == "mediation_mode":
                    if param.value not in [0, 1, 2, 3]:
                        raise ValueError("Invalid mediation mode")
                    self.filter.set_mediation_behavior(param.value)
                    self.get_logger().info(
                        f"Updated mediation mode to {param.value}"
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

    def _init_filter(self):
        """Initialize the Mediated Kalman filter."""
        self.filter = MediatedKalmanFilter(
            process_to_measurement_ratio=self.get_parameter(
                "process_measurement_ratio"
            ).value,
            sample_window=self.get_parameter("sample_window").value,
            mediation=self.get_parameter("mediation_mode").value,
            frequency=self.get_parameter("model_frequency").value,
            amplitude=self.get_parameter("model_amplitude").value,
        )

    def measurement_callback(self, msg):
        """Process incoming measurements."""
        # Extract measurement from message
        measurement = msg.point.x

        # Skip invalid measurements
        if math.isnan(measurement):
            self.get_logger().warn("Received NaN measurement, skipping update")
            return

        # Calculate current time in seconds
        current_time = self.get_clock().now() - self.initial_time
        time_secs = current_time.nanoseconds * 1e-9

        # Update filter with measurement
        output = self.filter.update(measurement, time_secs)

        # Publish state estimate
        state_msg = PointStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="filter_frame"
            ),
            point=Point(x=output.final.state.value, y=0.0, z=0.0),
        )
        self.state_pub.publish(state_msg)

        # Publish state upper bound
        state_upper_bound = output.final.state.value + output.final.state.bound
        state_upper_msg = PointStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="filter_frame"
            ),
            point=Point(x=state_upper_bound, y=0.0, z=0.0),
        )
        self.state_upper_bound_pub.publish(state_upper_msg)

        # Publish state lower bound
        state_lower_bound = output.final.state.value - output.final.state.bound
        state_lower_msg = PointStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="filter_frame"
            ),
            point=Point(x=state_lower_bound, y=0.0, z=0.0),
        )
        self.state_lower_bound_pub.publish(state_lower_msg)

        # Publish measurement upper bound
        meas_upper_bound = (
            output.final.measurement.value + output.final.measurement.bound
        )
        meas_upper_msg = PointStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="filter_frame"
            ),
            point=Point(x=meas_upper_bound, y=0.0, z=0.0),
        )
        self.measurement_upper_bound_pub.publish(meas_upper_msg)

        # Publish measurement lower bound
        meas_lower_bound = (
            output.final.measurement.value - output.final.measurement.bound
        )
        meas_lower_msg = PointStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(), frame_id="filter_frame"
            ),
            point=Point(x=meas_lower_bound, y=0.0, z=0.0),
        )
        self.measurement_lower_bound_pub.publish(meas_lower_msg)

        # Publish mediation point if mediation was detected
        if output.mediation_detected:
            mediation_point_msg = PointStamped(
                header=Header(
                    stamp=self.get_clock().now().to_msg(),
                    frame_id="filter_frame",
                ),
                point=Point(x=output.mediation_point, y=0.0, z=0.0),
            )
            self.mediation_point_pub.publish(mediation_point_msg)
            self.get_logger().info(
                f"Mediation detected at value: {output.mediation_point}"
            )

        # Publish diagnostics
        self._publish_diagnostics()

    def handle_reset(self, request, response):
        """Handle Mediated Kalman Filter reset requests."""
        self._init_filter()
        self.initial_time = self.get_clock().now()
        response.success = True
        response.message = "Mediated Kalman Filter reset successful"
        return response

    def handle_reset_parameters(self, request, response):
        """Reset all parameters to their default values."""
        try:
            # Set parameters directly
            parameters = []
            for name, value in self.default_params.items():
                param_type = Parameter.Type.DOUBLE
                if name == "mediation_mode":
                    param_type = Parameter.Type.INTEGER

                parameters.append(Parameter(name, param_type, value))

            self.set_parameters(parameters)

            for name, value in self.default_params.items():
                self.get_logger().info(f"Reset {name} to {value}")

            # Update filter with default values
            self.filter.set_process_measurement_ratio(
                self.default_params["process_measurement_ratio"]
            )
            self.filter.set_sample_window(self.default_params["sample_window"])
            self.filter.set_mediation_behavior(
                self.default_params["mediation_mode"]
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
        msg.name = "Mediated Kalman Filter"
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
                key="process_measurement_ratio",
                value=str(self.filter.scale),
            ),
            KeyValue(
                key="sample_window",
                value=str(self.filter.sample_window),
            ),
            KeyValue(
                key="mediation_active",
                value=str(self.filter.mediation),
            ),
            KeyValue(
                key="mediation_mode",
                value=str(self.filter.mediation_behavior),
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
    """Start the Mediated Kalman Filter node and begin processing."""
    rclpy.init(args=args)
    node = MediatedFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
