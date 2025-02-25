# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Filter node for the simplified demo."""

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from action import FilterMediation
from kinematic_arbiter.single_dof_demo.core.mediated_kalman_filter import (
    MedianFilter,
)
from kinematic_arbiter.single_dof_demo.ros2.domain_models import (
    State,
)


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
                    "process_measurement_ratio",
                    0.25,
                    self._create_float_descriptor(
                        0.0, 100.0, "Process to measurement ratio"
                    ),
                ),
                (
                    "window_time",
                    0.01,
                    self._create_float_descriptor(
                        0.0, 1.0, "Window time for adaptation"
                    ),
                ),
                ("mediation_mode", 0),
            ],
        )

        # Initialize filter
        self._init_filter()

        # Subscribers
        self.measurement_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "raw_measurements",
            self.measurement_callback,
            10,
            callback_group=self.callback_group,
        )

        # Publishers
        self.filtered_pub = self.create_publisher(
            PoseWithCovarianceStamped, "filtered_state", 10
        )
        self.processed_measurement_pub = self.create_publisher(
            PoseWithCovarianceStamped, "processed_measurement", 10
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

        # Action server
        self._mediation_action_server = ActionServer(
            self,
            FilterMediation,
            "set_mediation_mode",
            self.handle_mediation_request,
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
        self.filter = MedianFilter(
            process_to_measurement_ratio=self.get_parameter(
                "process_measurement_ratio"
            ).value,
            window_time=self.get_parameter("window_time").value,
            mediation=self.get_parameter("mediation_mode").value,
        )

    def measurement_callback(self, msg):
        """Process incoming measurement messages and update the filter."""
        # Convert message to state
        measurement = State(
            value=msg.pose.pose.position.x, variance=msg.pose.covariance[0]
        )

        # Update filter
        output = self.filter.update(
            measurement=measurement.value,
            t=self.get_clock().now().nanoseconds * 1e-9,
        )

        # Publish filtered state
        filtered_msg = PoseWithCovarianceStamped()
        filtered_msg.header = msg.header
        filtered_msg.pose.pose.position.x = output.final.state.value
        filtered_msg.pose.covariance[0] = self.filter.state_variance
        self.filtered_pub.publish(filtered_msg)

        # Publish processed measurement
        processed_msg = PoseWithCovarianceStamped()
        processed_msg.header = msg.header
        processed_msg.pose.pose.position.x = output.final.measurement.value
        processed_msg.pose.covariance[0] = self.filter.measurement_variance
        self.processed_measurement_pub.publish(processed_msg)

        # Publish diagnostics
        self._publish_diagnostics()

    def handle_reset(self, request, response):
        """Handle filter reset requests."""
        self._init_filter()
        response.success = True
        response.message = "Filter reset successful"
        return response

    async def handle_mediation_request(self, goal_handle):
        """Handle mediation mode change requests."""
        feedback_msg = FilterMediation.Feedback()

        try:
            # Update mediation mode
            self.filter.mediation = goal_handle.request.requested_mode

            # Simulate transition
            feedback_msg.progress = 1.0
            feedback_msg.status = "Mediation mode updated"
            goal_handle.publish_feedback(feedback_msg)

            goal_handle.succeed()

            result = FilterMediation.Result()
            result.success = True
            result.current_mode = self.filter.mediation
            result.message = "Mediation mode changed successfully"
            return result

        except Exception as e:
            goal_handle.abort()
            result = FilterMediation.Result()
            result.success = False
            result.current_mode = self.filter.mediation
            result.message = f"Failed to change mediation mode: {str(e)}"
            return result

    def _publish_diagnostics(self):
        """Publish diagnostics for the filter."""
        msg = DiagnosticStatus()
        msg.level = DiagnosticStatus.OK
        msg.name = "Mediated Kalman Filter"
        msg.message = f"Mediation Mode: {self.filter.mediation}"
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
