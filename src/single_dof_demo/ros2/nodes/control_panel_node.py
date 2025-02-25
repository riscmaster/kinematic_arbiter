# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Control panel node for the simplified demo."""

# !/usr/bin/env python3
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.action.client import GoalStatus
from std_srvs.srv import Trigger
from kinematic_arbiter.action import FilterMediation


class ControlPanelNode(Node):
    """Node providing control panel functionality for the simplified demo."""

    def __init__(self):
        """Initialize the control panel node."""
        super().__init__("control_panel")

        # Action client for mediation mode
        self._mediation_action_client = ActionClient(
            self, FilterMediation, "set_mediation_mode"
        )

        # Service client for filter reset
        self.reset_client = self.create_client(Trigger, "reset_filter")

        # Parameter client for filter node
        self.filter_param_client = self.create_client(
            SetParameters, "/filter_node/set_parameters"
        )

        self.get_logger().info("Control panel node initialized")

    async def set_mediation_mode(self, mode):
        """Set the mediation mode for the filter.

        Args:
            mode: The mediation mode to set

        Returns:
            Result of the mediation mode change request
        """
        goal_msg = FilterMediation.Goal()
        goal_msg.requested_mode = mode

        self.get_logger().info(f"Requesting mediation mode change to {mode}")

        # Wait for action server
        if not self._mediation_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available")
            return False

        # Send goal
        goal_future = self._mediation_action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )

        # Wait for result
        result = await goal_future
        if result.status == GoalStatus.STATUS_ACCEPTED:
            self.get_logger().info("Goal accepted")
            result_future = await result.get_result_async()
            return result_future.result
        else:
            self.get_logger().error("Goal rejected")
            return None

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the mediation mode change action."""
        progress = feedback_msg.feedback.progress
        self.get_logger().info(
            f"Mediation mode transition progress: {progress:.1%}"
        )

    async def reset_filter(self):
        """Reset the filter to initial state.

        Returns:
            bool: True if reset successful, False otherwise
        """
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Reset service not available")
            return False

        request = Trigger.Request()
        response = await self.reset_client.call_async(request)
        return response.success

    async def update_filter_parameters(
        self, process_measurement_ratio=None, window_time=None
    ):
        """Update filter parameters.

        Args:
            process_measurement_ratio: New ratio between process
                and measurement noise
            window_time: New time window for noise estimation

        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.filter_param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Parameter service not available")
            return False

        parameters = []
        if process_measurement_ratio is not None:
            parameters.append(
                Parameter(
                    name="process_measurement_ratio",
                    value=ParameterValue(
                        type=ParameterType.DOUBLE_ARRAY,
                        double_value=process_measurement_ratio,
                    ),
                )
            )

        if window_time is not None:
            parameters.append(
                Parameter(
                    name="window_time",
                    value=ParameterValue(
                        type=ParameterType.DOUBLE_ARRAY,
                        double_value=window_time,
                    ),
                )
            )

        request = SetParameters.Request(parameters=parameters)
        response = await self.filter_param_client.call_async(request)
        return all(result.successful for result in response.results)


def main(args=None):
    """Run the control panel node."""
    rclpy.init(args=args)
    node = ControlPanelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
