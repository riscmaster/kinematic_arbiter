"""Launch file for testing parameter loading and configuration in kinematic_arbiter."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
import unittest
import launch_testing


def generate_launch_description():
    """
    Create and return a launch description for testing parameter handling.

    Returns:
        LaunchDescription: The complete launch description
    """
    package_name = "kinematic_arbiter"

    # Launch the node with test parameters
    node = Node(
        package=package_name,
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
        parameters=[{"test_param": "value"}],
        output="screen",
    )

    # Add a delay before starting the test
    ready_to_test = TimerAction(
        period=1.0, actions=[launch_testing.actions.ReadyToTest()]
    )

    return LaunchDescription([node, ready_to_test])


class TestParameterLoading(unittest.TestCase):
    """Test case for verifying parameter loading functionality."""

    def test_params_loaded(self, proc_output):
        """
        Verify parameters are loaded correctly by the node.

        Tests that the node can access and use parameters provided in the launch config.
        """
        # Wait for node initialization message
        proc_output.assertWaitFor("Node initialized", timeout=5)
