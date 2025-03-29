"""Launch file for the Figure-8 simulator component test."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch_testing.actions import ReadyToTest
import unittest
import launch
import launch_testing


def generate_launch_description():
    """
    Create and return a launch description for the Figure-8 simulator test.

    This launches just the figure8_simulator_node to verify its functionality.

    Returns:
        LaunchDescription: The complete launch description
    """
    package_name = "kinematic_arbiter"

    # Get the path to the test config file
    config_file = os.path.join(
        get_package_share_directory(package_name),
        "test",
        "config",
        "figure8_test_config.yaml",
    )

    # Print the config file for debugging
    print(f"Using configuration file: {config_file}")

    # Launch the simulator node
    simulator_node = Node(
        package=package_name,
        executable="figure8_simulator_node",
        name="figure8_simulator",
        parameters=[config_file],
        output="screen",
    )

    # Add a command to list topics
    list_topics_cmd = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            'sleep 3 && echo "\nACTIVE TOPICS:" && ros2 topic list',
        ],
        output="screen",
        name="list_topics",
    )

    # Add a delay before starting the test
    ready_to_test = TimerAction(period=3.0, actions=[ReadyToTest()])

    return LaunchDescription([simulator_node, list_topics_cmd, ready_to_test])


def generate_test_description():
    """Generate the test description with the figure8_simulator node."""
    figure8_simulator_node = Node(
        package="kinematic_arbiter",
        executable="figure8_simulator_node",
        name="figure8_simulator",
        parameters=[{"config_file_path": "/path/to/config.yaml"}],
    )

    return launch.LaunchDescription(
        [figure8_simulator_node, launch_testing.actions.ReadyToTest()]
    )


class TestFigure8Simulator(unittest.TestCase):
    """Test case for verifying the Figure-8 simulator node functionality."""

    def test_simulator_node(self, proc_output):
        """Test that the simulator node launches and runs correctly."""
        # Basic test to see if the process started
        proc_output.assertWaitFor("Figure8 simulator initialized", timeout=10)
