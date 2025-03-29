"""Launch file for the Kinematic Arbiter component test."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch_testing.actions import ReadyToTest


def generate_launch_description():
    """
    Create and return a launch description for the Kinematic Arbiter test.

    This launches just the kinematic_arbiter_node to verify its functionality.

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

    # Launch the arbiter node
    arbiter_node = Node(
        package=package_name,
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
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

    return LaunchDescription([arbiter_node, list_topics_cmd, ready_to_test])
