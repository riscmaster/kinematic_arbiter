"""Launch file for the Figure-8 tracking test."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
from launch_testing.actions import ReadyToTest


def generate_launch_description():
    """
    Create and return a launch description for the Figure-8 tracking test.

    This launches the figure8_simulator_node and kinematic_arbiter_node with
    the test configuration, then starts the test after a brief delay.

    Returns:
        LaunchDescription: The complete launch description
    """
    # Print the test configuration
    package_name = "kinematic_arbiter"
    config_file = os.path.join(
        get_package_share_directory(package_name),
        "test",
        "config",
        "figure8_test_config.yaml",
    )

    print(f"Using configuration file: {config_file}")
    with open(config_file, "r") as f:
        print("Configuration contents:")
        print(f.read())

    # Launch the simulator node
    simulator_node = Node(
        package=package_name,
        executable="figure8_simulator_node",
        name="figure8_simulator",
        parameters=[config_file],
        output="screen",
    )

    # Launch the kinematic arbiter node
    arbiter_node = Node(
        package=package_name,
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
        parameters=[config_file],
        output="screen",
    )

    # Add a delay before starting the test to allow the nodes to initialize
    ready_to_test = TimerAction(
        period=2.0, actions=[ReadyToTest()]  # 2 second delay
    )

    return LaunchDescription([simulator_node, arbiter_node, ready_to_test])
