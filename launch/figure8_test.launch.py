#!/usr/bin/env python3
"""
Launch file for the Figure 8 simulation.

This launch file sets up a complete Figure 8 simulation environment using configuration
from a YAML file.
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate the launch description for the Figure 8 simulation."""
    # Get path to the package
    pkg_share = get_package_share_directory("kinematic_arbiter")

    # Define paths relative to package
    config_dir = os.path.join(pkg_share, "config")
    rviz_dir = os.path.join(pkg_share, "rviz")

    # Explicitly define paths to configuration files
    config_file = os.path.join(config_dir, "figure8_simulation.yaml")
    foxglove_layout = os.path.join(config_dir, "figure8_layout.json")
    rviz_config = os.path.join(rviz_dir, "figure8_visualization.rviz")

    # Print configuration paths for debugging
    print(f"Using configuration file: {config_file}")
    print(f"Using RViz config: {rviz_config}")
    print(f"Using Foxglove layout: {foxglove_layout}")

    # Define the config file argument with a default
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=config_file,
        description="Path to the configuration YAML file",
    )

    # Debug mode
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug output"
    )

    # Foxglove Studio argument
    use_foxglove_studio_arg = DeclareLaunchArgument(
        "use_foxglove_studio",
        default_value="false",
        description="Start Foxglove Studio (requires installation)",
    )

    # RViz config
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=rviz_config,
        description="Path to RViz configuration file",
    )

    # Debug output
    debug_output = LogInfo(
        msg=["Debug mode enabled"],
        condition=IfCondition(LaunchConfiguration("debug")),
    )

    # Figure 8 simulator node
    simulator_node = Node(
        package="kinematic_arbiter",
        executable="figure8_simulator_node",
        name="figure8_simulator",
        parameters=[LaunchConfiguration("config_file")],
        output="screen",
    )

    # Kinematic arbiter node - wrapped with a timer to delay startup
    arbiter_node = Node(
        package="kinematic_arbiter",
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
        parameters=[LaunchConfiguration("config_file")],
        output="screen",
    )

    # Delay the arbiter node startup to allow TF tree to be established
    delayed_arbiter_node = TimerAction(
        period=2.0,  # 2-second delay
        actions=[
            LogInfo(
                msg=[
                    "Starting kinematic_arbiter_node after delay to establish TF tree..."
                ]
            ),
            arbiter_node,
        ],
    )

    # Add RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    # Foxglove Bridge Node
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        parameters=[
            {"port": 8765},
            {"address": "0.0.0.0"},
            {"tls": False},
            {"topic_whitelist": [".*"]},
        ],
        output="screen",
    )

    # Foxglove Studio (optional)
    foxglove_studio_process = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration("use_foxglove_studio")),
        cmd=["foxglove-studio", "--load-layout", foxglove_layout],
        name="foxglove_studio",
        output="screen",
    )

    return LaunchDescription(
        [
            # Launch arguments
            config_arg,
            debug_arg,
            use_foxglove_studio_arg,
            rviz_config_arg,
            # Debug output
            debug_output,
            # Nodes
            simulator_node,
            # Use the delayed version instead of direct arbiter_node
            delayed_arbiter_node,
            rviz_node,
            foxglove_bridge_node,
            foxglove_studio_process,
        ]
    )
