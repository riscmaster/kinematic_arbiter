# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Launch configuration for the Kinematic Arbiter demonstration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for the kinematic arbiter demo."""
    pkg_share = get_package_share_directory("kinematic_arbiter")

    # Config files
    filter_params = os.path.join(pkg_share, "config", "filter_params.yaml")
    foxglove_layout = os.path.join(
        pkg_share, "config", "kalman_filter_layout.json"
    )

    # Launch Arguments
    use_foxglove_studio_arg = DeclareLaunchArgument(
        "use_foxglove_studio",
        default_value="false",
        description="Start Foxglove Studio (requires installation)",
    )

    return LaunchDescription(
        [
            use_foxglove_studio_arg,
            # Signal Generator Node
            Node(
                package="kinematic_arbiter",
                executable="signal_generator_node.py",
                name="signal_generator",
                parameters=[filter_params],
                output="screen",
            ),
            # Kalman Filter Node
            Node(
                package="kinematic_arbiter",
                executable="kalman_filter_node.py",
                name="kalman_filter_node",
                parameters=[filter_params],
                output="screen",
            ),
            # Mediated Filter Node
            Node(
                package="kinematic_arbiter",
                executable="mediated_filter_node.py",
                name="mediated_filter_node",
                parameters=[filter_params],
                output="screen",
            ),
            # Control Panel Node
            ExecuteProcess(
                cmd=[
                    "python3",
                    "install/kinematic_arbiter/lib/kinematic_arbiter/"
                    "control_panel_node.py",
                ],
                name="control_panel",
                output="screen",
            ),
            # Foxglove Bridge
            Node(
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
            ),
            # Foxglove Studio (optional)
            ExecuteProcess(
                condition=IfCondition(
                    LaunchConfiguration("use_foxglove_studio")
                ),
                cmd=["foxglove-studio", "--load-layout", foxglove_layout],
                name="foxglove_studio",
                output="screen",
            ),
        ]
    )
