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
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for the kinematic arbiter demo."""
    pkg_share = get_package_share_directory("kinematic_arbiter")

    # Launch Arguments
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true", description="Start RViz2"
    )

    # Config files
    rviz_config = os.path.join(pkg_share, "rviz", "filter_demo.rviz")

    return LaunchDescription(
        [
            use_rviz_arg,
            # Signal Generator Node
            ExecuteProcess(
                cmd=[
                    "python3",
                    "install/kinematic_arbiter/lib/kinematic_arbiter/"
                    "signal_generator_node.py",
                ],
                name="signal_generator",
                output="screen",
            ),
            # Filter Node
            ExecuteProcess(
                cmd=[
                    "python3",
                    "install/kinematic_arbiter/lib/kinematic_arbiter/"
                    "filter_node.py",
                ],
                name="filter",
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
            # RViz
            Node(
                condition=LaunchConfiguration("use_rviz"),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            ),
            # TF2 Static Transform Publisher
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_transform_publisher",
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "world",
                    "filter_frame",
                ],
            ),
        ]
    )
