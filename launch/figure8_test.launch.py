#! /usr/bin/env python3
"""
Launch file for the Figure 8 test simulation.

This launch file sets up a simulation environment with a Figure 8 trajectory generator,
a kinematic arbiter node for state estimation, and visualization tools.
It configures parameters for the simulation and provides event handlers for graceful shutdown.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate the launch description for the Figure 8 test simulation."""
    pkg_share = get_package_share_directory("kinematic_arbiter")

    # Foxglove layout file
    foxglove_layout = os.path.join(pkg_share, "config", "figure8_layout.json")

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

    # Launch arguments
    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="50.0",
        description="Rate at which to publish the trajectory (Hz)",
    )

    # Add parent frequency argument
    parent_frequency_arg = DeclareLaunchArgument(
        "parent_frequency",
        default_value="20.0",
        description="Parent frequency for the simulator (Hz, max 100Hz)",
    )

    filter_rate_arg = DeclareLaunchArgument(
        "filter_rate",
        default_value="50.0",
        description="Rate at which to run the Kalman filter (Hz)",
    )

    # Trajectory arguments
    max_velocity_arg = DeclareLaunchArgument(
        "max_velocity",
        default_value="1.0",
        description="Maximum velocity for Figure 8 trajectory (m/s)",
    )

    length_arg = DeclareLaunchArgument(
        "length",
        default_value="5.0",
        description="Length of Figure 8 trajectory (m)",
    )

    width_arg = DeclareLaunchArgument(
        "width",
        default_value="3.0",
        description="Width of Figure 8 trajectory (m)",
    )

    width_slope_arg = DeclareLaunchArgument(
        "width_slope",
        default_value="0.1",
        description="Width slope for Figure 8 trajectory",
    )

    angular_scale_arg = DeclareLaunchArgument(
        "angular_scale",
        default_value="0.001",
        description="Angular scale for Figure 8 trajectory",
    )

    # RViz config - update path to use our visualization file
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="src/kinematic_arbiter/rviz/figure8_visualization.rviz",
        description="Path to RViz configuration file",
    )

    # Namespace
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="test",
        description="Namespace for the nodes",
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
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {
                "publish_rate": LaunchConfiguration("publish_rate"),
                "parent_frequency": LaunchConfiguration("parent_frequency"),
                "frame_id": "map",
                "base_frame_id": "base_link",
                # Trajectory parameters
                "trajectory.max_vel": LaunchConfiguration("max_velocity"),
                "trajectory.length": LaunchConfiguration("length"),
                "trajectory.width": LaunchConfiguration("width"),
                "trajectory.width_slope": LaunchConfiguration("width_slope"),
                "trajectory.angular_scale": LaunchConfiguration(
                    "angular_scale"
                ),
                # Sensors configuration
                "sensors": ["position1", "position2"],
                # Position sensor 1 parameters (center of robot)
                "sensors.position1.position": [0.0, 0.0, 0.0],
                "sensors.position1.quaternion": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # w, x, y, z
                "sensors.position1.noise_sigma": 0.05,
                "sensors.position1.publish_rate": 20.0,
                # Position sensor 2 parameters (front of robot, noisier)
                "sensors.position2.position": [1.0, 0.0, 0.0],
                "sensors.position2.quaternion": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # w, x, y, z
                "sensors.position2.noise_sigma": 0.1,
                "sensors.position2.publish_rate": 10.0,
            }
        ],
    )

    # Add RViz node using our configuration
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
    )

    # Kinematic arbiter node
    arbiter_node = Node(
        package="kinematic_arbiter",
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {
                "publish_rate": LaunchConfiguration("filter_rate"),
                "frame_id": "map",
                "base_frame_id": "base_link_estimated",
                # Configure sensors to use
                "sensors": ["position1"],
            }
        ],
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
            # Arguments
            debug_arg,
            use_foxglove_studio_arg,
            publish_rate_arg,
            parent_frequency_arg,
            filter_rate_arg,
            max_velocity_arg,
            length_arg,
            width_arg,
            width_slope_arg,
            angular_scale_arg,
            rviz_config_arg,
            namespace_arg,
            # Logging
            debug_output,
            # Nodes
            simulator_node,
            rviz_node,
            arbiter_node,
            # Foxglove integration
            foxglove_bridge_node,
            foxglove_studio_process,
        ]
    )
