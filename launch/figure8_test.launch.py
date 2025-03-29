#! /usr/bin/env python3
"""
Launch file for the Figure 8 test simulation.

This launch file sets up a simulation environment with a Figure 8 trajectory generator,
a kinematic arbiter node for state estimation, and visualization tools.
It configures parameters for the simulation and provides event handlers for graceful shutdown.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
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

    # Frame IDs
    world_frame_arg = DeclareLaunchArgument(
        "world_frame_id",
        default_value="map",
        description="World frame ID",
    )

    body_frame_arg = DeclareLaunchArgument(
        "body_frame_id",
        default_value="base_link",
        description="Body frame ID",
    )

    estimated_body_frame_arg = DeclareLaunchArgument(
        "estimated_body_frame_id",
        default_value="base_link_estimated",
        description="Estimated body frame ID",
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
                # Main parameters
                "main_update_rate": 100.0,  # Core update rate
                "world_frame_id": LaunchConfiguration("world_frame_id"),
                "true_base_link": LaunchConfiguration("body_frame_id"),
                "noise_sigma": 0.01,  # Base noise level
                "time_jitter": 0.005,  # Small timing jitter
                # Trajectory parameters
                "trajectory.max_vel": LaunchConfiguration("max_velocity"),
                "trajectory.length": LaunchConfiguration("length"),
                "trajectory.width": LaunchConfiguration("width"),
                # Sensor rates
                "position_rate": 30.0,  # Position sensor update rate (Hz)
                "pose_rate": 50.0,  # Pose sensor update rate (Hz)
                "velocity_rate": 50.0,  # Velocity sensor update rate (Hz)
                "imu_rate": 100.0,  # IMU sensor update rate (Hz)
                # Position sensor parameters (offsets from body frame)
                "position_sensor.x_offset": 0.05,  # Slight offset for realism
                "position_sensor.y_offset": 0.0,
                "position_sensor.z_offset": 0.1,
                # Pose sensor parameters
                "pose_sensor.x_offset": -0.03,
                "pose_sensor.y_offset": 0.02,
                "pose_sensor.z_offset": 0.15,
                # Velocity sensor parameters
                "velocity_sensor.x_offset": 0.0,
                "velocity_sensor.y_offset": 0.0,
                "velocity_sensor.z_offset": 0.05,
                # IMU sensor parameters
                "imu_sensor.x_offset": 0.01,
                "imu_sensor.y_offset": -0.01,
                "imu_sensor.z_offset": 0.08,
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
                "max_delay_window": 0.5,
                "world_frame_id": LaunchConfiguration("world_frame_id"),
                "body_frame_id": "base_link",
                # TF configuration
                "tf_lookup_timeout": 0.1,
                "tf_fallback_to_identity": True,
                "tf_warning_throttle_period": 5.0,
                # Configure sensors to use for state estimation
                "position_sensors": ["position"],
                "pose_sensors": ["pose"],
                "velocity_sensors": ["velocity"],
                "imu_sensors": ["imu"],
                # Sensor configuration with frame details
                "sensors.position.topic": "test/sensors/position",
                "sensors.position.frame_id": "position",
                "sensors.position.reference_frame": "map",
                "sensors.pose.topic": "test/sensors/pose",
                "sensors.pose.frame_id": "pose",
                "sensors.pose.reference_frame": "map",
                "sensors.velocity.topic": "test/sensors/velocity",
                "sensors.velocity.frame_id": "velocity",
                "sensors.imu.topic": "test/sensors/imu",
                "sensors.imu.frame_id": "imu",
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
            # Launch arguments
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
            world_frame_arg,
            body_frame_arg,
            estimated_body_frame_arg,
            # Debug output
            debug_output,
            # Nodes
            simulator_node,
            rviz_node,
            arbiter_node,
            foxglove_bridge_node,
            foxglove_studio_process,
        ]
    )
