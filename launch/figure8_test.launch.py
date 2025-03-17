#! /usr/bin/env python3
"""
Launch file for the Figure 8 test simulation.

This launch file sets up a simulation environment with a Figure 8 trajectory generator,
a kinematic arbiter node for state estimation, and visualization tools.
It configures parameters for the simulation and provides event handlers for graceful shutdown.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    RegisterEventHandler,
    GroupAction,
    EmitEvent,
    ExecuteProcess,
    TimerAction,
)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    """Generate launch description with specified arguments and nodes."""
    # Launch arguments
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug output"
    )

    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate",
        default_value="50.0",
        description="Rate at which to publish simulator data",
    )

    filter_rate_arg = DeclareLaunchArgument(
        "filter_rate",
        default_value="20.0",
        description="Rate at which to publish filter estimates",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="$(find kinematic_arbiter)/rviz/figure8_visualization.rviz",
        description="Path to RViz config file",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace", default_value="test", description="Namespace for nodes"
    )

    # Debug flag for cleanup
    cleanup_debug = DeclareLaunchArgument(
        "cleanup_debug",
        default_value="false",
        description="Show debug output from cleanup",
    )

    # Cleanup shell commands for simulator node with debugging output
    cleanup_simulator_cmd_debug = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            'echo "Checking for existing figure8_simulator nodes..."; '
            "for pid in $(ps aux | grep figure8_simulator_node | grep -v grep | awk '{print $2}'); do "
            '  echo "Killing existing figure8_simulator_node with PID: $pid"; '
            "  kill -9 $pid; "
            "done",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("cleanup_debug")),
    )

    # Cleanup shell commands for arbiter node with debugging output
    cleanup_arbiter_cmd_debug = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            'echo "Checking for existing kinematic_arbiter nodes..."; '
            "for pid in $(ps aux | grep kinematic_arbiter_node | grep -v grep | awk '{print $2}'); do "
            '  echo "Killing existing kinematic_arbiter_node with PID: $pid"; '
            "  kill -9 $pid; "
            "done",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("cleanup_debug")),
    )

    # Silent cleanup commands using output redirection to /dev/null
    cleanup_simulator = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            "for pid in $(ps aux | grep figure8_simulator_node | grep -v grep | awk '{print $2}'); do "
            "  kill -9 $pid >/dev/null 2>&1; "
            "done >/dev/null 2>&1",
        ],
        output="log",  # Use log instead of silent
    )

    cleanup_arbiter = ExecuteProcess(
        cmd=[
            "bash",
            "-c",
            "for pid in $(ps aux | grep kinematic_arbiter_node | grep -v grep | awk '{print $2}'); do "
            "  kill -9 $pid >/dev/null 2>&1; "
            "done >/dev/null 2>&1",
        ],
        output="log",  # Use log instead of silent
    )

    # Add a small delay after cleanup to ensure processes have terminated
    delay_after_cleanup = TimerAction(
        period=1.0, actions=[LogInfo(msg="Starting nodes after cleanup...")]
    )

    # Debug outputs
    debug_output = LogInfo(
        condition=IfCondition(LaunchConfiguration("debug")),
        msg=["Launching with namespace: ", LaunchConfiguration("namespace")],
    )

    # Figure 8 simulator node
    simulator_node = Node(
        package="kinematic_arbiter",
        executable="figure8_simulator_node",
        name="figure8_simulator",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "publish_rate": LaunchConfiguration("publish_rate"),
                "figure8_scale_x": 5.0,
                "figure8_scale_y": 2.5,
                "period": 20.0,
                "sensor_noise_stddev": 0.05,
                "frame_id": "odom",
                "base_frame_id": "base_link",
                "sensor1_position": [1.0, 0.5, 0.2],  # front-right
                "sensor2_position": [1.0, -0.5, 0.2],  # front-left
            }
        ],
    )

    # Kinematic arbiter node
    arbiter_node = Node(
        package="kinematic_arbiter",
        executable="kinematic_arbiter_node",
        name="kinematic_arbiter",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "publish_rate": LaunchConfiguration("filter_rate"),
                "max_delay_window": 0.5,
                "frame_id": "odom",
                "position_sensors": ["position1", "position2"],
                "sensors.position1.topic": "/sensors/position1",
                "sensors.position1.frame_id": "sensor1_frame",
                "sensors.position2.topic": "/sensors/position2",
                "sensors.position2.frame_id": "sensor2_frame",
            }
        ],
    )

    # Add RViz visualization
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    # Group all nodes together to ensure they're managed as a unit
    nodes_group = GroupAction([simulator_node, arbiter_node, rviz_node])

    # Event handler for when RViz exits
    rviz_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=rviz_node,
            on_exit=[
                LogInfo(msg="RViz exited - shutting down all nodes"),
                EmitEvent(event=Shutdown(reason="RViz exited")),
            ],
        )
    )

    # Event handler for when simulator exits
    simulator_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=simulator_node,
            on_exit=[
                LogInfo(msg="Simulator exited - shutting down all nodes"),
                EmitEvent(event=Shutdown(reason="Simulator exited")),
            ],
        )
    )

    # Event handler for when arbiter exits
    arbiter_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=arbiter_node,
            on_exit=[
                LogInfo(msg="Arbiter exited - shutting down all nodes"),
                EmitEvent(event=Shutdown(reason="Arbiter exited")),
            ],
        )
    )

    # Event handler for shutdown
    shutdown_handler = RegisterEventHandler(
        OnShutdown(on_shutdown=[LogInfo(msg=["Shutting down launch file"])])
    )

    return LaunchDescription(
        [
            debug_arg,
            publish_rate_arg,
            filter_rate_arg,
            rviz_config_arg,
            namespace_arg,
            cleanup_debug,
            # First run cleanup commands
            cleanup_simulator_cmd_debug,
            cleanup_arbiter_cmd_debug,
            cleanup_simulator,
            cleanup_arbiter,
            # Small delay to ensure cleanup completes
            delay_after_cleanup,
            # Then proceed with normal launch
            debug_output,
            nodes_group,
            rviz_exit_handler,
            simulator_exit_handler,
            arbiter_exit_handler,
            shutdown_handler,
        ]
    )
