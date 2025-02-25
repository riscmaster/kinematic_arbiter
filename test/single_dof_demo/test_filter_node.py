# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Unit tests for the ROS 2 node that implements mediated Kalman filtering.

Tests focus on mediation mode control functionality.
"""

import unittest

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from kinematic_arbiter.action import FilterMediation


class TestFilterNode(unittest.TestCase):
    """Test cases for the filter node functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        rclpy.init()
        cls.node = Node("test_filter_node")
        cls.action_client = ActionClient(
            cls.node, FilterMediation, "set_mediation_mode"
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after tests have run."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_mediation_mode_change(self):
        """Test changing the mediation mode via action client."""
        goal_msg = FilterMediation.Goal()
        goal_msg.requested_mode = FilterMediation.Goal.ADJUST_STATE

        future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future)

        goal_handle = future.result()
        self.assertIsNotNone(goal_handle)
        self.assertTrue(goal_handle.accepted)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        result = result_future.result()
        self.assertTrue(result.result.success)
        self.assertEqual(
            result.result.current_mode, FilterMediation.Goal.ADJUST_STATE
        )


if __name__ == "__main__":
    unittest.main()
