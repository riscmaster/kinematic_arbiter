# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Integration tests for the Kinematic Arbiter package.

Tests validate signal filtering and state estimation functionality.
"""

import asyncio
import unittest

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


class TestIntegration(unittest.TestCase):
    """Integration tests for the Kinematic Arbiter package."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        rclpy.init()
        cls.node = Node("test_integration")
        cls.measurements = []
        cls.filtered_states = []

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after tests have run."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def measurement_callback(self, msg):
        """Store measurement values from incoming messages."""
        self.measurements.append(msg.pose.pose.position.x)

    def filtered_callback(self, msg):
        """Store filtered state values from incoming messages."""
        self.filtered_states.append(msg.pose.pose.position.x)

    async def test_signal_filtering(self):
        """Test that filtering reduces signal noise."""
        # Subscribe to topics
        self.node.create_subscription(
            PoseWithCovarianceStamped,
            "raw_measurements",
            self.measurement_callback,
            10,
        )
        self.node.create_subscription(
            PoseWithCovarianceStamped,
            "filtered_state",
            self.filtered_callback,
            10,
        )

        # Wait for messages
        await asyncio.sleep(2.0)

        # Verify data
        self.assertGreater(len(self.measurements), 0)
        self.assertGreater(len(self.filtered_states), 0)

        # Verify filtering effect
        meas_std = np.std(self.measurements)
        filt_std = np.std(self.filtered_states)
        self.assertLess(filt_std, meas_std)


if __name__ == "__main__":
    unittest.main()
