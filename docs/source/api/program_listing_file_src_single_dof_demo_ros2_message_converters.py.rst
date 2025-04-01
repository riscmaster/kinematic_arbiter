
.. _program_listing_file_src_single_dof_demo_ros2_message_converters.py:

Program Listing for File message_converters.py
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_ros2_message_converters.py>` (``src/single_dof_demo/ros2/message_converters.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """Kinematic Arbiter package component."""

   from geometry_msgs.msg import PoseWithCovarianceStamped
   from rclpy.time import Time

   from ..ros2.domain_models import State


   def pose_with_covariance_to_state(msg: PoseWithCovarianceStamped) -> State:
       """Convert PoseWithCovarianceStamped message to State."""
       return State(
           value=msg.pose.pose.position.x, variance=msg.pose.covariance[0]
       )


   def state_to_pose_with_covariance_stamped(
       state: State, stamp: Time
   ) -> PoseWithCovarianceStamped:
       """Convert State to PoseWithCovarianceStamped message."""
       msg = PoseWithCovarianceStamped()
       msg.header.stamp = stamp
       msg.pose.pose.position.x = state.value
       msg.pose.covariance = [0.0] * 36
       msg.pose.covariance[0] = state.variance
       return msg


   def ros_time_to_float(stamp: Time) -> float:
       """Convert Time to float representation."""
       return stamp.sec + stamp.nanosec * 1e-9
