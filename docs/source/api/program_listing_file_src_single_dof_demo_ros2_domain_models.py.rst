
.. _program_listing_file_src_single_dof_demo_ros2_domain_models.py:

Program Listing for File domain_models.py
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_ros2_domain_models.py>` (``src/single_dof_demo/ros2/domain_models.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """Kinematic Arbiter package component."""

   from dataclasses import dataclass
   from enum import Enum, auto


   @dataclass
   class State:
       """Represents the state with a value and variance."""

       value: float
       variance: float


   @dataclass
   class FilterParameters:
       """Parameters for configuring the filter."""

       process_measurement_ratio: float
       window_time: float


   class MediationMode(Enum):
       """Enumeration of mediation modes for the filter."""

       ADJUST_STATE = auto()
       ADJUST_MEASUREMENT = auto()
       REJECT_MEASUREMENT = auto()
       NO_ACTION = auto()
