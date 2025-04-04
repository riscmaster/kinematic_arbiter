
.. _program_listing_file_src_single_dof_demo_core_filter_output.py:

Program Listing for File filter_output.py
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_core_filter_output.py>` (``src/single_dof_demo/core/filter_output.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """
   Data structures for storing Kalman filter outputs at each stage of processing.

   Contains classes for:
   - Individual state/measurement estimates with bounds
   - Paired state and measurement estimates
   - Complete filter output with predicted, mediated and final estimates
   """


   class Estimate:
       """Represents an estimate with a value and a bound."""

       def __init__(self, value=0.0, bound=0.0):
           """Initialize an estimate.

           Args:
               value (float): The estimated value. Defaults to 0.0.
               bound (float): The bound on the estimate. Defaults to 0.0.
           """
           self.valuevalue = value
           self.boundbound = bound


   class StateMeasurementPair:
       """Represents a pair of state and measurement estimates."""

       def __init__(self):
           """Initialize a state-measurement pair with default estimates."""
           self.statestate = Estimate()
           self.measurementmeasurement = Estimate()


   class FilterOutput:
       """Contains filter output stages: predicted, mediated, and final estimates."""

       def __init__(self):
           """Initialize filter output with default state-measurement pairs."""
           self.predictedpredicted = StateMeasurementPair()
           self.mediatedmediated = StateMeasurementPair()
           self.finalfinal = StateMeasurementPair()
           self.mediation_detectedmediation_detected = False
           self.mediation_pointmediation_point = 0.0
