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
        self.value = value
        self.bound = bound


class StateMeasurementPair:
    """Represents a pair of state and measurement estimates."""

    def __init__(self):
        """Initialize a state-measurement pair with default estimates."""
        self.state = Estimate()
        self.measurement = Estimate()


class FilterOutput:
    """Represents the output of the filter containing predicted, mediated, and
    final estimates.
    """

    def __init__(self):
        """Initialize filter output with default state-measurement pairs."""
        self.predicted = StateMeasurementPair()
        self.mediated = StateMeasurementPair()
        self.final = StateMeasurementPair()
