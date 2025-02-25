# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""
Base Kalman filter implementation.

Provides fundamental Kalman filter functionality for state estimation.
"""

from . import filter_output
from .filter_utilities import variance_to_bound


class KalmanFilter(object):
    """Kalman filter implementation."""

    def __init__(self, process_noise: float, measurement_noise: float):
        """
        Initialize the KalmanFilter with the given process and measurement
        noise.

        Args:
            process_noise (float): The variance of the process noise.
            measurement_noise (float): The variance of the measurement noise.
        """
        self.state_estimate = 0.0
        self.state_variance = 1.0
        self.measurement_variance = measurement_noise
        self.process_variance = process_noise
        self.is_initialized = False

    def update(self, measurement: float) -> filter_output.FilterOutput:
        """
        Update the Kalman filter with a new measurement and return the
        filter output containing predicted, mediated, and final estimates.

        Args:
            measurement (float): The new measurement to incorporate into
            the filter.

        Returns:
            filter_output.FilterOutput: The output of the filter containing
            the predicted, mediated, and final state and measurement estimates.
        """
        output = filter_output.FilterOutput()
        # Prediction Step
        output.predicted.state.value = self.state_estimate
        predicted_variance = self.state_variance + self.process_variance
        output.predicted.state.bound = variance_to_bound(predicted_variance)
        output.predicted.measurement.value = measurement
        output.predicted.measurement.bound = variance_to_bound(
            self.measurement_variance
        )
        output.mediated = output.predicted

        # Initialization
        if not self.is_initialized:
            self.is_initialized = True
            self.state_estimate = measurement
            output.final.state.value = self.state_estimate
            output.final.state.bound = variance_to_bound(self.state_variance)
            output.final.measurement.value = measurement
            output.final.measurement.bound = variance_to_bound(
                self.measurement_variance
            )
            return output

        # Innovation
        innovation = measurement - self.state_estimate
        innovation_variance = predicted_variance + self.measurement_variance

        # Update
        output.final = output.mediated
        kalman_gain = predicted_variance / innovation_variance
        self.state_estimate += kalman_gain * innovation
        self.state_variance = (1.0 - kalman_gain) * predicted_variance
        output.final.state.value = self.state_estimate
        output.final.state.bound = variance_to_bound(self.state_variance)
        return output
