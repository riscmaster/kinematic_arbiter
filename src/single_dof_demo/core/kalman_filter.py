# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""
Base Kalman filter implementation with frequency model.

Provides fundamental Kalman filter functionality for state estimation
with a configurable frequency model.
"""

import math
from . import filter_output
from .filter_utilities import variance_to_bound


class KalmanFilter(object):
    """Kalman filter implementation with frequency model."""

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        frequency: float = 0.0,
        amplitude: float = 0.0,
    ):
        """Initialize the KalmanFilter.

        Initialize with the given process/measurement noise and frequency.

        Args:
            process_noise (float): The variance of the process noise.
            measurement_noise (float): The variance of the measurement noise.
            frequency (float): The frequency of the model in Hz. Default is 0.0.
            amplitude (float): The amplitude of the model. Default is 0.0.
        """
        self.state_estimate = 0.0
        self.state_variance = 10.0
        self.measurement_variance = measurement_noise
        self.process_variance = process_noise
        self.frequency = frequency
        self.amplitude = amplitude
        self.last_time = None
        self.is_initialized = False

    def set_process_noise(self, process_noise: float):
        """
        Update the process noise parameter.

        Args:
            process_noise (float): The new process noise variance.
        """
        self.process_variance = process_noise

    def set_measurement_noise(self, measurement_noise: float):
        """
        Update the measurement noise parameter.

        Args:
            measurement_noise (float): The new measurement noise variance.
        """
        self.measurement_variance = measurement_noise

    def set_frequency(self, frequency: float):
        """
        Update the frequency model parameter.

        Args:
            frequency (float): The new frequency in Hz.
        """
        self.frequency = frequency

    def set_amplitude(self, amplitude: float):
        """
        Update the amplitude model parameter.

        Args:
            amplitude (float): The new amplitude.
        """
        self.amplitude = amplitude

    def update(
        self, measurement: float, time_secs: float = None
    ) -> filter_output.FilterOutput:
        """Update the Kalman filter with a new measurement.

        Return the filter output containing predicted, mediated, and final estimates.

        Args:
            measurement (float): The new measurement to incorporate.
            time_secs (float, optional): Current time in seconds for time-based
                models.

        Returns:
            filter_output.FilterOutput: The output of the filter containing
            the predicted, mediated, and final state and measurement estimates.
        """
        output = filter_output.FilterOutput()

        # Handle time for frequency model
        dt = (
            0.0
            if time_secs is None or self.last_time is None
            else time_secs - self.last_time
        )
        frequency = 0.0 if time_secs is None else self.frequency
        dx = self.amplitude * math.sin(2 * math.pi * frequency * dt)
        predicted_state = self.state_estimate + dx

        output.predicted.state.value = predicted_state
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
            if time_secs is not None:
                self.last_time = time_secs
            return output

        # Innovation
        innovation = measurement - predicted_state
        innovation_variance = predicted_variance + self.measurement_variance

        # Update
        output.final = output.mediated
        kalman_gain = predicted_variance / innovation_variance
        self.state_estimate = predicted_state + kalman_gain * innovation
        self.state_variance = (1.0 - kalman_gain) * predicted_variance
        output.final.state.value = self.state_estimate
        output.final.state.bound = variance_to_bound(self.state_variance)

        # Update last time
        if time_secs is not None:
            self.last_time = time_secs

        return output
