# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""
Implementation of the Mediated Kalman Filter.

Extends traditional Kalman filter with measurement validation and dynamic noise
estimation. Key features:
- Chi-square based measurement validation
- Dynamic measurement noise estimation
- Automated process noise estimation
- Single parameter (ζ) tuning
"""

from . import filter_output
from .filter_utilities import bound_to_variance, variance_to_bound


class Mediation(object):
    """Enumeration of mediation behaviors for the filter."""

    ADJUST_STATE = 0
    ADJUST_MEASUREMENT = 1
    REJECT_MEASUREMENT = 2
    NO_ACTION = 3


class MediatedKalmanFilter(object):
    """Implementation of a Kalman filter with measurement mediation."""

    def __init__(
        self,
        process_to_measurement_ratio: float,
        window_time: float,
        mediation=Mediation.ADJUST_STATE,
    ):
        """Initialize the filter with configuration parameters.

        Args:
            process_to_measurement_ratio: Ratio between process and measurement
            noise
            window_time: Time window for noise estimation
            mediation: Mediation behavior to use when measurement is rejected
        """
        self.chi_squared_value = 6.64  # %99 confidence
        self.scale = process_to_measurement_ratio
        self.window_time = window_time
        self.state_estimate = 0.0
        self.state_variance = 1.0
        self.measurement_variance = 1.0
        self.process_variance = 0.0
        self.previous_t = None
        self.mediation = False
        self.mediation_behavior = mediation

    def update(
        self, measurement: float, t: float
    ) -> filter_output.FilterOutput:
        """Update filter state with new measurement.
        Args:
            measurement: New measurement value
            t: Timestamp of measurement

        Returns:
            FilterOutput containing predicted, mediated and final states
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
        if self.previous_t is None:
            self.previous_t = t
            self.state_estimate = measurement
            output.final.state.value = self.state_estimate
            output.final.state.bound = variance_to_bound(self.state_variance)
            output.final.measurement.value = measurement
            output.final.measurement.bound = variance_to_bound(
                self.measurement_variance
            )
            return output

        delta_t = t - self.previous_t
        self.previous_t = t
        sample_window = self.window_time / delta_t

        # Innovation
        innovation = measurement - self.state_estimate
        innovation_variance = predicted_variance + self.measurement_variance
        innovation_2 = innovation**2

        # Mediation test
        self.mediation = (
            innovation_2 / innovation_variance > self.chi_squared_value
        )
        if self.mediation and self.mediation_behavior != Mediation.NO_ACTION:
            if self.mediation_behavior == Mediation.REJECT_MEASUREMENT:
                output.final = output.mediated
                return output
            elif self.mediation_behavior == Mediation.ADJUST_STATE:
                innovation_variance = innovation_2 / self.chi_squared_value
                predicted_variance = (
                    innovation_variance - self.measurement_variance
                )
                output.mediated.state.bound = variance_to_bound(
                    predicted_variance
                )
            elif self.mediation_behavior == Mediation.ADJUST_MEASUREMENT:
                innovation_variance = innovation_2 / self.chi_squared_value
                self.measurement_variance = (
                    innovation_variance - predicted_variance
                )
                output.mediated.measurement.bound = bound_to_variance(
                    self.measurement_variance
                )

        # Update
        output.final = output.mediated
        self.measurement_variance += (
            innovation_2 - self.measurement_variance
        ) / sample_window
        self.process_variance += (
            self.scale * self.measurement_variance - self.process_variance
        ) / sample_window
        kalman_gain = predicted_variance / innovation_variance
        self.state_estimate += kalman_gain * innovation
        self.state_variance = (1.0 - kalman_gain) * predicted_variance
        output.final.state.value = self.state_estimate
        output.final.state.bound = variance_to_bound(self.state_variance)
        return output
