
.. _program_listing_file_src_single_dof_demo_core_mediated_kalman_filter.py:

Program Listing for File mediated_kalman_filter.py
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_core_mediated_kalman_filter.py>` (``src/single_dof_demo/core/mediated_kalman_filter.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

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

   import math
   from . import filter_output
   from .filter_utilities import variance_to_bound


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
           sample_window: int,
           mediation=Mediation.NO_ACTION,
           frequency: float = 0.0,
           amplitude: float = 0.0,
       ):
           """Initialize the filter with configuration parameters.

           Args:
               process_to_measurement_ratio: Ratio between process and measurement
               noise (ζ in the paper)
               sample_window: Number of samples for noise estimation (n in the paper)
               mediation: Mediation behavior to use when measurement is rejected
               frequency: Frequency of the model in Hz (default: 0.0)
               amplitude: Amplitude of the model (default: 0.0)
           """
           self.chi_squared_valuechi_squared_value = 6.64  # 99% confidence (χ_c in the paper)
           self.scalescale = process_to_measurement_ratio  # ζ in the paper
           self.sample_windowsample_window = sample_window  # n in the paper
           self.state_estimatestate_estimate = 0.0  # x̂_k in the paper
           self.state_variancestate_variance = 100.0  # P̂_k in the paper
           self.measurement_variancemeasurement_variance = 100.0  # R_k in the paper
           self.process_varianceprocess_variance = 100.0  # Q_k in the paper
           self.previous_tprevious_t = None
           self.mediationmediation = False
           self.mediation_behaviormediation_behavior = mediation
           self.frequencyfrequency = frequency
           self.amplitudeamplitude = amplitude
           self.last_timelast_time = None
           self.is_initializedis_initialized = False
           self.previous_state_estimateprevious_state_estimate = 0.0  # x̂_{k-1} in the paper

       def set_process_measurement_ratio(self, ratio: float):
           """Update the process to measurement ratio parameter.

           Args:
               ratio: The new process to measurement ratio (ζ).
           """
           self.scalescale = ratio
           # Adjust process_variance to maintain the proper ratio with measurement_variance
           self.process_varianceprocess_variance = max(
               self.process_varianceprocess_variance, self.measurement_variancemeasurement_variance
           )
           # Note: In high dimensional cases where measurements don't directly map to every state,
           # a collective maximum bound on state covariance would be tracked and used
           # across various measurements for the observable states.

       def set_sample_window(self, window: int):
           """Update the sample window parameter.

           Args:
               window: The new sample window size (n).
           """
           self.sample_windowsample_window = window

       def set_mediation_behavior(self, behavior: int):
           """Update the mediation behavior.

           Args:
               behavior: The new mediation behavior (from Mediation enum).
           """
           self.mediation_behaviormediation_behavior = behavior

       def set_frequency(self, frequency: float):
           """Update the frequency model parameter.

           Args:
               frequency: The new frequency in Hz.
           """
           self.frequencyfrequency = frequency

       def set_amplitude(self, amplitude: float):
           """Update the amplitude model parameter.

           Args:
               amplitude: The new amplitude.
           """
           self.amplitudeamplitude = amplitude

       def reset(self):
           """Reset the filter state."""
           self.state_estimatestate_estimate = 0.0
           self.state_variancestate_variance = 100.0
           self.measurement_variancemeasurement_variance = 100.0
           self.process_varianceprocess_variance = 10.0
           self.previous_tprevious_t = None
           self.mediationmediation = False
           self.last_timelast_time = None
           self.is_initializedis_initialized = False
           self.previous_state_estimateprevious_state_estimate = 0.0

       def update(
           self, measurement: float, t: float = None
       ) -> filter_output.FilterOutput:
           """Update filter state with new measurement.

           Args:
               measurement: New measurement value (y_k in the paper)
               t: Timestamp of measurement (optional)

           Returns:
               FilterOutput containing predicted, mediated and final states
           """
           output = filter_output.FilterOutput()

           # Handle time for frequency model
           if t is None and self.previous_tprevious_t is not None:
               t = self.previous_tprevious_t + 0.01  # Default time step if none provided

           dt = 0.0 if self.last_timelast_time is None else (t - self.last_timelast_time)
           frequency = 0.0 if t is None else self.frequencyfrequency
           dx = self.amplitudeamplitude * math.sin(2 * math.pi * frequency * dt)

           # Store previous state estimate for process noise calculation
           self.previous_state_estimateprevious_state_estimate = self.state_estimatestate_estimate

           # Prediction Step (x̌_k = A_{k-1} x̂_{k-1} + ν_k)
           predicted_state = self.state_estimatestate_estimate + dx
           output.predicted.state.value = predicted_state

           # Ensure variances are non-negative
           self.process_varianceprocess_variance = max(0.0, self.process_varianceprocess_variance)
           self.state_variancestate_variance = max(0.0, self.state_variancestate_variance)
           self.measurement_variancemeasurement_variance = max(
               1e-8, self.measurement_variancemeasurement_variance
           )  # Minimum to avoid division by zero

           # Prediction Step (P̌_k = A_{k-1} P̂_{k-1} A_{k-1}^T + Q_k)
           predicted_variance = self.state_variancestate_variance + self.process_varianceprocess_variance
           output.predicted.state.bound = variance_to_bound(predicted_variance)
           output.predicted.measurement.value = measurement
           output.predicted.measurement.bound = variance_to_bound(
               self.measurement_variancemeasurement_variance
           )
           output.mediated = output.predicted

           # Initialization
           if not self.is_initializedis_initialized:
               self.is_initializedis_initialized = True
               self.previous_tprevious_t = t
               self.last_timelast_time = t
               self.state_estimatestate_estimate = measurement
               output.final.state.value = self.state_estimatestate_estimate
               output.final.state.bound = variance_to_bound(self.state_variancestate_variance)
               output.final.measurement.value = measurement
               output.final.measurement.bound = variance_to_bound(
                   self.measurement_variancemeasurement_variance
               )
               return output

           self.previous_tprevious_t = t

           # Innovation (y_k - C_k x̌_k)
           innovation = measurement - predicted_state
           # Innovation covariance (C_k P̌_k C_k^T + R_k)
           innovation_variance = predicted_variance + self.measurement_variancemeasurement_variance
           innovation_2 = innovation**2

           # Mediation test (chi-square test)
           # (y_k - C_k x̌_k)^T (C_k P̌_k C_k^T + R_k)^{-1} (y_k - C_k x̌_k) < χ_c
           self.mediationmediation = (
               innovation_2 / innovation_variance > self.chi_squared_valuechi_squared_value
           )

           if self.mediationmediation and self.mediation_behaviormediation_behavior != Mediation.NO_ACTION:
               # Store the mediation point in the output for publishing
               output.mediation_detected = True
               output.mediation_point = measurement

               if self.mediation_behaviormediation_behavior == Mediation.REJECT_MEASUREMENT:
                   output.final = output.mediated
                   # Update last time
                   self.last_timelast_time = t
                   return output
               elif self.mediation_behaviormediation_behavior == Mediation.ADJUST_STATE:
                   # Adjust state variance to satisfy chi-square test
                   innovation_variance = innovation_2 / self.chi_squared_valuechi_squared_value
                   predicted_variance = max(
                       0.0, innovation_variance - self.measurement_variancemeasurement_variance
                   )
                   output.mediated.state.bound = variance_to_bound(
                       predicted_variance
                   )
               elif self.mediation_behaviormediation_behavior == Mediation.ADJUST_MEASUREMENT:
                   # Adjust measurement variance to satisfy chi-square test
                   innovation_variance = innovation_2 / self.chi_squared_valuechi_squared_value
                   self.measurement_variancemeasurement_variance = max(
                       1e-8, innovation_variance - predicted_variance
                   )
                   output.mediated.measurement.bound = variance_to_bound(
                       self.measurement_variancemeasurement_variance
                   )

           # Update Step
           output.final = output.mediated

           # Kalman gain (K_k = P̌_k C_k^T (C_k P̌_k C_k^T + R_k)^{-1})
           kalman_gain = predicted_variance / innovation_variance

           # Store predicted state for process noise calculation (x̌_k)
           predicted_state_before_update = predicted_state

           # Update state (x̂_k = x̌_k + K_k (y_k - C_k x̌_k))
           self.state_estimatestate_estimate = predicted_state + kalman_gain * innovation

           # Update state variance (P̂_k = (I - K_k C_k) P̌_k)
           new_state_variance = (1.0 - kalman_gain) * predicted_variance
           self.state_variancestate_variance = max(0.0, new_state_variance)

           # Use sample window directly from the paper
           # Ensure sample window is at least 1 to avoid division by zero
           n = max(1, self.sample_windowsample_window)

           # Dynamic Measurement Noise Update
           # R̂_k = R̂_{k-1} + ((y_k - C_k x̂_k)(y_k - C_k x̂_k)^T - R̂_{k-1}) / n
           new_measurement_variance = (
               self.measurement_variancemeasurement_variance
               + (innovation_2 - self.measurement_variancemeasurement_variance) / n
           )
           self.measurement_variancemeasurement_variance = max(1e-6, new_measurement_variance)

           # Dynamic Process Noise Update
           # Q̂_k = Q̂_{k-1} + (ζ (x̌_k - x̂_k)(x̌_k - x̂_k)^T - Q̂_{k-1}) / n
           state_update_diff = predicted_state_before_update - self.state_estimatestate_estimate
           new_process_variance = (
               self.process_varianceprocess_variance
               + (self.scalescale * (state_update_diff**2) - self.process_varianceprocess_variance) / n
           )
           self.process_varianceprocess_variance = max(0.0, new_process_variance)

           output.final.state.value = self.state_estimatestate_estimate
           output.final.state.bound = variance_to_bound(self.state_variancestate_variance)

           # Update last time
           self.last_timelast_time = t

           return output
