
.. _program_listing_file_src_single_dof_demo_core_signal_generator.py:

Program Listing for File signal_generator.py
============================================

|exhale_lsh| :ref:`Return to documentation for file <file_src_single_dof_demo_core_signal_generator.py>` (``src/single_dof_demo/core/signal_generator.py``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: py

   # Copyright (c) 2024 Spencer Maughan
   #
   # Permission is hereby granted, free of charge, to any person obtaining a copy
   # of this software and associated documentation files (the "Software"), to deal
   # in the Software without restriction.

   """
   Signal generation utilities for testing Kalman filters.

   Provides functions to generate clean and noisy layered sinusoidal test signals
   with configurable parameters like frequency, amplitude and noise level.
   """

   import random
   import math
   import numpy as np


   class SignalParams:
       """Parameters for configuring signal generation.

       This class defines parameters for generating test signals composed of
       multiple sinusoidal components with random frequencies and amplitudes.
       It does not define a sampling frequency - that should be determined by
       the application using this generator.

       Attributes:
           seed: Random seed for reproducible signal generation
           max_frequency: Maximum frequency for signal components (Hz)
           max_amplitude: Maximum amplitude for signal components
           number_of_signals: Number of sinusoidal components to combine
       """

       def __init__(
           self,
           seed=543,
           max_frequency=20.0,
           max_amplitude=0.3,
           number_of_signals=10,
       ):
           """Initialize signal parameters.

           Args:
               seed: Random seed for reproducible signal generation
               max_frequency: Maximum frequency for signal components (Hz)
               max_amplitude: Maximum amplitude for signal components
               number_of_signals: Number of sinusoidal components to combine
           """
           self.seedseed = seed
           self.max_frequencymax_frequency = max_frequency
           self.max_amplitudemax_amplitude = max_amplitude
           self.number_of_signalsnumber_of_signals = number_of_signals


   class SingleDofSignalGenerator:
       """Signal generator for single degree of freedom systems."""

       def __init__(self, params=SignalParams()):
           """Initialize the signal generator."""
           self.paramsparams = params
           self.frequenciesfrequencies = []
           self.amplitudesamplitudes = []
           self.phasesphases = []
           self.seedseed = None
           self.resetreset()

       def reset(self):
           """Reset the signal components with a new random seed.

           Returns:
               int: The new random seed used
           """
           # Generate a new random seed
           self.seedseed = random.randint(0, 10000)
           random.seed(self.seedseed)
           np.random.seed(self.seedseed)

           # Generate random frequencies, amplitudes, and phases
           self.frequenciesfrequencies = []
           self.amplitudesamplitudes = []
           self.phasesphases = []

           for _ in range(self.paramsparams.number_of_signals):
               self.frequenciesfrequencies.append(
                   random.uniform(0.1, self.paramsparams.max_frequency)
               )
               self.amplitudesamplitudes.append(
                   random.uniform(0.1, self.paramsparams.max_amplitude)
               )
               self.phasesphases.append(random.uniform(0, 2 * math.pi))

           return self.seedseed

       def get_value(self, t):
           """Get the clean signal value at time t.

           Args:
               t: Time in seconds

           Returns:
               float: Signal value
           """
           value = 0.0
           for i in range(len(self.frequenciesfrequencies)):
               value += self.amplitudesamplitudes[i] * math.sin(
                   2 * math.pi * self.frequenciesfrequencies[i] * t + self.phasesphases[i]
               )
           return value

       def generate_signal(self, t, noise_level=0.2):
           """Generate both clean and noisy signals at time t.

           Args:
               t: Time in seconds
               noise_level: Standard deviation of Gaussian noise

           Returns:
               tuple: (clean_signal, noisy_signal)
           """
           clean_signal = self.get_valueget_value(t)
           noise = np.random.normal(0, noise_level)
           noisy_signal = clean_signal + noise
           return clean_signal, noisy_signal

       def generate_complete_signal(self, duration=1.0, sample_frequency=2000):
           """Generate both clean and noisy signals over a specified duration.

           Args:
               duration: Total duration of the signal (seconds)
               sample_frequency: Number of samples per second

           Returns:
               tuple: (clean_signal, noisy_signal, signal_time)
           """
           np.random.seed(self.paramsparams.seed)
           number_of_samples = int(duration * sample_frequency)
           signal_time = np.linspace(
               start=0.0, stop=duration, num=number_of_samples
           )
           frequencies = np.random.uniform(
               0, self.paramsparams.max_frequency, self.paramsparams.number_of_signals
           )
           amplitudes = np.random.uniform(
               0, self.paramsparams.max_amplitude, self.paramsparams.number_of_signals
           )
           signal = np.zeros(len(signal_time))
           for i, t in enumerate(signal_time):
               for frequency, amplitude in zip(frequencies, amplitudes):
                   signal[i] += amplitude * np.sin(2 * np.pi * frequency * t)

           noisy_signal = signal + np.random.normal(
               0.0, self.paramsparams.max_amplitude, len(signal)
           )
           return signal, noisy_signal, signal_time
