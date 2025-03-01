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

    seed = 543
    max_frequency = 20.0
    max_amplitude = 0.3
    number_of_signals = 10


class SingleDofSignalGenerator:
    """Signal generator for single degree of freedom systems."""

    def __init__(self, params=SignalParams()):
        """Initialize the signal generator."""
        self.params = params
        self.frequencies = []
        self.amplitudes = []
        self.phases = []
        self.seed = None
        self.reset()

    def reset(self):
        """Reset the signal components with a new random seed.

        Returns:
            int: The new random seed used
        """
        # Generate a new random seed
        self.seed = random.randint(0, 10000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Generate random frequencies, amplitudes, and phases
        self.frequencies = []
        self.amplitudes = []
        self.phases = []

        for _ in range(self.params.number_of_signals):
            self.frequencies.append(
                random.uniform(0.1, self.params.max_frequency)
            )
            self.amplitudes.append(
                random.uniform(0.1, self.params.max_amplitude)
            )
            self.phases.append(random.uniform(0, 2 * math.pi))

        return self.seed

    def get_value(self, t):
        """Get the clean signal value at time t.

        Args:
            t: Time in seconds

        Returns:
            float: Signal value
        """
        value = 0.0
        for i in range(len(self.frequencies)):
            value += self.amplitudes[i] * math.sin(
                2 * math.pi * self.frequencies[i] * t + self.phases[i]
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
        clean_signal = self.get_value(t)
        noise = np.random.normal(0, noise_level)
        noisy_signal = clean_signal + noise
        return clean_signal, noisy_signal
