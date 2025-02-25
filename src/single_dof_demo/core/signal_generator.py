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
        np.random.seed(params.seed)
        self.frequencies = np.random.uniform(
            0, params.max_frequency, params.number_of_signals
        )
        self.amplitudes = np.random.uniform(
            0, params.max_amplitude, params.number_of_signals
        )

    def generate_signal(self, t_secs: float) -> float:
        """Generate clean and noisy test signals."""
        clean_signal = sum(
            amplitude * np.sin(2 * np.pi * frequency * t_secs)
            for frequency, amplitude in zip(self.frequencies, self.amplitudes)
        )
        noisy_signal = clean_signal + np.random.normal(
            0.0, self.params.max_amplitude
        )
        return clean_signal, noisy_signal
