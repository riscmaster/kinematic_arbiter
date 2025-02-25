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
    """Parameters for configuring signal generation."""

    seed = 543
    duration = 10.0
    sample_frequency = 200.0
    max_frequency = 20.0
    max_amplitude = 0.3
    number_of_signals = 10


def generate_signals(params=SignalParams()):
    """Generate clean and noisy test signals.

    Args:
        params: SignalParams object containing generation parameters

    Returns:
        Tuple containing:
        - Clean signal array
        - Noisy signal array
        - Time points array
    """
    np.random.seed(params.seed)
    number_of_samples = int(params.duration * params.sample_frequency)
    signal_time = np.linspace(
        start=0.0, stop=params.duration, num=number_of_samples
    )
    frequencies = np.random.uniform(
        0, params.max_frequency, params.number_of_signals
    )
    amplitudes = np.random.uniform(
        0, params.max_amplitude, params.number_of_signals
    )
    signal = np.zeros(len(signal_time))
    for i, t in enumerate(signal_time):
        for frequency, amplitude in zip(frequencies, amplitudes):
            signal[i] += amplitude * np.sin(2 * np.pi * frequency * t)

    noisy_signal = signal + np.random.normal(
        0.0, params.max_amplitude, len(signal)
    )
    return signal, noisy_signal, signal_time
