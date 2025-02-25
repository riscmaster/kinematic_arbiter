# Copyright (c) 2024 Spencer Maughan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Kinematic Arbiter package component."""

from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class State:
    """Represents the state with a value and variance."""

    value: float
    variance: float


@dataclass
class FilterParameters:
    """Parameters for configuring the filter."""

    process_measurement_ratio: float
    window_time: float


class MediationMode(Enum):
    """Enumeration of mediation modes for the filter."""

    ADJUST_STATE = auto()
    ADJUST_MEASUREMENT = auto()
    REJECT_MEASUREMENT = auto()
    NO_ACTION = auto()
