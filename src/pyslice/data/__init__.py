"""
Data structures for PySlice signal handling.

Provides Signal class and related data structures for handling
multidimensional data with calibrated dimensions and metadata.
"""

from .signal import (
    Signal,
    Dimensions,
    Dimension,
    GeneralMetadata,
    SEASerializable,
    SignalSet,
    AcquisitionSet,
    generate_uuid,
    load,
)

__all__ = [
    "Signal",
    "Dimensions",
    "Dimension",
    "GeneralMetadata",
    "SEASerializable",
    "SignalSet",
    "AcquisitionSet",
    "generate_uuid",
    "load",
]
