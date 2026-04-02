#!/usr/bin/env python3
"""
Input Validation Utilities
===========================

This module provides standardized input validation functions for the IFI package.

Functions:
    validate_sampling_frequency: Validate sampling frequency parameter.
    validate_frequency: Validate frequency parameter.
    validate_signal: Validate signal data.
    validate_signals_match: Validate that two signals have matching lengths.
    validate_positive_number: Validate that a number is positive.
    validate_wavelength: Validate wavelength parameter.
    validate_method: Validate method parameter.
    validate_shot_number: Validate shot number parameter.
    validate_output_directory: Validate output directory parameter.
    validate_analysis_type: Validate analysis type parameter.
    validate_methods_list: Validate methods list parameter.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .log_manager import LogManager, log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.log_manager import LogManager, log_tag

logger = LogManager().get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_sampling_frequency(fs: float, name: str = "sampling_frequency") -> float:
    """Validate sampling frequency parameter."""
    if fs is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(fs, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(fs)}")

    if fs <= 0:
        raise ValidationError(f"{name} must be positive, got {fs}")

    if not np.isfinite(fs):
        raise ValidationError(f"{name} must be finite, got {fs}")

    return float(fs)


def validate_frequency(freq: float, name: str = "frequency") -> float:
    """Validate frequency parameter."""
    if freq is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(freq, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(freq)}")

    if freq <= 0:
        raise ValidationError(f"{name} must be positive, got {freq}")

    if not np.isfinite(freq):
        raise ValidationError(f"{name} must be finite, got {freq}")

    return float(freq)


def validate_signal(
    signal: np.ndarray | pd.Series | list,
    name: str = "signal",
    min_length: int = 1,
) -> np.ndarray:
    """
    Validate signal data.

    Args:
        signal(np.ndarray | pd.Series | list): Signal data
        name(str): Parameter name for error messages
        min_length(int): Minimum required length

    Returns:
        np.ndarray: Validated signal as numpy array
    """
    if signal is None:
        raise ValidationError(f"{name} cannot be None")

    # Convert to numpy array if needed
    if isinstance(signal, (list, pd.Series)):
        signal = np.array(signal)
    elif not isinstance(signal, np.ndarray):
        raise ValidationError(f"{name} must be array-like, got {type(signal)}")

    if len(signal) < min_length:
        raise ValidationError(
            f"{name} must have at least {min_length} samples, got {len(signal)}"
        )

    if signal.size == 0:
        raise ValidationError(f"{name} cannot be empty")

    # Check for NaN or Inf values
    if np.any(np.isnan(signal)):
        logger.warning(f"{log_tag('VALID','SIGNL')} {name} contains NaN values")

    if np.any(np.isinf(signal)):
        logger.warning(f"{log_tag('VALID','SIGNL')} {name} contains infinite values")

    return signal


def validate_signals_match(
    signal1: np.ndarray,
    signal2: np.ndarray,
    name1: str = "signal1",
    name2: str = "signal2",
) -> None:
    """
    Validate that two signals have matching lengths.

    Args:
        signal1(np.ndarray): First signal
        signal2(np.ndarray): Second signal
        name1(str): Name of first signal for error messages
        name2(str): Name of second signal for error messages
    """
    if len(signal1) != len(signal2):
        raise ValidationError(
            f"{name1} and {name2} must have the same length, "
            f"got {len(signal1)} and {len(signal2)}"
        )


def validate_positive_number(value: int | float, name: str = "value") -> float:
    """Validate that a number of a "certain parameter ({name})" is positive."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")

    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")

    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")

    return float(value)


def validate_wavelength(wavelength: float, name: str = "wavelength") -> float:
    """Validate a "certain wavelength ({name})" parameter."""
    if wavelength is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(wavelength, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(wavelength)}")

    if wavelength <= 0:
        raise ValidationError(f"{name} must be positive, got {wavelength}")

    if not np.isfinite(wavelength):
        raise ValidationError(f"{name} must be finite, got {wavelength}")

    return float(wavelength)


def validate_method(method: str, valid_methods: list, name: str = "method") -> str:
    """Validate a "certain method ({name})" parameter."""
    if method is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(method, str):
        raise ValidationError(f"{name} must be a string, got {type(method)}")

    if method not in valid_methods:
        raise ValidationError(f"{name} must be one of {valid_methods}, got '{method}'")

    return method


def validate_shot_number(
    shot_number: str | int, name: str = "shot_number"
) -> str:
    """Validate a "shot number ({name})" parameter."""
    if shot_number is None:
        raise ValidationError(f"{name} cannot be None")

    # Convert to string and validate
    shot_str = str(shot_number).strip()

    if not shot_str:
        raise ValidationError(f"{name} cannot be empty")

    # Check if it's a valid number
    try:
        shot_int = int(shot_str)
        if shot_int <= 0:
            raise ValidationError(f"{name} must be positive, got {shot_int}")
    except ValueError:
        raise ValidationError(f"{name} must be a valid number, got '{shot_str}'")  # noqa: B904

    return shot_str


def validate_output_directory(
    output_dir: str | Path, name: str = "output_dir"
) -> str:
    """Validate a "output directory ({name})" parameter."""
    if output_dir is None:
        raise ValidationError(f"{name} cannot be None")

    output_str = str(output_dir).strip()

    if not output_str:
        raise ValidationError(f"{name} cannot be empty")

    return output_str


def validate_analysis_type(analysis_type: str, name: str = "analysis_type") -> str:
    """Validate a "post-processing analysis type ({name})" parameter."""
    valid_types = ["phase", "density", "spectrum", "stft", "cwt"]

    if analysis_type is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(analysis_type, str):
        raise ValidationError(f"{name} must be a string, got {type(analysis_type)}")

    if analysis_type not in valid_types:
        raise ValidationError(
            f"{name} must be one of {valid_types}, got '{analysis_type}'"
        )

    return analysis_type


def validate_methods_list(methods: list, name: str = "methods") -> list:
    """Validate a "phase reconstruction method ({name})" parameter."""
    if methods is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(methods, list):
        raise ValidationError(f"{name} must be a list, got {type(methods)}")

    if len(methods) == 0:
        raise ValidationError(f"{name} cannot be empty")

    valid_methods = ["stacking", "stft", "cwt", "cdm", "cordic"]

    for method in methods:
        if not isinstance(method, str):
            raise ValidationError(f"All methods must be strings, got {type(method)}")
        if method not in valid_methods:
            raise ValidationError(
                f"Invalid method '{method}'. Must be one of {valid_methods}"
            )

    return methods
