#!/usr/bin/env python3
"""
Input Validation Utilities
=========================

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
"""

import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
try:
    from .common import LogManager, log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag

LogManager()
logger = LogManager().get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_sampling_frequency(fs: float, name: str = "sampling_frequency") -> float:
    """
    Validate sampling frequency parameter.

    Args:
        fs(float): Sampling frequency value
        name(str): Parameter name for error messages

    Returns:
        float: Validated sampling frequency

    Raises:
        ValidationError: If validation fails.
    """
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
    """
    Validate frequency parameter.

    Args:
        freq(float): Frequency value
        name(str): Parameter name for error messages

    Returns:
        float: Validated frequency

    Raises:
        ValidationError: If validation fails.
    """
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
    signal: Union[np.ndarray, pd.Series, list],
    name: str = "signal",
    min_length: int = 1,
) -> np.ndarray:
    """
    Validate signal data.

    Args:
        signal(Union[np.ndarray, pd.Series, list]): Signal data
        name(str): Parameter name for error messages
        min_length(int): Minimum required length

    Returns:
        np.ndarray: Validated signal as numpy array

    Raises:
        ValidationError: If validation fails.
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

    Raises:
        ValidationError: If validation fails
    """
    if len(signal1) != len(signal2):
        raise ValidationError(
            f"{name1} and {name2} must have the same length, "
            f"got {len(signal1)} and {len(signal2)}"
        )


def validate_positive_number(value: Union[int, float], name: str = "value") -> float:
    """
    Validate that a number is positive.

    Args:
        value(Union[int, float]): Value to validate
        name(str): Parameter name for error messages

    Returns:
        float: Validated value

    Raises:
        ValidationError: If validation fails.
    """
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
    """
    Validate wavelength parameter.

    Args:
        wavelength(float): Wavelength value
        name(str): Parameter name for error messages

    Returns:
        float: Validated wavelength

    Raises:
        ValidationError: If validation fails.
    """
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
    """
    Validate method parameter.

    Args:
        method(str): Method name
        valid_methods(list): List of valid methods
        name(str): Parameter name for error messages

    Returns:
        str: Validated method name

    Raises:
        ValidationError: If validation fails.
    """
    if method is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(method, str):
        raise ValidationError(f"{name} must be a string, got {type(method)}")

    if method not in valid_methods:
        raise ValidationError(f"{name} must be one of {valid_methods}, got '{method}'")

    return method


def validate_shot_number(
    shot_number: Union[str, int], name: str = "shot_number"
) -> str:
    """
    Validate shot number parameter.

    Args:
        shot_number(Union[str, int]): Shot number
        name(str): Parameter name for error messages

    Returns:
        str: Validated shot number as string

    Raises:
        ValidationError: If validation fails.
    """
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
        raise ValidationError(f"{name} must be a valid number, got '{shot_str}'")

    return shot_str


def validate_output_directory(
    output_dir: Union[str, Path], name: str = "output_dir"
) -> str:
    """
    Validate output directory parameter.

    Args:
        output_dir(Union[str, Path]): Output directory path
        name(str): Parameter name for error messages

    Returns:
        str: Validated output directory path

    Raises:
        ValidationError: If validation fails.
    """
    if output_dir is None:
        raise ValidationError(f"{name} cannot be None")

    output_str = str(output_dir).strip()

    if not output_str:
        raise ValidationError(f"{name} cannot be empty")

    return output_str


def validate_analysis_type(analysis_type: str, name: str = "analysis_type") -> str:
    """
    Validate analysis type parameter.

    Args:
        analysis_type(str): Analysis type
        name(str): Parameter name for error messages

    Returns:
        str: Validated analysis type

    Raises:
        ValidationError: If validation fails.
    """
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
    """
    Validate methods list parameter.

    Args:
        methods(list): List of methods
        name(str): Parameter name for error messages

    Returns:
        list: Validated methods list

    Raises:
        ValidationError: If validation fails.
    """
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
