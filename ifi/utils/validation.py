#!/usr/bin/env python3
"""
Input Validation Utilities
=========================

This module provides standardized input validation functions for the IFI package.
"""

import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path

from ifi.utils.common import LogManager
LogManager()
logger = LogManager().get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_sampling_frequency(fs: float, name: str = "sampling_frequency") -> float:
    """
    Validate sampling frequency parameter.
    
    Args:
        fs: Sampling frequency value
        name: Parameter name for error messages
        
    Returns:
        Validated sampling frequency
        
    Raises:
        ValidationError: If validation fails
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
        freq: Frequency value
        name: Parameter name for error messages
        
    Returns:
        Validated frequency
        
    Raises:
        ValidationError: If validation fails
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


def validate_signal(signal: Union[np.ndarray, pd.Series, list], 
                   name: str = "signal", 
                   min_length: int = 1) -> np.ndarray:
    """
    Validate signal data.
    
    Args:
        signal: Signal data
        name: Parameter name for error messages
        min_length: Minimum required length
        
    Returns:
        Validated signal as numpy array
        
    Raises:
        ValidationError: If validation fails
    """
    if signal is None:
        raise ValidationError(f"{name} cannot be None")
    
    # Convert to numpy array if needed
    if isinstance(signal, (list, pd.Series)):
        signal = np.array(signal)
    elif not isinstance(signal, np.ndarray):
        raise ValidationError(f"{name} must be array-like, got {type(signal)}")
    
    if len(signal) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} samples, got {len(signal)}")
    
    if signal.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(signal)):
        logger.warning(f"{name} contains NaN values")
    
    if np.any(np.isinf(signal)):
        logger.warning(f"{name} contains infinite values")
    
    return signal


def validate_signals_match(signal1: np.ndarray, signal2: np.ndarray, 
                          name1: str = "signal1", name2: str = "signal2") -> None:
    """
    Validate that two signals have matching lengths.
    
    Args:
        signal1: First signal
        signal2: Second signal
        name1: Name of first signal for error messages
        name2: Name of second signal for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if len(signal1) != len(signal2):
        raise ValidationError(f"{name1} and {name2} must have the same length, "
                            f"got {len(signal1)} and {len(signal2)}")


def validate_positive_number(value: Union[int, float], name: str = "value") -> float:
    """
    Validate that a number is positive.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If validation fails
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
        wavelength: Wavelength value
        name: Parameter name for error messages
        
    Returns:
        Validated wavelength
        
    Raises:
        ValidationError: If validation fails
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
        method: Method name
        valid_methods: List of valid methods
        name: Parameter name for error messages
        
    Returns:
        Validated method name
        
    Raises:
        ValidationError: If validation fails
    """
    if method is None:
        raise ValidationError(f"{name} cannot be None")
    
    if not isinstance(method, str):
        raise ValidationError(f"{name} must be a string, got {type(method)}")
    
    if method not in valid_methods:
        raise ValidationError(f"{name} must be one of {valid_methods}, got '{method}'")
    
    return method


def validate_shot_number(shot_number: Union[str, int], name: str = "shot_number") -> str:
    """
    Validate shot number parameter.
    
    Args:
        shot_number: Shot number
        name: Parameter name for error messages
        
    Returns:
        Validated shot number as string
        
    Raises:
        ValidationError: If validation fails
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


def validate_output_directory(output_dir: Union[str, Path], name: str = "output_dir") -> str:
    """
    Validate output directory parameter.
    
    Args:
        output_dir: Output directory path
        name: Parameter name for error messages
        
    Returns:
        Validated output directory path
        
    Raises:
        ValidationError: If validation fails
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
        analysis_type: Analysis type
        name: Parameter name for error messages
        
    Returns:
        Validated analysis type
        
    Raises:
        ValidationError: If validation fails
    """
    valid_types = ["phase", "density", "spectrum", "stft", "cwt"]
    
    if analysis_type is None:
        raise ValidationError(f"{name} cannot be None")
    
    if not isinstance(analysis_type, str):
        raise ValidationError(f"{name} must be a string, got {type(analysis_type)}")
    
    if analysis_type not in valid_types:
        raise ValidationError(f"{name} must be one of {valid_types}, got '{analysis_type}'")
    
    return analysis_type


def validate_methods_list(methods: list, name: str = "methods") -> list:
    """
    Validate methods list parameter.
    
    Args:
        methods: List of methods
        name: Parameter name for error messages
        
    Returns:
        Validated methods list
        
    Raises:
        ValidationError: If validation fails
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
            raise ValidationError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    return methods

