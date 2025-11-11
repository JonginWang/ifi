#!/usr/bin/env python3
"""
Phase Analysis Module
=====================

This module implements three methodologies for detecting phase changes between
reference and probe signals, especially useful when signals are weakened by
noise or IQ offset issues.

Methods:
    1. Signal Stacking: Stack 2^n periods at 2^n T intervals to enhance signal
    2. STFT Ridge Analysis: Use overlap STFT with ridge detection and band analysis
    3. CWT Phase Reconstruction: Use filtfilt + FIR decimation + CWT for phase recovery

Functions:
    _cordic_rotation_jit: Numba JIT compiled CORDIC rotation for arrays
    _cordic_vectoring_jit: Numba JIT compiled CORDIC vectoring mode for arrays

Classes:
    CORDICProcessor: CORDIC algorithm based phase calculation
    SignalStacker: Implements signal stacking methodology
    STFTRidgeAnalyzer: Implements STFT ridge analysis
    CWTPhaseReconstructor: Implements CWT phase reconstruction
    PhaseChangeDetector: Main class for phase change detection
"""

import numpy as np
from scipy import signal as spsig
from typing import Tuple, Optional, List, Dict, Any, Union
from numba import njit, prange
import ssqueezepy as ssqpy
from ssqueezepy.experimental import scale_to_freq
try:
    from ..utils.common import LogManager, log_tag
    from .spectrum import SpectrumAnalysis
    from .phi2ne import PhaseConverter
    from ..utils.cache_setup import setup_project_cache  # noqa: F401
    from ..utils.validation import (
        validate_sampling_frequency,
        validate_signal,
        validate_frequency,
        validate_positive_number,
        validate_signals_match,
        validate_method,
    )
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag
    from ifi.analysis.spectrum import SpectrumAnalysis
    from ifi.analysis.phi2ne import PhaseConverter
    from ifi.utils.cache_setup import setup_project_cache  # noqa: F401
    from ifi.utils.validation import (
        validate_sampling_frequency,
        validate_signal,
        validate_frequency,
        validate_positive_number,
        validate_signals_match,
        validate_method,
    )

# cache_config = setup_project_cache()
logger = LogManager().get_logger(__name__)


@njit
def _cordic_rotation_jit(
    x: np.ndarray,
    y: np.ndarray,
    target_angles: np.ndarray,
    angles: np.ndarray,
    n_iterations: int,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba JIT compiled CORDIC rotation for arrays.

    Args:
        x (np.ndarray): Real components array
        y (np.ndarray): Imaginary components array
        target_angles (np.ndarray): Target angles array
        angles (np.ndarray): Precomputed angle table
        n_iterations (int): Number of CORDIC iterations
        scale_factor (float): CORDIC scale factor

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (magnitudes, phases)
    """
    n = len(x)

    # Initialize arrays
    x_rot = x.copy()
    y_rot = y.copy()
    angle_accum = np.zeros(n)

    # Determine rotation directions
    directions = np.where(target_angles < 0, -1, 1)

    # CORDIC iterations - vectorized
    for i in range(n_iterations):
        # Calculate rotation factors
        factor = 2.0 ** (-i)

        # Apply rotations where needed
        for j in range(n):
            # Check rotation condition for each element
            should_rotate = (
                directions[j] * angle_accum[j] < directions[j] * target_angles[j]
            )

            if should_rotate:
                x_new = x_rot[j] - directions[j] * y_rot[j] * factor
                y_new = y_rot[j] + directions[j] * x_rot[j] * factor
                angle_accum[j] += directions[j] * angles[i]
            else:
                x_new = x_rot[j] + directions[j] * y_rot[j] * factor
                y_new = y_rot[j] - directions[j] * x_rot[j] * factor
                angle_accum[j] -= directions[j] * angles[i]

            x_rot[j] = x_new
            y_rot[j] = y_new

    # Calculate magnitudes and normalize
    magnitudes = np.sqrt(x_rot**2 + y_rot**2) / scale_factor

    return magnitudes, angle_accum


@njit(parallel=True)
def _cordic_vectoring_jit(
    x: np.ndarray,
    y: np.ndarray,
    angle_table: np.ndarray,  # alpha[i] = atan(2^-i)
    scale_factor: float,  # ∏√(1+2^{-2i}) = 1/K_n
    zero_tol: float = 0.0,  # example: 1e-15
    nan_phase: bool = False,  # True: set phase to NaN for zero vectors
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba JIT compiled CORDIC vectoring mode for arrays.

    Vectoring mode: Drive y -> 0, compute phase directly.
    This is optimal when we only need phase calculation (not rotation to target angle).

    Args:
        x (np.ndarray): Real components array
        y (np.ndarray): Imaginary components array
        angle_table (np.ndarray): Precomputed angle table [arctan(2^(-i)) for i in range(n_iter)]
        scale_factor (float): CORDIC scale factor K_n
        zero_tol (float): Tolerance for detecting zero vectors (default: 0.0)
        nan_phase (bool): If True, set phase to NaN for zero vectors (default: False)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Tuple of (magnitudes, phases) where phases are in [-π, π] range
    """
    n = x.shape[0]
    n_iter = angle_table.shape[0]

    pow2 = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        pow2[i] = 1.0 / (1 << i)

    xr = x.copy()
    yr = y.copy()
    z = np.zeros(n, dtype=np.float64)

    # (0,0) mask: detect vectors close to (0,0)
    zero_mask = np.empty(n, dtype=np.uint8)
    for j in range(n):
        zero_mask[j] = 1 if (abs(xr[j]) + abs(yr[j]) <= zero_tol) else 0

    # Quadrant folding: reverse x < 0 and correct z by ±π
    # vectoring iteration converges more stably in the right half-plane (x≥0), avoiding overflow at boundaries like (-1,0)
    for j in range(n):
        if zero_mask[j] == 1:
            continue
        if xr[j] < 0.0:
            yold = yr[j]
            xr[j] = -xr[j]
            yr[j] = -yr[j]
            z[j] = np.pi if yold >= 0.0 else -np.pi

    # Main CORDIC loop - parallelized across elements
    for i in range(n_iter):
        a = angle_table[i]
        f = pow2[i]
        for j in prange(n):
            if zero_mask[j] == 1:
                continue
            dj = -1.0 if yr[j] >= 0.0 else 1.0
            x_new = xr[j] - dj * yr[j] * f
            y_new = yr[j] + dj * xr[j] * f
            # core patch: invert phase accumulation sign
            z[j] = z[j] - dj * a
            xr[j] = x_new
            yr[j] = y_new

    mags = np.abs(xr) / scale_factor

    # (0,0) sample processing
    for j in range(n):
        if zero_mask[j] == 1:
            mags[j] = 0.0
            z[j] = np.nan if nan_phase else 0.0

    # final phase [-π, π] wrapping
    for j in range(n):
        if zero_mask[j] == 1:
            continue
        zj = z[j]
        while zj <= -np.pi:
            zj += 2.0 * np.pi
        while zj > np.pi:
            zj -= 2.0 * np.pi
        z[j] = zj

    return mags, z


class CORDICProcessor:
    """
    CORDIC (COordinate Rotation DIgital Computer) processor for phase calculations.

    Attributes:
        n_iterations (int): Number of CORDIC iterations
        angles (np.ndarray): Precomputed angle table [arctan(2^(-i)) for i in range(n_iterations)]
        scale_factor (float): CORDIC scale factor K_n
    """

    def __init__(self, n_iterations: int = 16):
        """
        Initialize CORDIC processor.

        Args:
            n_iterations (int): Number of CORDIC iterations (default: 16)
        """
        self.n_iterations = n_iterations
        # Use float arithmetic for negative powers
        self.angles = np.array([np.arctan(2.0 ** (-i)) for i in range(n_iterations)])
        self.scale_factor = np.prod(
            np.sqrt(1.0 + 2.0 ** (-2 * np.arange(n_iterations)))
        )

    def cordic(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        target_angle: Optional[Union[float, np.ndarray]] = None,
        method: str = "rotation",
        zero_tol: float = 0.0,
        nan_phase: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, float]]:
        """
        Unified CORDIC method that handles both scalars and arrays for rotation or vectoring mode.

        Args:
            x (float | np.ndarray): Real component(s)
            y (float | np.ndarray): Imaginary component(s)
            target_angle (float | np.ndarray, optional): Target angle(s) for rotation mode.
                                                         Required for "rotation" method, ignored for "vectoring".
            method (str): Method to use ("rotation" or "vectoring"). Default is "rotation".
            zero_tol (float): Tolerance for detecting zero vectors in vectoring mode (default: 0.0).
                              Ignored for rotation mode.
            nan_phase (bool): If True, set phase to NaN for zero vectors in vectoring mode (default: False).
                              Ignored for rotation mode.
            **kwargs: Additional keyword arguments (currently unused, reserved for future use).
            
        Returns:
            Tuple containing (magnitudes, phases, scale_factor):
            - If inputs are scalars: (float, float, float)
            - If inputs are arrays: (np.ndarray, np.ndarray, float)
            
        Raises:
            ValueError: If method is invalid, or if arrays have mismatched lengths.
            TypeError: If target_angle is required but not provided for rotation mode.
        """
        # Validate method
        if method not in ("rotation", "vectoring"):
            raise ValueError(f"Invalid method: {method}. Must be 'rotation' or 'vectoring'")
        
        # Check if inputs are arrays (not strings, bytes, or other non-numeric sequence types)
        # Use hasattr to detect array-like objects (including plot lists, numpy arrays, lists, tuples)
        # Exclude strings and bytes which have __len__ but are not numeric arrays
        has_length = hasattr(x, "__len__") and not isinstance(x, (str, bytes))
        
        if not has_length:
            # Scalar input: convert to arrays for unified processing
            try:
                x = np.asarray([x], dtype=np.float64)
                y = np.asarray([y], dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Inputs x and y must be numeric scalars or array-like objects. "
                    f"Got x type: {type(x)}, y type: {type(y)}. Error: {e}"
                )
            
            if method == "rotation":
                if target_angle is None:
                    raise TypeError("target_angle is required for rotation mode")
                try:
                    target_angle = np.asarray([target_angle], dtype=np.float64)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"target_angle must be a numeric scalar or array-like object. "
                        f"Got type: {type(target_angle)}. Error: {e}"
                    )
            is_scalar = True
        else:
            # Array-like input: ensure numpy arrays with proper dtype
            # Handle list, tuple, numpy array, or other array-like inputs (including plot lists)
            try:
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Inputs x and y must be numeric scalars or array-like objects. "
                    f"Got x type: {type(x)}, y type: {type(y)}. Error: {e}"
                )
            is_scalar = False
            
            # Validate array lengths
            if len(x) != len(y):
                raise ValueError(f"x and y arrays must have the same length. Got {len(x)} and {len(y)}")
            
            if method == "rotation":
                if target_angle is None:
                    raise TypeError("target_angle is required for rotation mode")
                try:
                    target_angle = np.asarray(target_angle, dtype=np.float64)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"target_angle must be a numeric scalar or array-like object. "
                        f"Got type: {type(target_angle)}. Error: {e}"
                    )
                if len(target_angle) != len(x):
                    raise ValueError(
                        f"target_angle array must have the same length as x and y. "
                        f"Got {len(target_angle)} and {len(x)}"
                    )
        
        n = len(x)
        
        # Extract vectoring mode parameters from kwargs (for backward compatibility)
        # Explicit parameters take precedence over kwargs
        # Only use kwargs if explicit parameter is at default value
        if "zero_tol" in kwargs:
            kwargs_zero_tol = kwargs["zero_tol"]
            # Use kwargs value only if explicit parameter is default (0.0)
            if zero_tol == 0.0 and kwargs_zero_tol != 0.0:
                zero_tol = kwargs_zero_tol
        
        if "nan_phase" in kwargs:
            kwargs_nan_phase = kwargs["nan_phase"]
            # Use kwargs value only if explicit parameter is default (False)
            if not nan_phase and kwargs_nan_phase:
                nan_phase = kwargs_nan_phase
        
        # Warn if rotation mode receives vectoring-specific parameters
        if method == "rotation":
            if (kwargs.get("zero_tol") is not None and kwargs.get("zero_tol") != 0.0) or kwargs.get("nan_phase"):
                import warnings
                warnings.warn(
                    "zero_tol and nan_phase parameters are ignored in rotation mode. "
                    "These parameters are only used in vectoring mode.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Select implementation based on array size and method
        if n > 1000:
            # Use Numba JIT for large arrays
            if method == "rotation":
                magnitudes, phases = _cordic_rotation_jit(
                    x, y, target_angle, self.angles, self.n_iterations, self.scale_factor
                )
            elif method == "vectoring":
                magnitudes, phases = _cordic_vectoring_jit(
                    x, y, self.angles, self.scale_factor, zero_tol, nan_phase
                )
        else:
            # Use pure NumPy for small arrays
            if method == "rotation":
                magnitudes, phases = self._cordic_rotation_numpy(x, y, target_angle)
            elif method == "vectoring":
                magnitudes, phases = self._cordic_vectoring_numpy(x, y, zero_tol, nan_phase)
        
        # Return scalars if input was scalar
        if is_scalar:
            return float(magnitudes[0]), float(phases[0]), self.scale_factor
        
        return magnitudes, phases, self.scale_factor


    def _cordic_rotation_numpy(
        self, x: np.ndarray, y: np.ndarray, target_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy vectorized CORDIC rotation for small arrays.

        Args:
            x (np.ndarray): Real components array
            y (np.ndarray): Imaginary components array
            target_angles (np.ndarray): Target angles array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (magnitudes, phases)
        """
        n = len(x)

        # Initialize arrays
        x_rot = x.copy()
        y_rot = y.copy()
        angle_accum = np.zeros(n)

        # Determine rotation directions
        directions = np.where(target_angles < 0, -1, 1)

        # CORDIC iterations - pure NumPy
        for i in range(self.n_iterations):
            # Create masks for rotation direction
            should_rotate = directions * angle_accum < directions * target_angles

            # Calculate rotation factors
            factor = 2.0 ** (-i)

            # Apply rotations where needed
            x_new = np.where(
                should_rotate,
                x_rot - directions * y_rot * factor,
                x_rot + directions * y_rot * factor,
            )
            y_new = np.where(
                should_rotate,
                y_rot + directions * x_rot * factor,
                y_rot - directions * x_rot * factor,
            )
            angle_accum = np.where(
                should_rotate,
                angle_accum + directions * self.angles[i],
                angle_accum - directions * self.angles[i],
            )

            x_rot, y_rot = x_new, y_new

        # Calculate magnitudes and normalize
        magnitudes = np.sqrt(x_rot**2 + y_rot**2) / self.scale_factor

        return magnitudes, angle_accum

    def _cordic_vectoring_numpy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zero_tol: float = 0.0,
        nan_phase: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy vectorized CORDIC vectoring mode for small arrays.

        Args:
            x (np.ndarray): Real components array
            y (np.ndarray): Imaginary components array
            zero_tol (float): Tolerance for detecting zero vectors (default: 0.0)
            nan_phase (bool): If True, set phase to NaN for zero vectors (default: False)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (magnitudes, phases)
        """
        n = len(x)

        # Initialize arrays
        xr = x.copy()
        yr = y.copy()
        z = np.zeros(n, dtype=np.float64)

        # Precompute pow2 array
        pow2 = np.array(
            [1.0 / (1 << i) for i in range(self.n_iterations)], dtype=np.float64
        )

        # (0,0) mask: detect vectors close to (0,0)
        zero_mask = (np.abs(xr) + np.abs(yr) <= zero_tol).astype(np.uint8)

        # Quadrant folding: reverse x < 0 and correct z by ±π
        # vectoring iteration converges more stably in the right half-plane (x≥0), avoiding overflow at boundaries like (-1,0)
        for j in range(n):
            if zero_mask[j] == 1:
                continue
            if xr[j] < 0.0:
                yold = yr[j]
                xr[j] = -xr[j]
                yr[j] = -yr[j]
                z[j] = np.pi if yold >= 0.0 else -np.pi

        # CORDIC iterations - pure NumPy
        for i in range(self.n_iterations):
            a = self.angles[i]
            f = pow2[i]

            # Handle zero vectors - skip computation
            non_zero_mask = zero_mask == 0

            # Determine rotation direction to drive y -> 0
            # Classic vectoring: d = -sign(y)
            dj = np.where(yr >= 0.0, -1.0, 1.0)

            # CORDIC rotation with phase sign inversion
            x_new = np.where(non_zero_mask, xr - dj * yr * f, xr)
            y_new = np.where(non_zero_mask, yr + dj * xr * f, yr)
            z = np.where(
                non_zero_mask, z - dj * a, z
            )  # core patch: invert phase accumulation sign

            xr = x_new
            yr = y_new

        # Magnitude = |xr| / scale_factor (yr -> 0 after iterations)
        magnitudes = np.abs(xr) / self.scale_factor

        # (0,0) sample processing
        for j in range(n):
            if zero_mask[j] == 1:
                magnitudes[j] = 0.0
                z[j] = np.nan if nan_phase else 0.0

        # final phase [-π, π] wrapping
        for j in range(n):
            if zero_mask[j] == 1:
                continue
            zj = z[j]
            while zj <= -np.pi:
                zj += 2.0 * np.pi
            while zj > np.pi:
                zj -= 2.0 * np.pi
            z[j] = zj

        return magnitudes, z

    def extract_phase_samples(
        self,
        signal: np.ndarray,
        f0: float,
        fs: float,
        samples_per_period: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract phase samples using CORDIC algorithm.

        Uses Hilbert Transform to create analytic signal, then extracts phase using CORDIC.
        By default, uses all samples to maintain maximum time resolution.
        For computational efficiency, can resample using anti-aliasing filter.

        Args:
            signal (np.ndarray): Input signal
            f0 (float): Fundamental frequency
            fs (float): Sampling frequency
            samples_per_period (Optional[int]):
                - If None: Use all samples (maintains time resolution, recommended for stacked signals)
                - If int: Extract this many samples per period using anti-aliased decimation
                - Default: None (use all samples)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (time_points, phase_values)
        """
        # Create analytic signal using Hilbert Transform
        # This is the correct way to extract instantaneous phase from a real signal.
        # Analytic signal: z(t) = signal(t) + j * H{signal(t)}
        # where H{} is the Hilbert Transform.
        analytic = spsig.hilbert(signal)

        if samples_per_period is None:
            # Use all samples - maintains maximum time resolution
            # This is recommended for stacked signals where noise is already reduced
            time_points = np.arange(len(signal)) / fs
            x_values = np.real(analytic)
            y_values = np.imag(analytic)
        else:
            # Resample using anti-aliasing filter to prevent phase distortion
            # This maintains phase information while reducing computational load
            T = 1 / f0
            samples_per_T = int(T * fs)

            if samples_per_T < 4:
                # If period is too short, use every sample
                time_points = np.arange(len(signal)) / fs
                x_values = np.real(analytic)
                y_values = np.imag(analytic)
            else:
                # Calculate decimation factor to achieve desired samples_per_period
                # Target: samples_per_period samples per period
                # Current: samples_per_T samples per period
                decimation_factor = max(1, samples_per_T // samples_per_period)

                # Check if decimation is needed and feasible
                if decimation_factor == 1 or samples_per_period >= samples_per_T:
                    # No decimation needed, use all samples
                    time_points = np.arange(len(signal)) / fs
                    x_values = np.real(analytic)
                    y_values = np.imag(analytic)
                elif len(signal) < decimation_factor * 2:
                    # Signal too short for decimation, use all samples
                    time_points = np.arange(len(signal)) / fs
                    x_values = np.real(analytic)
                    y_values = np.imag(analytic)
                else:
                    # Use decimate with anti-aliasing filter to preserve phase information
                    # decimate uses FIR filter to prevent aliasing before downsampling
                    # This ensures phase information is preserved during downsampling
                    try:
                        x_values_decimated = spsig.decimate(
                            np.real(analytic), decimation_factor, ftype="fir"
                        )
                        y_values_decimated = spsig.decimate(
                            np.imag(analytic), decimation_factor, ftype="fir"
                        )

                        # Calculate time points for decimated signal
                        time_points = np.arange(len(x_values_decimated)) * (
                            decimation_factor / fs
                        )

                        x_values = x_values_decimated
                        y_values = y_values_decimated
                    except (ValueError, RuntimeError):
                        # If decimation fails (e.g., invalid cutoff frequency), fall back to all samples
                        time_points = np.arange(len(signal)) / fs
                        x_values = np.real(analytic)
                        y_values = np.imag(analytic)

        # Use unified CORDIC vectoring mode to calculate phase from analytic signal
        # This computes atan2(imag, real) which gives the instantaneous phase
        magnitudes, phases, _ = self.cordic(
            x_values, y_values, method="vectoring", target_angle=None
        )

        # Unwrap phases to maintain continuity
        # CORDIC vectoring mode outputs phases in [-π, π] range.
        # When actual phase changes exceed this range, wrapping occurs.
        # Use numpy.unwrap to automatically detect and correct 2π jumps.
        if len(phases) > 1:
            phases = np.unwrap(phases)

        return time_points, phases


class SignalStacker:
    """
    Implements signal stacking methodology for phase change detection.

    Method: Stack 2^n periods at 2^n T intervals to enhance signal strength.
    """

    def __init__(self, fs: float):
        """
        Initialize SignalStacker.

        Args:
            fs: Sampling frequency in Hz
        """
        # Validate input parameters
        self.fs = validate_sampling_frequency(fs, "sampling_frequency")

        self.spectrum_analyzer = SpectrumAnalysis()
        self.phase_converter = PhaseConverter()
        self.cordic_processor = CORDICProcessor()

    def find_fundamental_frequency(
        self, signal: np.ndarray, f_range: Tuple[float, float] = (0.1, None)
    ) -> float:
        """
        Find the fundamental frequency of the signal.

        This method delegates to SpectrumAnalysis.find_center_frequency_fft() for
        efficient frequency detection while maintaining the same interface for
        backward compatibility.

        Args:
            signal: Input signal
            f_range: Frequency range to search (min_f, max_f)

        Returns:
            Fundamental frequency in Hz

        Raises:
            ValueError: If f_range is invalid or no frequencies found in range
        """
        # Validate input parameters
        signal = validate_signal(signal, "signal", min_length=2)

        if len(f_range) != 2:
            raise ValueError("f_range must be a tuple of (min_f, max_f)")

        # Set default maximum frequency to Nyquist frequency
        if f_range[1] is None:
            f_range_processed = (validate_frequency(f_range[0], "f_range[0]"), self.fs / 2)
        else:
            f_range_processed = (
                validate_frequency(f_range[0], "f_range[0]"),
                validate_frequency(f_range[1], "f_range[1]"),
            )

        if f_range_processed[0] >= f_range_processed[1]:
            raise ValueError("f_range[0] must be less than f_range[1]")

        # Delegate to SpectrumAnalysis.find_center_frequency_fft for efficient computation
        # This uses positive frequencies only and proper normalization
        fundamental_freq = self.spectrum_analyzer.find_center_frequency_fft(
            signal, self.fs, f_range=f_range_processed
        )

        return fundamental_freq

    def stack_signals(
        self,
        signal: np.ndarray,
        f0: float,
        n_stacks: int = 4,
        time_axis: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack 2^n periods at 2^n T intervals with optional time axis support.

        Args:
            signal: Input signal
            f0: Fundamental frequency
            n_stacks: Number of stacking levels (default: 4 for 2^4 = 16 periods)
            time_axis: Optional time axis for precise period extraction

        Returns:
            Tuple of (stacked_signal, time_points)
        """
        # Validate input parameters
        signal = validate_signal(signal, "signal", min_length=2)
        f0 = validate_frequency(f0, "fundamental_frequency")
        n_stacks = validate_positive_number(n_stacks, "n_stacks")

        if time_axis is not None:
            time_axis = validate_signal(time_axis, "time_axis", min_length=2)
            if len(time_axis) != len(signal):
                raise ValueError("time_axis must have the same length as signal")

        T = 1 / f0  # Period
        samples_per_period = int(T * self.fs)

        stacked_signal = np.zeros_like(signal)
        total_weight = 0
        time_points = []

        if time_axis is not None:
            # Use time axis to find exact period boundaries
            period_boundaries = [0]
            current_time = time_axis[0]

            while current_time < time_axis[-1]:
                # Find closest time index
                time_idx = np.argmin(np.abs(time_axis - current_time))
                period_boundaries.append(time_idx)
                current_time += T

            # Extract segments based on time boundaries
            segments = []
            for i in range(len(period_boundaries) - 1):
                start_idx = period_boundaries[i]
                end_idx = period_boundaries[i + 1]

                if end_idx <= len(signal):
                    segment = signal[start_idx:end_idx]
                    segments.append(segment)
                    time_points.append(time_axis[start_idx])

            if segments:
                # Stack segments
                min_length = min(len(seg) for seg in segments)
                stacked_segments = []

                for seg in segments:
                    if len(seg) >= min_length:
                        stacked_segments.append(seg[:min_length])

                if stacked_segments:
                    stacked_signal[:min_length] = np.mean(stacked_segments, axis=0)
                    time_points = time_points[: len(stacked_segments)]
        else:
            # Original algorithm with period-based extraction
            for n in range(int(n_stacks)):
                period_count = 2**n
                interval_samples = period_count * samples_per_period

                if interval_samples > len(signal):
                    break

                # Extract segments and stack
                segments = []
                segment_times = []

                for i in range(
                    0, len(signal) - int(interval_samples) + 1, int(interval_samples)
                ):
                    segment = signal[i : i + int(interval_samples)]
                    segments.append(segment)
                    segment_times.append(i / self.fs)  # Time at start of segment

                if segments:
                    # Stack segments
                    stacked_segment = np.mean(segments, axis=0)
                    weight = len(segments)

                    # Add to result with weight
                    for i, seg in enumerate(segments):
                        start_idx = i * interval_samples
                        end_idx = start_idx + len(stacked_segment)
                        if end_idx <= len(stacked_signal):
                            stacked_signal[start_idx:end_idx] += (
                                stacked_segment * weight
                            )
                            total_weight += weight
                            time_points.append(segment_times[i])

            # Normalize
            if total_weight > 0:
                stacked_signal /= total_weight

        return stacked_signal, np.array(time_points)

    def compute_phase_difference_cdm(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute phase difference using CDM method from phi2ne.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Fundamental frequency (if None, will be detected automatically)

        Returns:
            Tuple of (phase_difference, fundamental_frequency)
        """
        if f0 is None:
            # Detect fundamental frequency from reference signal
            f0 = self.find_fundamental_frequency(ref_signal)

        # Validate fundamental frequency against Nyquist theorem
        if f0 >= self.fs / 2:
            raise ValueError(
                f"Fundamental frequency {f0:.2f} Hz exceeds Nyquist frequency {self.fs / 2:.2f} Hz. "
                f"Signal must be sampled at least 2x the highest frequency component."
            )

        # Check if frequency is too close to Nyquist (should be at least 10% margin)
        if f0 >= 0.9 * self.fs / 2:
            import logging

            logging.warning(
                f"{log_tag('STACK', 'CDM')} Fundamental frequency {f0:.2f} Hz is close to Nyquist frequency {self.fs / 2:.2f} Hz. "
                f"{log_tag('STACK', 'CDM')} Consider increasing sampling rate for better accuracy."
            )

        try:
            phase_diff = self.phase_converter.calc_phase_cdm(
                ref_signal,
                probe_signal,
                self.fs,
                f0,
                isbpf=True,
                isconj=False,
                islpf=True,
                iszif=False,
                isflip=False,
            )
        except Exception as e:
            # If CDM fails, raise the original error instead of silent fallback
            logging.error(
                f"{log_tag('STACK', 'CDM')} CDM method failed: {e}."
                f"This may indicate insufficient sampling rate "
                f"or invalid signal characteristics for CDM analysis."
            )
            raise RuntimeError(
                f"CDM method failed: {e}. This may indicate insufficient sampling rate "
                f"or invalid signal characteristics for CDM analysis."
            ) from e

        return phase_diff, f0

    def compute_phase_difference_cordic(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute phase difference using CORDIC algorithm.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Fundamental frequency (if None, will be detected automatically)

        Returns:
            Tuple of (time_points, phase_difference, fundamental_frequency)
        """
        if f0 is None:
            # Detect fundamental frequency from reference signal
            f0 = self.find_fundamental_frequency(ref_signal)

        # Extract phase samples using CORDIC
        ref_times, ref_phases = self.cordic_processor.extract_phase_samples(
            ref_signal, f0, self.fs
        )
        probe_times, probe_phases = self.cordic_processor.extract_phase_samples(
            probe_signal, f0, self.fs
        )

        # Align time points
        common_times = np.intersect1d(ref_times, probe_times)

        if len(common_times) == 0:
            # Fallback: use all available time points
            all_times = np.union1d(ref_times, probe_times)
            ref_interp = np.interp(all_times, ref_times, ref_phases)
            probe_interp = np.interp(all_times, probe_times, probe_phases)
            phase_diff = probe_interp - ref_interp
            return all_times, phase_diff, f0

        # Interpolate to common time points
        ref_interp = np.interp(common_times, ref_times, ref_phases)
        probe_interp = np.interp(common_times, probe_times, probe_phases)

        # Calculate phase difference
        phase_diff = probe_interp - ref_interp

        # Normalize to prevent drift
        if len(phase_diff) > 0:
            phase_diff = phase_diff - np.mean(phase_diff[: min(100, len(phase_diff))])

        return common_times, phase_diff, f0

    def detect_phase_changes_unified(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
        method: str = "stacking",
    ) -> Dict[str, Any]:
        """
        Unified phase change detection for constant, linear, and nonlinear phase changes.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Fundamental frequency (if None, will be detected automatically)
            method: Method to use ('stacking', 'cdm', 'cordic')

        Returns:
            Dictionary containing comprehensive phase change analysis
        """
        if f0 is None:
            f0 = self.find_fundamental_frequency(ref_signal, (0.1, self.fs / 2))

        # Get phase difference using specified method
        if method == "stacking":
            phase_diff, detected_f0 = self.compute_phase_difference(
                ref_signal, probe_signal, f0, "stacking"
            )
            times = np.arange(len(phase_diff)) / self.fs
        elif method == "cdm":
            phase_diff, detected_f0 = self.compute_phase_difference(
                ref_signal, probe_signal, f0, "cdm"
            )
            times = np.arange(len(phase_diff)) / self.fs
        elif method == "cordic":
            times, phase_diff, detected_f0 = self.compute_phase_difference_cordic(
                ref_signal, probe_signal, f0
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Comprehensive phase change analysis
        analysis = self._analyze_phase_changes_comprehensive(phase_diff, times, f0)

        return {
            "method": method,
            "fundamental_frequency": detected_f0,
            "sampling_frequency": self.fs,
            "time_points": times,
            "phase_difference": phase_diff,
            "analysis": analysis,
        }

    def _analyze_phase_changes_comprehensive(
        self, phase_diff: np.ndarray, times: np.ndarray, f0: float
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of phase changes including constant, linear, and nonlinear components.

        Args:
            phase_diff: Phase difference array
            times: Time points
            f0: Fundamental frequency

        Returns:
            Comprehensive phase change analysis
        """
        if len(phase_diff) < 3:
            return {"error": "Insufficient data points for analysis"}

        x = np.arange(len(phase_diff))

        # 1. Constant phase change analysis
        constant_phase = np.mean(phase_diff)
        constant_std = np.std(phase_diff)

        # 2. Linear phase change analysis
        linear_coeffs = np.polyfit(x, phase_diff, 1)
        linear_slope = linear_coeffs[0] * self.fs  # Convert to rad/s
        linear_intercept = linear_coeffs[1]

        # Linear fit quality
        y_linear = linear_coeffs[0] * x + linear_coeffs[1]
        ss_res_linear = np.sum((phase_diff - y_linear) ** 2)
        ss_tot = np.sum((phase_diff - np.mean(phase_diff)) ** 2)
        r_squared_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0

        # 3. Nonlinear phase change analysis
        # Quadratic fit
        quad_coeffs = np.polyfit(x, phase_diff, 2)
        y_quad = quad_coeffs[0] * x**2 + quad_coeffs[1] * x + quad_coeffs[2]
        ss_res_quad = np.sum((phase_diff - y_quad) ** 2)
        r_squared_quad = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0

        # Cubic fit for higher-order nonlinearity
        if len(phase_diff) > 4:
            cubic_coeffs = np.polyfit(x, phase_diff, 3)
            y_cubic = (
                cubic_coeffs[0] * x**3
                + cubic_coeffs[1] * x**2
                + cubic_coeffs[2] * x
                + cubic_coeffs[3]
            )
            ss_res_cubic = np.sum((phase_diff - y_cubic) ** 2)
            r_squared_cubic = 1 - (ss_res_cubic / ss_tot) if ss_tot > 0 else 0
        else:
            y_cubic = y_quad
            r_squared_cubic = r_squared_quad

        # 4. Residual analysis
        residuals_linear = phase_diff - y_linear
        residuals_quad = phase_diff - y_quad
        residuals_cubic = phase_diff - y_cubic

        # 5. Change type classification
        change_type = self._classify_phase_change_type(
            r_squared_linear,
            r_squared_quad,
            r_squared_cubic,
            constant_std,
            linear_slope,
            quad_coeffs[0] if len(quad_coeffs) > 0 else 0,
        )

        # 6. Signal quality metrics
        snr_estimate = self._estimate_signal_quality(phase_diff, residuals_linear)

        return {
            "constant_phase": {
                "mean": constant_phase,
                "std": constant_std,
                "range": np.max(phase_diff) - np.min(phase_diff),
            },
            "linear_phase": {
                "slope_rad_per_sample": linear_coeffs[0],
                "slope_rad_per_sec": linear_slope,
                "intercept": linear_intercept,
                "r_squared": r_squared_linear,
                "fit": y_linear,
            },
            "nonlinear_phase": {
                "quadratic_coeffs": quad_coeffs.tolist(),
                "cubic_coeffs": cubic_coeffs.tolist() if len(phase_diff) > 4 else None,
                "r_squared_quadratic": r_squared_quad,
                "r_squared_cubic": r_squared_cubic,
                "quadratic_fit": y_quad,
                "cubic_fit": y_cubic,
            },
            "residuals": {
                "linear_std": np.std(residuals_linear),
                "quadratic_std": np.std(residuals_quad),
                "cubic_std": np.std(residuals_cubic),
                "linear_max": np.max(np.abs(residuals_linear)),
                "quadratic_max": np.max(np.abs(residuals_quad)),
                "cubic_max": np.max(np.abs(residuals_cubic)),
            },
            "change_classification": change_type,
            "signal_quality": snr_estimate,
            "model_comparison": {
                "linear_vs_constant": r_squared_linear
                - 0.5,  # Threshold for linear detection
                "quadratic_vs_linear": r_squared_quad - r_squared_linear,
                "cubic_vs_quadratic": r_squared_cubic - r_squared_quad,
            },
        }

    def _classify_phase_change_type(
        self,
        r_linear: float,
        r_quad: float,
        r_cubic: float,
        constant_std: float,
        linear_slope: float,
        quad_coeff: float,
    ) -> str:
        """
        Classify the type of phase change based on analysis results.

        Args:
            r_linear: R-squared for linear fit
            r_quad: R-squared for quadratic fit
            r_cubic: R-squared for cubic fit
            constant_std: Standard deviation of constant component
            linear_slope: Linear slope in rad/s
            quad_coeff: Quadratic coefficient

        Returns:
            Classification string
        """
        # Thresholds for classification
        linear_threshold = 0.8
        quad_threshold = 0.9
        cubic_threshold = 0.95

        # Constant change detection
        if constant_std < 0.1 and abs(linear_slope) < 0.01:
            return "constant"

        # Linear change detection
        if r_linear > linear_threshold and (r_quad - r_linear) < 0.1:
            return "linear"

        # Quadratic change detection
        if r_quad > quad_threshold and (r_cubic - r_quad) < 0.05:
            return "quadratic"

        # Cubic or higher-order change
        if r_cubic > cubic_threshold:
            return "cubic_or_higher"

        # Complex or noisy change
        if r_linear < 0.5:
            return "complex_or_noisy"

        return "mixed"

    def _estimate_signal_quality(
        self, phase_diff: np.ndarray, residuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate signal quality metrics.

        Args:
            phase_diff: Phase difference array
            residuals: Residuals from linear fit

        Returns:
            Signal quality metrics
        """
        # Signal-to-noise ratio estimate
        signal_power = np.var(phase_diff)
        noise_power = np.var(residuals)
        snr_db = (
            10 * np.log10(signal_power / noise_power)
            if noise_power > 0
            else float("inf")
        )

        # Phase stability
        phase_stability = 1.0 / (1.0 + np.std(phase_diff))

        # Linearity index (how well linear model fits)
        linearity_index = (
            1.0 - (np.var(residuals) / np.var(phase_diff))
            if np.var(phase_diff) > 0
            else 0
        )

        return {
            "snr_db": snr_db,
            "phase_stability": phase_stability,
            "linearity_index": linearity_index,
            "signal_power": signal_power,
            "noise_power": noise_power,
        }

    def analyze_linear_phase_drift(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze linear phase drift between reference and probe signals.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Fundamental frequency (if None, will be detected automatically)

        Returns:
            Dictionary containing linear drift analysis results
        """
        if f0 is None:
            f0 = self.find_fundamental_frequency(ref_signal)

        # Extract phase samples for both signals
        ref_times, ref_phases = self.cordic_processor.extract_phase_samples(
            ref_signal, f0, self.fs
        )
        probe_times, probe_phases = self.cordic_processor.extract_phase_samples(
            probe_signal, f0, self.fs
        )

        # Align time points
        common_times = np.intersect1d(ref_times, probe_times)

        if len(common_times) < 2:
            return {"error": "Insufficient common time points for drift analysis"}

        # Interpolate to common time points
        ref_interp = np.interp(common_times, ref_times, ref_phases)
        probe_interp = np.interp(common_times, probe_times, probe_phases)

        # Calculate phase difference
        phase_diff = probe_interp - ref_interp

        # Linear fit to phase difference
        x = np.arange(len(phase_diff))
        coeffs = np.polyfit(x, phase_diff, 1)
        linear_slope = coeffs[0]  # radians per sample
        linear_intercept = coeffs[1]

        # Convert to radians per second
        slope_rad_per_sec = linear_slope * self.fs

        # Calculate R-squared for linear fit
        y_pred = coeffs[0] * x + coeffs[1]
        ss_res = np.sum((phase_diff - y_pred) ** 2)
        ss_tot = np.sum((phase_diff - np.mean(phase_diff)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate residual statistics
        residuals = phase_diff - y_pred
        residual_std = np.std(residuals)
        residual_max = np.max(np.abs(residuals))

        # Detect non-linear components (quadratic fit)
        quad_coeffs = np.polyfit(x, phase_diff, 2)
        y_quad = quad_coeffs[0] * x**2 + quad_coeffs[1] * x + quad_coeffs[2]
        ss_res_quad = np.sum((phase_diff - y_quad) ** 2)
        r_squared_quad = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0

        # Non-linear component strength
        non_linear_strength = r_squared_quad - r_squared

        return {
            "linear_slope_rad_per_sample": linear_slope,
            "linear_slope_rad_per_sec": slope_rad_per_sec,
            "linear_intercept": linear_intercept,
            "r_squared_linear": r_squared,
            "r_squared_quadratic": r_squared_quad,
            "non_linear_strength": non_linear_strength,
            "residual_std": residual_std,
            "residual_max": residual_max,
            "time_points": common_times,
            "phase_difference": phase_diff,
            "linear_fit": y_pred,
            "quadratic_fit": y_quad,
            "fundamental_frequency": f0,
        }

    def compute_phase_difference(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
        method: str = "cdm",
    ) -> Tuple[np.ndarray, float]:
        """
        Compute phase difference between reference and probe signals using specified method.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Fundamental frequency (if None, will be detected automatically)
            method: Method to use ('cdm', 'cordic', 'stacking')

        Returns:
            Tuple of (phase_difference, fundamental_frequency)
        """
        # Validate input parameters
        ref_signal = validate_signal(ref_signal, "ref_signal", min_length=2)
        probe_signal = validate_signal(probe_signal, "probe_signal", min_length=2)
        validate_signals_match(ref_signal, probe_signal, "ref_signal", "probe_signal")

        valid_methods = ["cdm", "cordic", "stacking"]
        method = validate_method(method, valid_methods, "method")

        if f0 is not None:
            f0 = validate_frequency(f0, "fundamental_frequency")
        else:
            # Detect fundamental frequency from reference signal
            f0 = self.find_fundamental_frequency(ref_signal)

        if method == "cdm":
            return self.compute_phase_difference_cdm(ref_signal, probe_signal, f0)
        elif method == "cordic":
            times, phase_diff, f0 = self.compute_phase_difference_cordic(
                ref_signal, probe_signal, f0
            )
            return phase_diff, f0
        elif method == "stacking":
            # Original stacking method
            ref_stacked, _ = self.stack_signals(ref_signal, f0)
            probe_stacked, _ = self.stack_signals(probe_signal, f0)

            # Compute phase difference using Hilbert transform
            ref_analytic = spsig.hilbert(ref_stacked)
            probe_analytic = spsig.hilbert(probe_stacked)

            ref_phase = np.angle(ref_analytic)
            probe_phase = np.angle(probe_analytic)

            phase_diff = probe_phase - ref_phase

            # Unwrap phase
            phase_diff = np.unwrap(phase_diff)

            return phase_diff, f0
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'cdm', 'cordic', or 'stacking'"
            )


class STFTRidgeAnalyzer:
    """
    Implements STFT ridge analysis for phase change detection.

    Method: Use overlap STFT with ridge detection and band analysis.
    """

    def __init__(self, fs: float):
        """
        Initialize STFTRidgeAnalyzer.

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.spectrum_analyzer = SpectrumAnalysis()
        self.phase_converter = PhaseConverter()

    def compute_stft_with_ridge(
        self, 
        signal: np.ndarray, 
        f0: float, 
        nperseg: int = 1024, 
        noverlap: int = 512,
        penalty: float = 2.0,
        n_bin: int = 15,
        library: str = "ssqueezepy"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT and find frequency ridge using robust ridge tracking algorithm.

        This method uses SpectrumAnalysis.find_freq_ridge() which employs
        ssqueezepy.extract_ridges() or ridge_tracking.py algorithms with penalty
        and bin settings for more robust ridge detection compared to simple argmax.

        Args:
            signal: Input signal
            f0: Target frequency
            nperseg: Number of points per segment
            noverlap: Number of points to overlap
            penalty: Penalty for ridge extraction (0.5, 2, 5, 20, 40). Default is 2.0.
                Higher penalty reduces frequency jumps.
            n_bin: Number of bins for ridge extraction (> 15 for single, ~4 for multiple).
                Default is 15.
            library: Library to use for ridge extraction ('ssqueezepy' or 'MATLAB').
                Default is 'ssqueezepy'.

        Returns:
            Tuple of (frequencies, times, ridge_frequencies)
        """
        # Compute STFT using SpectrumAnalysis for consistency
        f, t, Zxx = self.spectrum_analyzer.compute_stft(
            signal, self.fs, nperseg=nperseg, noverlap=noverlap
        )

        # Use robust ridge tracking algorithm instead of simple argmax
        # This handles noise better and avoids false detection of low-frequency components
        ridge_freqs = self.spectrum_analyzer.find_freq_ridge(
            Zxx, f, penalty=penalty, n_bin=n_bin, method="stft", library=library
        )

        return f, t, ridge_freqs

    def extract_phase_from_ridge(
        self,
        signal: np.ndarray,
        ridge_freqs: np.ndarray,
        times: np.ndarray,
        nperseg: int = 1024,
        noverlap: int = 512,
    ) -> np.ndarray:
        """
        Extract phase from ridge frequencies.

        Args:
            signal: Input signal
            ridge_freqs: Ridge frequencies (from compute_stft_with_ridge)
            times: Time points (from compute_stft_with_ridge)
            nperseg: Number of points per segment (must match compute_stft_with_ridge)
            noverlap: Number of points to overlap (must match compute_stft_with_ridge)

        Returns:
            Phase values at ridge frequencies
        """
        # Compute STFT using SpectrumAnalysis for consistency
        f, t, Zxx = self.spectrum_analyzer.compute_stft(
            signal, self.fs, nperseg=nperseg, noverlap=noverlap
        )

        # Extract phase at ridge frequencies
        # Find closest frequency indices for each ridge frequency
        f_indices = np.searchsorted(f, ridge_freqs)
        # Clamp indices to valid range
        f_indices = np.clip(f_indices, 0, len(f) - 1)
        
        # Ensure time indices match
        if len(times) != len(ridge_freqs):
            raise ValueError(
                f"Length mismatch: times ({len(times)}) and ridge_freqs ({len(ridge_freqs)}) must have same length"
            )
        
        # Find time indices corresponding to times array
        t_indices = np.searchsorted(t, times)
        t_indices = np.clip(t_indices, 0, len(t) - 1)
        
        # Extract phases using vectorized operations
        phases = np.angle(Zxx[f_indices, t_indices])

        return phases

    def compute_phase_difference(
        self, ref_signal: np.ndarray, probe_signal: np.ndarray, f0: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute phase difference using STFT ridge analysis.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Target frequency

        Returns:
            Tuple of (times, ref_phases, probe_phases)
        """
        # Compute STFT and ridge for both signals
        f_ref, t_ref, ridge_freqs_ref = self.compute_stft_with_ridge(ref_signal, f0)
        f_probe, t_probe, ridge_freqs_probe = self.compute_stft_with_ridge(
            probe_signal, f0
        )

        # Use common time points
        t_common = t_ref if len(t_ref) <= len(t_probe) else t_probe
        ridge_freqs_common = ridge_freqs_ref[: len(t_common)]

        # Extract phases
        ref_phases = self.extract_phase_from_ridge(
            ref_signal, ridge_freqs_common, t_common
        )
        probe_phases = self.extract_phase_from_ridge(
            probe_signal, ridge_freqs_common, t_common
        )

        # Unwrap phases
        ref_phases = np.unwrap(ref_phases)
        probe_phases = np.unwrap(probe_phases)

        return t_common, ref_phases, probe_phases


class CWTPhaseReconstructor:
    """
    Implements CWT phase reconstruction for phase change detection.

    Method: Use filtfilt + FIR decimation + CWT for phase recovery.
    """

    def __init__(self, fs: float):
        """
        Initialize CWTPhaseReconstructor.

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.spectrum_analyzer = SpectrumAnalysis()
        self.phase_converter = PhaseConverter()

    def design_fir_filter(self, f0: float, bandwidth: float = 0.1) -> np.ndarray:
        """
        Design FIR filter for the target frequency band using PhaseConverter._create_bpf.

        This method delegates to PhaseConverter._create_bpf() to avoid code duplication.
        The filter design logic is already well-implemented in phi2ne.py.

        Args:
            f0: Target frequency (Hz)
            bandwidth: Bandwidth as fraction of f0 (default: 0.1 = 10%)

        Returns:
            FIR filter coefficients (bandpass filter)
        """
        # Calculate bandpass filter edges
        # Use a transition band of bandwidth * 0.5 on each side
        transition = f0 * bandwidth * 0.5
        f_pass1 = f0 * (1 - bandwidth / 2)  # Lower passband edge
        f_pass2 = f0 * (1 + bandwidth / 2)  # Upper passband edge
        f_stop1 = f_pass1 - transition  # Lower stopband edge
        f_stop2 = f_pass2 + transition  # Upper stopband edge

        # Ensure frequencies are within valid range
        nyq = 0.5 * self.fs
        if f_stop2 >= nyq:
            f_stop2 = nyq * 0.99  # Slightly below Nyquist
            if f_pass2 >= f_stop2:
                f_pass2 = f_stop2 * 0.95
        if f_stop1 < 0:
            f_stop1 = 0.01 * nyq  # Small positive value
            if f_pass1 <= f_stop1:
                f_pass1 = f_stop1 * 1.1

        # Delegate to PhaseConverter._create_bpf for filter design
        # This reuses the well-tested implementation from phi2ne.py
        filter_coeffs = self.phase_converter._create_bpf(
            f_stop1, f_pass1, f_pass2, f_stop2, self.fs, approx=False
        )

        return filter_coeffs

    def decimate_signal(
        self, signal: np.ndarray, decimation_factor: int = 4
    ) -> np.ndarray:
        """
        Decimate signal using anti-aliasing filter.

        Uses scipy.signal.decimate for phase-preserving downsampling.
        This is equivalent to the previous custom implementation but uses
        the standard library function for consistency with extract_phase_samples.

        Args:
            signal: Input signal
            decimation_factor: Decimation factor

        Returns:
            Decimated signal
        """
        # Use scipy.signal.decimate for phase-preserving decimation
        # This uses FIR anti-aliasing filter internally, same as extract_phase_samples
        try:
            decimated_signal = spsig.decimate(signal, decimation_factor, ftype="fir")
        except (ValueError, RuntimeError):
            # If decimation fails (e.g., signal too short), return original signal
            # This can happen when signal length < decimation_factor * filter_length
            decimated_signal = signal

        return decimated_signal

    def compute_cwt_phase(
        self, signal: np.ndarray, f0: float, voices_per_octave: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CWT and extract phase at target frequency.

        Args:
            signal: Input signal
            f0: Target frequency
            voices_per_octave: Number of voices per octave

        Returns:
            Tuple of (times, phases)
        """
        # Compute CWT using ssqueezepy
        result = ssqpy.cwt(signal, wavelet="gmw", nv=voices_per_octave)

        if isinstance(result, tuple):
            Wx, scales, *_ = result
        else:
            Wx = result
            scales = None

        # Convert scales to frequencies
        if scales is not None:
            try:
                # Try using scale_to_freq with proper parameters
                freqs = scale_to_freq(scales, wavelet="gmw", fs=self.fs, N=len(signal))
            except Exception:
                # Fallback: manual frequency calculation
                freqs = self.fs / scales
        else:
            # Fallback frequency calculation
            freqs = np.logspace(0, 2, Wx.shape[0])

        # Find frequency band around f0
        f_tolerance = f0 * 0.1
        f_mask = (freqs >= f0 - f_tolerance) & (freqs <= f0 + f_tolerance)

        if not np.any(f_mask):
            # If no frequencies in range, use closest
            f_idx = np.argmin(np.abs(freqs - f0))
            f_mask[f_idx] = True

        # Extract phase at target frequency band
        Wx_band = Wx[f_mask, :]
        phases = np.angle(Wx_band)

        # Average phases across frequency band
        avg_phases = np.mean(phases, axis=0)

        # Create time array
        times = np.arange(len(avg_phases)) / self.fs

        return times, avg_phases

    def compute_phase_difference(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: float,
        decimation_factor: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute phase difference using CWT phase reconstruction.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Target frequency
            decimation_factor: Decimation factor

        Returns:
            Tuple of (times, ref_phases, probe_phases)
        """
        # Decimate signals
        ref_decimated = self.decimate_signal(ref_signal, decimation_factor)
        probe_decimated = self.decimate_signal(probe_signal, decimation_factor)

        # Adjust frequency for decimated signal
        f0_decimated = f0 / decimation_factor
        fs_decimated = self.fs / decimation_factor

        # Compute CWT phases
        t_ref, ref_phases = self.compute_cwt_phase(ref_decimated, f0_decimated)
        t_probe, probe_phases = self.compute_cwt_phase(probe_decimated, f0_decimated)

        # Use common time points
        min_len = min(len(t_ref), len(t_probe))
        t_common = t_ref[:min_len]
        ref_phases = ref_phases[:min_len]
        probe_phases = probe_phases[:min_len]

        # Unwrap phases
        ref_phases = np.unwrap(ref_phases)
        probe_phases = np.unwrap(probe_phases)

        return t_common, ref_phases, probe_phases


class PhaseChangeDetector:
    """
    Main class for phase change detection using multiple methodologies.
    """

    def __init__(self, fs: float):
        """
        Initialize PhaseChangeDetector.

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.signal_stacker = SignalStacker(fs)
        self.stft_analyzer = STFTRidgeAnalyzer(fs)
        self.cwt_reconstructor = CWTPhaseReconstructor(fs)

    def detect_phase_changes(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
        methods: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect phase changes using multiple methods.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Target frequency (if None, will be detected automatically)
            methods: List of methods to use ['stacking', 'stft', 'cwt', 'cdm', 'cordic']

        Returns:
            Dictionary containing results from all methods
        """
        if methods is None:
            methods = ["stacking", "stft", "cwt", "cdm", "cordic"]

        if f0 is None:
            f0 = self.signal_stacker.find_fundamental_frequency(ref_signal)

        results = {
            "fundamental_frequency": f0,
            "sampling_frequency": self.fs,
            "methods": {},
        }

        # Method 1: Signal Stacking (original)
        if "stacking" in methods:
            try:
                phase_diff, detected_f0 = self.signal_stacker.compute_phase_difference(
                    ref_signal, probe_signal, f0, method="stacking"
                )
                results["methods"]["stacking"] = {
                    "phase_difference": phase_diff,
                    "detected_frequency": detected_f0,
                    "times": np.arange(len(phase_diff)) / self.fs,
                }
            except Exception as e:
                results["methods"]["stacking"] = {"error": str(e)}

        # Method 2: CDM (Complex Demodulation)
        if "cdm" in methods:
            try:
                phase_diff, detected_f0 = self.signal_stacker.compute_phase_difference(
                    ref_signal, probe_signal, f0, method="cdm"
                )
                results["methods"]["cdm"] = {
                    "phase_difference": phase_diff,
                    "detected_frequency": detected_f0,
                    "times": np.arange(len(phase_diff)) / self.fs,
                }
            except Exception as e:
                results["methods"]["cdm"] = {"error": str(e)}

        # Method 3: CORDIC
        if "cordic" in methods:
            try:
                times, phase_diff, detected_f0 = (
                    self.signal_stacker.compute_phase_difference_cordic(
                        ref_signal, probe_signal, f0
                    )
                )
                results["methods"]["cordic"] = {
                    "phase_difference": phase_diff,
                    "detected_frequency": detected_f0,
                    "times": times,
                }
            except Exception as e:
                results["methods"]["cordic"] = {"error": str(e)}

        # Method 4: STFT Ridge Analysis
        if "stft" in methods:
            try:
                times, ref_phases, probe_phases = (
                    self.stft_analyzer.compute_phase_difference(
                        ref_signal, probe_signal, f0
                    )
                )
                phase_diff = probe_phases - ref_phases
                results["methods"]["stft"] = {
                    "times": times,
                    "ref_phases": ref_phases,
                    "probe_phases": probe_phases,
                    "phase_difference": phase_diff,
                }
            except Exception as e:
                results["methods"]["stft"] = {"error": str(e)}

        # Method 5: CWT Phase Reconstruction
        if "cwt" in methods:
            try:
                times, ref_phases, probe_phases = (
                    self.cwt_reconstructor.compute_phase_difference(
                        ref_signal, probe_signal, f0
                    )
                )
                phase_diff = probe_phases - ref_phases
                results["methods"]["cwt"] = {
                    "times": times,
                    "ref_phases": ref_phases,
                    "probe_phases": probe_phases,
                    "phase_difference": phase_diff,
                }
            except Exception as e:
                results["methods"]["cwt"] = {"error": str(e)}

        return results

    def compare_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results from different methods.

        Args:
            results: Results from detect_phase_changes

        Returns:
            Comparison analysis
        """
        comparison = {"method_comparison": {}, "recommendations": []}

        methods = results["methods"]

        # Compare phase difference statistics
        for method_name, method_result in methods.items():
            if "error" in method_result:
                comparison["method_comparison"][method_name] = {
                    "status": "failed",
                    "error": method_result["error"],
                }
                continue

            phase_diff = method_result["phase_difference"]

            comparison["method_comparison"][method_name] = {
                "status": "success",
                "mean_phase_diff": np.mean(phase_diff),
                "std_phase_diff": np.std(phase_diff),
                "max_phase_diff": np.max(phase_diff),
                "min_phase_diff": np.min(phase_diff),
                "phase_range": np.max(phase_diff) - np.min(phase_diff),
            }

        # Generate recommendations
        successful_methods = [
            name
            for name, result in comparison["method_comparison"].items()
            if result["status"] == "success"
        ]

        if len(successful_methods) > 1:
            # Compare consistency
            phase_means = [
                comparison["method_comparison"][name]["mean_phase_diff"]
                for name in successful_methods
            ]
            phase_stds = [
                comparison["method_comparison"][name]["std_phase_diff"]
                for name in successful_methods
            ]

            mean_consistency = np.std(phase_means)
            std_consistency = np.std(phase_stds)

            if mean_consistency < 0.1:  # Less than 0.1 radian difference
                comparison["recommendations"].append(
                    "Methods show good consistency in phase detection"
                )
            else:
                comparison["recommendations"].append(
                    "Methods show significant differences - check signal quality"
                )

        # Recommend best method based on stability
        if successful_methods:
            best_method = min(
                successful_methods,
                key=lambda x: comparison["method_comparison"][x]["std_phase_diff"],
            )
            comparison["recommendations"].append(
                f"Recommended method: {best_method} (lowest phase variance)"
            )

        return comparison

    def analyze_linear_phase_drift(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
        methods: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze linear phase drift using multiple methods.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Target frequency (if None, will be detected automatically)
            methods: List of methods to use ['stacking', 'stft', 'cwt', 'cdm', 'cordic']

        Returns:
            Dictionary containing linear drift analysis from all methods
        """
        if methods is None:
            methods = ["stacking", "stft", "cwt", "cdm", "cordic"]

        if f0 is None:
            f0 = self.signal_stacker.find_fundamental_frequency(ref_signal)

        results = {
            "fundamental_frequency": f0,
            "sampling_frequency": self.fs,
            "linear_drift_analysis": {},
        }

        # Method 1: Signal Stacking with linear drift analysis
        if "stacking" in methods:
            try:
                drift_analysis = self.signal_stacker.analyze_linear_phase_drift(
                    ref_signal, probe_signal, f0
                )
                results["linear_drift_analysis"]["stacking"] = drift_analysis
            except Exception as e:
                results["linear_drift_analysis"]["stacking"] = {"error": str(e)}

        # Method 2: STFT Ridge Analysis with linear drift
        if "stft" in methods:
            try:
                times, ref_phases, probe_phases = (
                    self.stft_analyzer.compute_phase_difference(
                        ref_signal, probe_signal, f0
                    )
                )
                phase_diff = probe_phases - ref_phases

                # Linear fit analysis
                x = np.arange(len(phase_diff))
                coeffs = np.polyfit(x, phase_diff, 1)
                linear_slope = coeffs[0] * self.fs  # Convert to rad/s

                # R-squared calculation
                y_pred = coeffs[0] * x + coeffs[1]
                ss_res = np.sum((phase_diff - y_pred) ** 2)
                ss_tot = np.sum((phase_diff - np.mean(phase_diff)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                results["linear_drift_analysis"]["stft"] = {
                    "linear_slope_rad_per_sec": linear_slope,
                    "r_squared": r_squared,
                    "time_points": times,
                    "phase_difference": phase_diff,
                    "linear_fit": y_pred,
                }
            except Exception as e:
                results["linear_drift_analysis"]["stft"] = {"error": str(e)}

        # Method 3: CWT Phase Reconstruction with linear drift
        if "cwt" in methods:
            try:
                times, ref_phases, probe_phases = (
                    self.cwt_reconstructor.compute_phase_difference(
                        ref_signal, probe_signal, f0
                    )
                )
                phase_diff = probe_phases - ref_phases

                # Linear fit analysis
                x = np.arange(len(phase_diff))
                coeffs = np.polyfit(x, phase_diff, 1)
                linear_slope = coeffs[0] * self.fs  # Convert to rad/s

                # R-squared calculation
                y_pred = coeffs[0] * x + coeffs[1]
                ss_res = np.sum((phase_diff - y_pred) ** 2)
                ss_tot = np.sum((phase_diff - np.mean(phase_diff)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                results["linear_drift_analysis"]["cwt"] = {
                    "linear_slope_rad_per_sec": linear_slope,
                    "r_squared": r_squared,
                    "time_points": times,
                    "phase_difference": phase_diff,
                    "linear_fit": y_pred,
                }
            except Exception as e:
                results["linear_drift_analysis"]["cwt"] = {"error": str(e)}

        # Method 4: CDM with linear drift analysis
        if "cdm" in methods:
            try:
                phase_diff, detected_f0 = self.signal_stacker.compute_phase_difference(
                    ref_signal, probe_signal, f0, method="cdm"
                )

                # Linear fit analysis
                x = np.arange(len(phase_diff))
                coeffs = np.polyfit(x, phase_diff, 1)
                linear_slope = coeffs[0] * self.fs  # Convert to rad/s

                # R-squared calculation
                y_pred = coeffs[0] * x + coeffs[1]
                ss_res = np.sum((phase_diff - y_pred) ** 2)
                ss_tot = np.sum((phase_diff - np.mean(phase_diff)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                results["linear_drift_analysis"]["cdm"] = {
                    "linear_slope_rad_per_sec": linear_slope,
                    "r_squared": r_squared,
                    "time_points": np.arange(len(phase_diff)) / self.fs,
                    "phase_difference": phase_diff,
                    "linear_fit": y_pred,
                }
            except Exception as e:
                results["linear_drift_analysis"]["cdm"] = {"error": str(e)}

        # Method 5: CORDIC with linear drift analysis
        if "cordic" in methods:
            try:
                drift_analysis = self.signal_stacker.analyze_linear_phase_drift(
                    ref_signal, probe_signal, f0
                )
                results["linear_drift_analysis"]["cordic"] = drift_analysis
            except Exception as e:
                results["linear_drift_analysis"]["cordic"] = {"error": str(e)}

        return results

    def detect_phase_changes_unified(
        self,
        ref_signal: np.ndarray,
        probe_signal: np.ndarray,
        f0: Optional[float] = None,
        methods: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified phase change detection for constant, linear, and nonlinear phase changes.
        Uses SignalStacker's unified detection method.

        Args:
            ref_signal: Reference signal
            probe_signal: Probe signal
            f0: Target frequency (if None, will be detected automatically)
            methods: List of methods to use ['stacking', 'cdm', 'cordic']

        Returns:
            Dictionary containing unified phase change analysis from all methods
        """
        if methods is None:
            methods = ["stacking", "cdm", "cordic"]

        if f0 is None:
            f0 = self.signal_stacker.find_fundamental_frequency(ref_signal)

        results = {
            "fundamental_frequency": f0,
            "sampling_frequency": self.fs,
            "unified_analysis": {},
        }

        # Use SignalStacker's unified detection for each method
        for method in methods:
            try:
                unified_result = self.signal_stacker.detect_phase_changes_unified(
                    ref_signal, probe_signal, f0, method
                )
                results["unified_analysis"][method] = unified_result
            except Exception as e:
                results["unified_analysis"][method] = {"error": str(e)}

        return results
