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

Classes:
    PhaseChangeDetector: Main class for phase change detection
    SignalStacker: Implements signal stacking methodology
    STFTRidgeAnalyzer: Implements STFT ridge analysis
    CWTPhaseReconstructor: Implements CWT phase reconstruction
"""

from ifi.utils.cache_setup import setup_project_cache

# cache_config = setup_project_cache()

import numpy as np
from scipy import signal as spsig
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List, Dict, Any
from numba import njit

# Import ssqueezepy after cache setup (which patches torch import)
import ssqueezepy as ssqpy
from ssqueezepy.experimental import scale_to_freq

from ifi.utils.common import LogManager
from ifi.analysis.spectrum import SpectrumAnalysis
from ifi.analysis.phi2ne import PhaseConverter

LogManager()


@njit
def _cordic_rotation_vectorized_jit(
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
        x: Real components array
        y: Imaginary components array
        target_angles: Target angles array
        angles: Precomputed angle table
        n_iterations: Number of CORDIC iterations
        scale_factor: CORDIC scale factor

    Returns:
        Tuple of (magnitudes, phases)
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


class CORDICProcessor:
    """
    CORDIC (COordinate Rotation DIgital Computer) processor for phase calculations.
    """

    def __init__(self, n_iterations: int = 16):
        """
        Initialize CORDIC processor.

        Args:
            n_iterations: Number of CORDIC iterations (default: 16)
        """
        self.n_iterations = n_iterations
        # Use float arithmetic for negative powers
        self.angles = np.array([np.arctan(2.0 ** (-i)) for i in range(n_iterations)])
        self.scale_factor = np.prod(
            np.sqrt(1.0 + 2.0 ** (-2 * np.arange(n_iterations)))
        )

    def cordic_rotation(
        self, x: float, y: float, target_angle: float
    ) -> Tuple[float, float, float]:
        """
        Perform CORDIC rotation to find magnitude and phase.

        Args:
            x: Real component
            y: Imaginary component
            target_angle: Target angle for rotation

        Returns:
            Tuple of (magnitude, phase, scale_factor)
        """
        # Initialize
        x_rot = x
        y_rot = y
        angle_accum = 0.0

        # Determine rotation direction
        if target_angle < 0:
            direction = -1
        else:
            direction = 1

        # CORDIC iterations
        for i in range(self.n_iterations):
            if direction * angle_accum < direction * target_angle:
                x_new = x_rot - direction * y_rot * 2.0 ** (-i)
                y_new = y_rot + direction * x_rot * 2.0 ** (-i)
                angle_accum += direction * self.angles[i]
            else:
                x_new = x_rot + direction * y_rot * 2.0 ** (-i)
                y_new = y_rot - direction * x_rot * 2.0 ** (-i)
                angle_accum -= direction * self.angles[i]

            x_rot, y_rot = x_new, y_new

        # Calculate magnitude and normalize
        magnitude = np.sqrt(x_rot**2 + y_rot**2) / self.scale_factor

        return magnitude, angle_accum, self.scale_factor

    def cordic_rotation_vectorized(
        self, x: np.ndarray, y: np.ndarray, target_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Vectorized CORDIC rotation for arrays of complex numbers.

        Args:
            x: Real components array
            y: Imaginary components array
            target_angles: Target angles array

        Returns:
            Tuple of (magnitudes, phases, scale_factor)
        """
        # Ensure arrays are the same length
        n = len(x)
        if len(y) != n or len(target_angles) != n:
            raise ValueError("All input arrays must have the same length")

        # Use Numba JIT for large arrays, pure NumPy for small arrays
        if n > 1000:  # Threshold for JIT compilation
            magnitudes, phases = _cordic_rotation_vectorized_jit(
                x, y, target_angles, self.angles, self.n_iterations, self.scale_factor
            )
        else:
            # Use pure NumPy vectorization for small arrays
            magnitudes, phases = self._cordic_rotation_numpy_vectorized(
                x, y, target_angles
            )

        return magnitudes, phases, self.scale_factor

    def _cordic_rotation_numpy_vectorized(
        self, x: np.ndarray, y: np.ndarray, target_angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure NumPy vectorized CORDIC rotation for small arrays.

        Args:
            x: Real components array
            y: Imaginary components array
            target_angles: Target angles array

        Returns:
            Tuple of (magnitudes, phases)
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

    def extract_phase_samples(
        self, signal: np.ndarray, f0: float, fs: float, samples_per_period: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract phase samples using CORDIC algorithm at specific points in each period.
        Uses proper CORDIC rotation for phase calculation.

        Args:
            signal: Input signal
            f0: Fundamental frequency
            fs: Sampling frequency
            samples_per_period: Number of samples to extract per period (default: 4)

        Returns:
            Tuple of (time_points, phase_values)
        """
        T = 1 / f0
        samples_per_T = int(T * fs)

        if samples_per_T < 4:
            # If period is too short, use every sample
            samples_per_T = 1
            samples_per_period = 1

        # Calculate all time points and signal indices at once
        time_points = []
        signal_indices = []

        for period in range(len(signal) // samples_per_T):
            period_start = period * samples_per_T

            for sample_idx in range(samples_per_period):
                # Calculate offset within the period
                offset = int(sample_idx * samples_per_T / samples_per_period)
                signal_idx = period_start + offset

                if signal_idx < len(signal) - 1:
                    time_points.append(signal_idx / fs)
                    signal_indices.append(signal_idx)

        if not signal_indices:
            return np.array([]), np.array([])

        # Convert to numpy arrays for vectorized operations
        signal_indices = np.array(signal_indices)
        time_points = np.array(time_points)

        # Extract all x and y values at once
        x_values = signal[signal_indices]
        y_values = signal[signal_indices + 1]

        # Use vectorized CORDIC rotation
        magnitudes, phases, _ = self.cordic_rotation_vectorized(
            x_values, y_values, np.zeros(len(x_values))
        )

        # Unwrap phases to maintain continuity
        if len(phases) > 1:
            phase_diffs = np.diff(phases)
            # Find jumps > π and correct them
            jump_indices = np.where(np.abs(phase_diffs) > np.pi)[0]
            for idx in jump_indices:
                if phase_diffs[idx] > np.pi:
                    phases[idx + 1 :] -= 2 * np.pi
                elif phase_diffs[idx] < -np.pi:
                    phases[idx + 1 :] += 2 * np.pi

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
        from ifi.utils.validation import validate_sampling_frequency

        self.fs = validate_sampling_frequency(fs, "sampling_frequency")

        self.spectrum_analyzer = SpectrumAnalysis()
        self.phase_converter = PhaseConverter()
        self.cordic_processor = CORDICProcessor()

    def find_fundamental_frequency(
        self, signal: np.ndarray, f_range: Tuple[float, float] = (0.1, None)
    ) -> float:
        """
        Find the fundamental frequency of the signal.

        Args:
            signal: Input signal
            f_range: Frequency range to search (min_f, max_f)

        Returns:
            Fundamental frequency in Hz
        """
        # Validate input parameters
        from ifi.utils.validation import validate_signal, validate_frequency

        signal = validate_signal(signal, "signal", min_length=2)

        if len(f_range) != 2:
            raise ValueError("f_range must be a tuple of (min_f, max_f)")

        # Set default maximum frequency to Nyquist frequency
        if f_range[1] is None:
            f_range = (validate_frequency(f_range[0], "f_range[0]"), self.fs / 2)
        else:
            f_range = (
                validate_frequency(f_range[0], "f_range[0]"),
                validate_frequency(f_range[1], "f_range[1]"),
            )

        if f_range[0] >= f_range[1]:
            raise ValueError("f_range[0] must be less than f_range[1]")

        # Use FFT to find dominant frequency
        freqs = fftfreq(len(signal), 1 / self.fs)
        fft_vals = np.abs(fft(signal))

        # Filter to frequency range
        mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
        freqs_filtered = freqs[mask]
        fft_vals_filtered = fft_vals[mask]

        if len(freqs_filtered) == 0:
            raise ValueError(f"No frequencies found in range {f_range}")

        # Find peak frequency
        peak_idx = np.argmax(fft_vals_filtered)
        fundamental_freq = freqs_filtered[peak_idx]

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
        from ifi.utils.validation import (
            validate_signal,
            validate_frequency,
            validate_positive_number,
        )

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
            period_boundaries = []
            current_time = 0

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
                f"Fundamental frequency {f0:.2f} Hz is close to Nyquist frequency {self.fs / 2:.2f} Hz. "
                f"Consider increasing sampling rate for better accuracy."
            )

        try:
            phase_diff = self.phase_converter.calc_phase_cdm(
                ref_signal, probe_signal, self.fs, f0, isbpf=True, isconj=True
            )
        except Exception as e:
            # If CDM fails, raise the original error instead of silent fallback
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
        from ifi.utils.validation import (
            validate_signal,
            validate_signals_match,
            validate_method,
        )

        ref_signal = validate_signal(ref_signal, "ref_signal", min_length=2)
        probe_signal = validate_signal(probe_signal, "probe_signal", min_length=2)
        validate_signals_match(ref_signal, probe_signal, "ref_signal", "probe_signal")

        valid_methods = ["cdm", "cordic", "stacking"]
        method = validate_method(method, valid_methods, "method")

        if f0 is not None:
            from ifi.utils.validation import validate_frequency

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
        self, signal: np.ndarray, f0: float, nperseg: int = 1024, noverlap: int = 512
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT and find frequency ridge.

        Args:
            signal: Input signal
            f0: Target frequency
            nperseg: Number of points per segment
            noverlap: Number of points to overlap

        Returns:
            Tuple of (frequencies, times, ridge_frequencies)
        """
        # Compute STFT
        f, t, Sxx = spsig.stft(signal, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        # Find ridge around target frequency
        f_tolerance = f0 * 0.1  # 10% tolerance
        f_mask = (f >= f0 - f_tolerance) & (f <= f0 + f_tolerance)

        # Find ridge in the band
        Sxx_band = Sxx[f_mask, :]
        f_band = f[f_mask]

        # Find peak frequency at each time
        ridge_idx = np.argmax(np.abs(Sxx_band), axis=0)
        ridge_freqs = f_band[ridge_idx]

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
            ridge_freqs: Ridge frequencies
            times: Time points
            nperseg: Number of points per segment
            noverlap: Number of points to overlap

        Returns:
            Phase values at ridge frequencies
        """
        # Compute STFT
        f, t, Sxx = spsig.stft(signal, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        # Extract phase at ridge frequencies
        phases = []
        for i, (t_val, ridge_freq) in enumerate(zip(times, ridge_freqs)):
            # Find closest frequency
            f_idx = np.argmin(np.abs(f - ridge_freq))
            phase = np.angle(Sxx[f_idx, i])
            phases.append(phase)

        return np.array(phases)

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

    def design_fir_filter(self, f0: float, bandwidth: float = 0.1) -> np.ndarray:
        """
        Design FIR filter for the target frequency band.

        Args:
            f0: Target frequency
            bandwidth: Bandwidth as fraction of f0

        Returns:
            FIR filter coefficients
        """
        # Design bandpass filter
        low_freq = f0 * (1 - bandwidth / 2)
        high_freq = f0 * (1 + bandwidth / 2)

        # Normalize frequencies
        low_norm = low_freq / (self.fs / 2)
        high_norm = high_freq / (self.fs / 2)

        # Design FIR filter
        filter_order = 101
        filter_coeffs = spsig.firwin(
            filter_order, [low_norm, high_norm], pass_zero=False, window="hamming"
        )

        return filter_coeffs

    def decimate_signal(
        self, signal: np.ndarray, decimation_factor: int = 4
    ) -> np.ndarray:
        """
        Decimate signal using FIR filter.

        Args:
            signal: Input signal
            decimation_factor: Decimation factor

        Returns:
            Decimated signal
        """
        # Design anti-aliasing filter
        filter_order = 101
        cutoff = 0.8 / decimation_factor  # 80% of Nyquist
        filter_coeffs = spsig.firwin(filter_order, cutoff, window="hamming")

        # Apply zero-phase filtering
        filtered_signal = spsig.filtfilt(filter_coeffs, 1, signal)

        # Decimate
        decimated_signal = filtered_signal[::decimation_factor]

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
