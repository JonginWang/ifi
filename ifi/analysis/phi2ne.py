#!/usr/bin/env python3
"""
Phase to Density Conversion
===========================

This module provides functions and classes for converting interferometer phase
measurements to line-integrated electron density. It supports multiple phase
extraction methods (IQ, CDM, FPGA) and includes Numba-optimized core
calculations for high performance.

Classes:
    PhaseConverter: Main class for phase-to-density conversion with multiple methods.

Phase Extraction Methods:
    - IQ Method: Uses I/Q signals directly (atan2 or asin2 cross-product)
    - CDM Method: Complex Demodulation with BPF, Hilbert transform, and LPF
    - FPGA Method: Direct phase difference from FPGA-processed data

Shot Number Ranges:
    - 30000-39265: IQ method (94 GHz)
    - 39302-41398: FPGA method (94 GHz)
    - 41542+: CDM method (94 GHz or 280 GHz based on filename)

Author: J. Wang
Date: 2025-01-16
"""

import configparser

import numpy as np
import pandas as pd
from scipy import constants
from scipy.signal import filtfilt, hilbert

from ..utils.if_utils import (
    assign_interferometry_params_to_shot,
    get_interferometry_params_by_section,
)
from ..utils.log_manager import LogManager, log_tag
from .functions.demod_jit import (
    accumulate_phase_diff_jit,
    adjust_interpolated_baseline_jit,
    apply_magnitude_threshold_jit,
    calculate_differential_phase_jit,
    compute_magnitude_stats_jit,
    normalize_iq_signals_jit,
    phase_to_density_jit,
)
from .functions.filters_remez import (
    create_bpf,
    # create_hpf,
    create_lpf,
    plot_filter_response,
)
from .functions.interpolateNonFiniteValues import interpolateNonFinite

logger = LogManager().get_logger(__name__)


class PhaseConverter:
    """
    Main class for converting interferometer phase measurements to electron density.

    This class provides methods for phase extraction (IQ, CDM, FPGA) and phase-to-density
    conversion. It uses Numba-optimized core calculations for high performance and supports
    multiple experimental configurations through automatic parameter detection.

    Attributes:
        config (configparser.ConfigParser): Configuration parser for interferometry parameters.
        constants (dict): Physical constants dictionary (c, m_e, eps0, qe) loaded from scipy.constants.

    Methods:
        Phase Extraction:
            calc_phase_iq_atan2: Calculate phase from I/Q signals using atan2 method.
            calc_phase_iq_asin2: Calculate phase from I/Q signals using asin2 (cross-product) method.
            calc_phase_iq: Convenience method for IQ phase calculation (default: atan2).
            calc_phase_cdm: Calculate phase using Complex Demodulation (CDM) method.
            calc_phase_fpga: Calculate phase difference from FPGA-processed data.
        
        Phase-to-Density Conversion:
            phase_to_density: Convert phase (radians) to line-integrated density (m^-2).
        
        Parameter Management:
            get_params: Get interferometry parameters for a specific frequency.
            get_analysis_params: Get analysis parameters based on shot number and filename.
        
        Signal Processing:
            _create_lpf: Create low-pass FIR filter using remez algorithm.
            _create_hpf: Create high-pass FIR filter using remez algorithm.
            _create_bpf: Create band-pass FIR filter using remez algorithm.
            _plot_filter_response: Plot frequency response of a filter.
        
        Data Correction:
            correct_baseline: Correct baseline offset in density data using IP or trigger time.

    Examples:
        ```python
        from ifi.analysis.phi2ne import PhaseConverter
        import numpy as np
        
        # Initialize converter
        converter = PhaseConverter()
        
        # Get parameters for a shot
        params = converter.get_analysis_params(45821, "45821_056.csv")
        print(f"Method: {params['method']}, Frequency: {params['freq_ghz']} GHz")
        
        # Calculate phase from I/Q signals (IQ method)
        i_signal = np.array([1.0, 0.5, -0.5, -1.0])
        q_signal = np.array([0.0, 0.866, 0.866, 0.0])
        phase, _ = converter.calc_phase_iq(i_signal, q_signal, iscxprod=False)
        
        # Convert phase to density
        density = converter.phase_to_density(
            phase,
            freq_hz=params['freq'],
            n_path=params['n_path']
        )
        
        # CDM method (requires reference and probe signals)
        ref_signal = np.sin(2 * np.pi * 15e6 * t)
        probe_signal = np.sin(2 * np.pi * 15e6 * t + phase)
        phase_cdm, _ = converter.calc_phase_cdm(
            ref_signal, probe_signal,
            fs=50e6, f_center=15e6
        )
        
        # Correct baseline
        density_corrected = converter.correct_baseline(
            density_df, time_axis, mode='trig'
        )
        ```
    """

    def __init__(self, config_path: str = "ifi/analysis/config_if.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        self.constants = {}
        if self.config.has_section("constants"):
            for key, name in self.config["constants"].items():
                if hasattr(constants, name):
                    self.constants[key] = getattr(constants, name)
                else:
                    raise ValueError(f"Constant '{name}' not found in scipy.constants")

    def get_params(self, freq_ghz: int) -> dict[str, any]:
        """
        Get interferometry parameters for a specific frequency from configuration.

        Args:
            freq_ghz (int): Interferometer frequency in GHz (e.g., 94 or 280).

        Returns:
            Dict[str, Any]: Dictionary containing frequency-specific parameters:
                - freq (float): Frequency in Hz.
                - n_ch (int): Number of channels.
                - n_path (int): Number of interferometer passes.
                - Other parameters from config file section.

        Raises:
            ValueError: If the frequency configuration section is not found in config file.

        Examples:
            ```python
            converter = PhaseConverter()
            params_94 = converter.get_params(94)
            params_280 = converter.get_params(280)
            print(f"94 GHz: {params_94['freq']/1e9} GHz, n_path={params_94['n_path']}")
            ```
        """
        section = f"{freq_ghz}GHz"
        return get_interferometry_params_by_section(self.config_path, section)

    def get_analysis_params(self, shot_num: int, filename: str) -> dict:
        """
        Get interferometry analysis parameters based on shot number and filename.

        This method wraps the shared interferometry utility and uses the class's
        configuration path. It provides better integration with other
        PhaseConverter methods.

        Args:
            shot_num (int): Shot number (typically 5-digit integer).
            filename (str): Name of the data file (e.g., "45821_056.csv", "45821_ALL.csv").
                The filename pattern determines frequency and channel mapping.

        Returns:
            Dict: Dictionary containing analysis parameters compatible with phase_to_density:
                - method (str): Phase extraction method.
                - freq (float): Frequency in Hz.
                - freq_ghz (float): Frequency in GHz.
                - ref_col (str or None): Reference channel name.
                - probe_cols (list): List of probe channel names.
                - n_path (int): Number of interferometer passes.
                - Additional method-specific parameters.

        Notes:
            - This method uses the class's config_path (default: "ifi/analysis/config_if.ini").
            - For detailed shot number ranges and filename patterns, see
              `ifi.utils.if_utils.assign_interferometry_params_to_shot`.

        Examples:
            ```python
            converter = PhaseConverter()
            params = converter.get_analysis_params(45821, "45821_056.csv")
            # Use params with phase_to_density
            density = converter.phase_to_density(phase, analysis_params=params)
            ```
        """
        return assign_interferometry_params_to_shot(
            shot_num,
            filename,
            config_path=self.config_path,
        )

    def _create_lpf(
        self,
        f_pass: float,
        f_stop: float,
        fs: float,
        approx: bool = False,
        max_taps: int | None = None,
    ) -> np.ndarray:
        """Delegate LPF design to shared filter utilities."""
        return create_lpf(f_pass, f_stop, fs, approx=approx, max_taps=max_taps)

    def _create_bpf(
        self,
        f_stop1: float,
        f_pass1: float,
        f_pass2: float,
        f_stop2: float,
        fs: float,
        approx: bool = False,
    ) -> np.ndarray:
        """Delegate BPF design to shared filter utilities."""
        return create_bpf(
            f_stop1,
            f_pass1,
            f_pass2,
            f_stop2,
            fs,
            approx=approx,
        )

    def _plot_filter_response(
        self,
        taps: np.ndarray,
        fs: float | None = None,
        title: str = "Frequency Response",
    ) -> None:
        """Delegate filter-response plotting to shared filter utilities."""
        plot_filter_response(taps, fs=fs, title=title)

    def phase_to_density(
        self,
        phase: np.ndarray,
        freq_hz: float = None,
        n_path: int = None,
        analysis_params: dict = None,
        wavelength: float = None,
    ) -> np.ndarray:
        """
        Convert phase (in radians) to line-integrated electron density (m^-2).

        This method implements the plasma interferometry formula to convert phase measurements
        to line-integrated electron density. The conversion uses physical constants and
        Numba-optimized core calculations for performance.

        Formula:
            ?�n_e dl = (c * n_c / (? * freq)) * phase / n_path
            where n_c = m_e * eps0 * (2? * freq)² / qe² (critical density)

        Args:
            phase (np.ndarray): Phase array in radians (1D array).
            freq_hz (float, optional): Interferometer frequency in Hz.
                Takes precedence over analysis_params and wavelength if provided.
            n_path (int, optional): Number of interferometer passes (e.g., 2 for double-pass).
                Takes precedence over analysis_params if provided.
            analysis_params (Dict, optional): Dictionary from get_analysis_params() containing:
                - 'freq' (float): Frequency in Hz.
                - 'n_path' (int): Number of passes.
                Used if freq_hz and n_path are not provided.
            wavelength (float, optional): Wavelength in meters.
                Will be converted to frequency using c = λf.
                Only used if freq_hz is not provided.

        Returns:
            np.ndarray: Line-integrated electron density in m^-2 (same length as phase).

        Raises:
            ValueError: If no valid frequency source is found (freq_hz, analysis_params, or wavelength).

        Notes:
            - Parameter precedence: freq_hz/n_path > analysis_params > wavelength > default (94 GHz).
            - Physical constants are loaded from scipy.constants during initialization.
            - The core calculation uses Numba JIT compilation for performance.
            - The formula assumes small phase approximation (valid for typical plasma densities).

        Examples:
            ```python
            converter = PhaseConverter()
            phase = np.array([0.1, 0.2, 0.3])  # radians
            
            # Direct parameters
            density = converter.phase_to_density(phase, freq_hz=94e9, n_path=2)
            
            # Using analysis parameters
            params = converter.get_analysis_params(45821, "45821_056.csv")
            density = converter.phase_to_density(phase, analysis_params=params)
            
            # Using wavelength
            density = converter.phase_to_density(
                phase, wavelength=3.19e-3  # 94 GHz wavelength in meters
            )
            ```
        """
        # Convert wavelength to frequency if provided
        if wavelength is not None:
            if freq_hz is not None:

                logger.warning(
                    f"{log_tag('PHI2N', 'PARAM')} Both freq_hz and wavelength provided. Using freq_hz."
                )
            else:
                # Convert wavelength to frequency: f = c / λ
                c = self.constants["c"]  # Speed of light
                freq_hz = c / wavelength

        # Determine frequency and n_path from various sources
        if freq_hz is not None and n_path is not None:
            # Direct parameters provided
            freq = freq_hz
            passes = n_path
        elif analysis_params is not None:
            # Use parameters from analysis params dictionary
            freq = analysis_params["freq"]  # Should be in Hz
            passes = analysis_params["n_path"]
        else:
            # Fallback to default 94GHz parameters
            freq_params = self.get_params(94)
            freq = freq_params["freq"]
            passes = freq_params["n_path"]

        # Get physical constants
        m_e = self.constants["m_e"]
        eps0 = self.constants["eps0"]
        qe = self.constants["qe"]
        c = self.constants["c"]

        # Use numba-optimized core calculation
        return phase_to_density_jit(phase, freq, c, m_e, eps0, qe, passes)

    def calc_phase_iq_atan2(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        isnorm: bool = True,
        isflip: bool = False,
        magnitude_threshold: float = None,
        interpolate_nan: bool = False,
        interpolation_method: str = "linear",
        interpolation_degree: int = 1,
        return_magnitude_stats: bool = True,
    ) -> np.ndarray | tuple:
        """
        Calculate phase from I and Q signals using atan2 method.

        This method normalizes I/Q signals to the unit circle and calculates the phase
        using arctangent. The phase is unwrapped to handle 2? discontinuities and
        calibrated to start at zero.

        Args:
            i_signal (np.ndarray): In-phase signal array (1D).
            q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.
            isnorm (bool, optional): Whether to normalize I/Q signals to unit circle.
                Default is True. If False, signals are used as-is (must already be normalized).
            isflip (bool, optional): Whether to flip the sign of the phase.
                Default is False. Useful for correcting phase convention.
            magnitude_threshold (float, optional): Minimum magnitude threshold for filtering.
                Values with magnitude below this threshold will be set to NaN.
                If None and return_magnitude_stats=True, mean magnitude * 0.1 is used as default threshold.
                If None and return_magnitude_stats=False, no threshold filtering is applied. Default is None.
            interpolate_nan (bool, optional): Whether to interpolate NaN values in the phase array.
                Default is False. Uses interpolateNonFiniteValues if True.
            interpolation_method (str, optional): Interpolation method when interpolate_nan=True.
                Options: "nearest" (degree=0), "linear" (degree=1), "quadratic" (degree=2), "cubic" (degree=3).
                Default is "linear".
            interpolation_degree (int, optional): Spline degree for interpolation (0-3).
                Only used if interpolation_method is not one of the named methods.
                Default is 1.
            return_magnitude_stats (bool, optional): Whether to return magnitude statistics.
                If True, returns tuple (phase, stats_dict) where stats_dict contains
                'mean_magnitude' and 'min_magnitude'. Default is True.
                When True and magnitude_threshold=None, automatically uses mean * 0.1 as threshold.

        Returns:
            np.ndarray or tuple: Phase array in radians (same length as input signals).
                Phase is unwrapped and calibrated to start at 0.
                If return_magnitude_stats=True, returns tuple (phase, stats_dict).

        Notes:
            - Normalization uses Numba-optimized function for performance.
            - Phase unwrapping handles 2? jumps automatically.
            - The first phase value is always 0 (calibrated).
            - Magnitude statistics are computed from the original magnitude before normalization.
            - Threshold filtering sets low-quality samples to NaN before interpolation.

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5, -1.0])
            q = np.array([0.0, 0.866, 0.866, 0.0])
            # Default: returns tuple (phase, stats_dict)
            phase, _ = converter.calc_phase_iq_atan2(i, q)
            # Result: [0, ?/3, 2?/3, ?] (approximately)
            
            # With magnitude threshold and statistics
            phase, stats = converter.calc_phase_iq_atan2(
                i, q,
                magnitude_threshold=0.5,
                interpolate_nan=True,
                return_magnitude_stats=True
            )
            print(f"Mean magnitude: {stats['mean_magnitude']:.3f}")
            
            # To get only phase without statistics
            phase = converter.calc_phase_iq_atan2(i, q, return_magnitude_stats=False)
            ```
        """
        # 1. Normalize I and Q signals (numba-optimized)
        if isnorm:
            i_norm, q_norm, iq_mag = normalize_iq_signals_jit(i_signal, q_signal)
        else:
            i_norm = i_signal
            q_norm = q_signal
            iq_mag = np.sqrt(i_signal**2 + q_signal**2)

        # Compute magnitude statistics if requested
        stats_dict = {}
        if return_magnitude_stats or magnitude_threshold is not None:
            mean_mag, min_mag = compute_magnitude_stats_jit(iq_mag)
            stats_dict = {
                "mean_magnitude": mean_mag,
                "min_magnitude": min_mag,
            }
            # If return_magnitude_stats=True and magnitude_threshold=None, use mean * 0.1 as default
            if magnitude_threshold is None and return_magnitude_stats:
                magnitude_threshold = mean_mag * 0.1

        # 2. Calculate phase by arctan (direct calculation)
        phase = np.unwrap(np.arctan2(q_norm, i_norm))
        phase -= phase[0]  # Calibrate to start at 0
        if isflip:
            phase *= -1

        # 3. Apply magnitude threshold filtering if specified
        if magnitude_threshold is not None:
            mask = apply_magnitude_threshold_jit(iq_mag, magnitude_threshold)
            phase[~mask] = np.nan

        # 4. Interpolate NaN values if requested
        if interpolate_nan and np.any(np.isnan(phase)):
            if interpolateNonFinite is None:
                logger.warning(
                    f"{log_tag('PHI2N', 'INTERP')} interpolateNonFinite not available. Skipping NaN interpolation."
                )
            else:
                # Map interpolation method to degree
                method_to_degree = {
                    "nearest": 0,
                    "linear": 1,
                    "quadratic": 2,
                    "cubic": 3,
                }
                degree = method_to_degree.get(interpolation_method, interpolation_degree)
                try:
                    phase = interpolateNonFinite(
                        phase,
                        xCoords=None,  # Uniform spacing
                        maxNonFiniteNeighbors=-1,  # No limit
                        degree=degree,
                        allowExtrapolation=False,
                        smooth=0,
                    )
                except Exception as e:
                    logger.warning(
                        f"{log_tag('PHI2N', 'INTERP')} Interpolation failed: {e}. Returning phase with NaN values."
                    )

        if return_magnitude_stats:
            return phase, stats_dict
        return phase

    def calc_phase_iq_asin2(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        isnorm: bool = True,
        isflip: bool = False,
        magnitude_threshold: float = None,
        interpolate_nan: bool = False,
        interpolation_method: str = "linear",
        interpolation_degree: int = 1,
        adjust_baseline: bool = True,
        return_magnitude_stats: bool = True,
    ) -> np.ndarray | tuple:
        """
        Calculate phase from I and Q signals using asin2 (cross-product) method.

        This method uses a differential cross-product approach to calculate phase,
        which is more robust to noise than direct atan2. It computes phase differences
        between consecutive samples and accumulates them to reconstruct absolute phase.

        The algorithm is based on MATLAB's IQPostprocessing script and uses:
        1. I/Q normalization to unit circle
        2. Differential phase calculation: ?? = arcsin(cross_product / magnitude_product)
        3. Phase accumulation via cumulative sum

        Args:
            i_signal (np.ndarray): In-phase signal array (1D).
            q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.
            isnorm (bool, optional): Whether to normalize I/Q signals to unit circle.
                Default is True. If False, signals are used as-is (must already be normalized).
            isflip (bool, optional): Whether to flip the sign of the accumulated phase.
                Default is False. Useful for correcting phase convention.
            magnitude_threshold (float, optional): Minimum magnitude threshold for filtering.
                Values with magnitude below this threshold will be set to NaN.
                If None and return_magnitude_stats=True, mean magnitude * 0.1 is used as default threshold.
                If None and return_magnitude_stats=False, no threshold filtering is applied. Default is None.
            interpolate_nan (bool, optional): Whether to interpolate NaN values in the phase array.
                Default is False. Uses interpolateNonFiniteValues if True.
            interpolation_method (str, optional): Interpolation method when interpolate_nan=True.
                Options: "nearest" (degree=0), "linear" (degree=1), "quadratic" (degree=2), "cubic" (degree=3).
                Default is "linear".
            interpolation_degree (int, optional): Spline degree for interpolation (0-3).
                Only used if interpolation_method is not one of the named methods.
                Default is 1.
            adjust_baseline (bool, optional): Whether to adjust the baseline of the interpolated values.
                Default is True. If False, the interpolated values are not adjusted.
            return_magnitude_stats (bool, optional): Whether to return magnitude statistics.
                If True, returns tuple (phase, stats_dict) where stats_dict contains
                'mean_magnitude' and 'min_magnitude'. Default is True.
                When True and magnitude_threshold=None, automatically uses mean * 0.1 as threshold.

        Returns:
            np.ndarray or tuple: Phase array in radians (same length as input signals).
                Phase is accumulated from differential phase differences.
                If return_magnitude_stats=True, returns tuple (phase, stats_dict).

        Notes:
            - Uses Numba-optimized functions for normalization, differential calculation, and accumulation.
            - NaN values in phase differences are replaced with 0.0 before accumulation.
            - The cross-product method is more robust to amplitude variations than atan2.
            - The output length matches input length (last differential value is duplicated).
            - Magnitude statistics are computed from the original magnitude before normalization.
            - Threshold filtering sets low-quality samples to NaN before interpolation.
            - If adjust_baseline is True, the baseline of the interpolated values is adjusted.

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5, -1.0])
            q = np.array([0.0, 0.866, 0.866, 0.0])
            # Default: returns tuple (phase, stats_dict)
            phase, _ = converter.calc_phase_iq_asin2(i, q)
            # Result: Accumulated phase from differential cross-products
            
            # With magnitude threshold and statistics
            phase, stats = converter.calc_phase_iq_asin2(
                i, q,
                magnitude_threshold=0.5,
                interpolate_nan=True,
                return_magnitude_stats=True
            )
            print(f"Mean magnitude: {stats['mean_magnitude']:.3f}")
            
            # To get only phase without statistics
            phase = converter.calc_phase_iq_asin2(i, q, return_magnitude_stats=False)
            ```
        """
        # 1. Normalize I and Q signals (numba-optimized)
        if isnorm:
            i_norm, q_norm, iq_mag = normalize_iq_signals_jit(i_signal, q_signal)
        else:
            i_norm = i_signal
            q_norm = q_signal
            iq_mag = np.sqrt(i_signal**2 + q_signal**2)

        # Compute magnitude statistics if requested
        stats_dict = {}
        if return_magnitude_stats or magnitude_threshold is not None:
            mean_mag, min_mag = compute_magnitude_stats_jit(iq_mag)
            stats_dict = {
                "mean_magnitude": mean_mag,
                "min_magnitude": min_mag,
            }
            # If return_magnitude_stats=True and magnitude_threshold=None, use mean * 0.1 as default
            if magnitude_threshold is None and return_magnitude_stats:
                magnitude_threshold = mean_mag * 0.1

        # 2. Calculate the differential phase (numba-optimized)
        phase_diff = calculate_differential_phase_jit(i_norm, q_norm)

        # 3. Apply magnitude threshold filtering if specified
        if magnitude_threshold is not None:
            # phase_diff has length len(i_signal) - 1, so use iq_mag[:-1] for threshold
            mask = apply_magnitude_threshold_jit(iq_mag[:-1], magnitude_threshold)
            phase_diff[~mask] = np.nan

        # 4. Interpolate NaN values if requested
        if interpolate_nan and np.any(np.isnan(phase_diff)):
            if interpolateNonFinite is None:
                logger.warning(
                    f"{log_tag('PHI2N', 'INTERP')} interpolateNonFinite not available. Skipping NaN interpolation."
                )
            else:
                # Map interpolation method to degree
                method_to_degree = {
                    "nearest": 0,
                    "linear": 1,
                    "quadratic": 2,
                    "cubic": 3,
                }
                degree = method_to_degree.get(interpolation_method, interpolation_degree)
                
                # Store original NaN positions before interpolation
                phase_diff_original = phase_diff.copy()
                
                try:
                    phase_diff_interp = interpolateNonFinite(
                        phase_diff,
                        xCoords=None,  # Uniform spacing
                        maxNonFiniteNeighbors=-1,  # No limit
                        degree=degree,
                        allowExtrapolation=False,
                        smooth=0,
                    )
                    
                    # Adjust baseline: subtract the last finite value before each NaN segment
                    if adjust_baseline:
                        if degree == 0:
                            # For nearest (degree=0), set interpolated values to 0 (equivalent to baseline adjustment)
                            nan_mask = np.isnan(phase_diff_original)
                            phase_diff_interp[nan_mask] = 0.0
                            phase_diff = phase_diff_interp
                        else:
                            # For degree >= 1, adjust interpolated values by baseline
                            phase_diff = adjust_interpolated_baseline_jit(phase_diff_interp, phase_diff_original)
                    else:
                        phase_diff = phase_diff_interp
                except Exception as e:
                    logger.warning(
                        f"{log_tag('PHI2N', 'INTERP')} Interpolation failed: {e}. Returning phase with NaN values."
                    )
        # Handle any potential NaNs, as in the MATLAB code
        elif interpolate_nan is False:
            phase_diff[np.isnan(phase_diff)] = 0.0

        # 5. Accumulate the phase differences (numba-optimized)
        phase_accum = accumulate_phase_diff_jit(phase_diff)

        # Match the sign from the MATLAB script
        if isflip:
            phase_accum *= -1
            
        if return_magnitude_stats:
            return phase_accum, stats_dict

        return phase_accum

    def calc_phase_iq(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        iscxprod: bool = False,
        isnorm: bool = True,
        isflip: bool = False,
        magnitude_threshold: float = None,
        interpolate_nan: bool = False,
        interpolation_method: str = "linear",
        interpolation_degree: int = 1,
        adjust_baseline: bool = True,
        return_magnitude_stats: bool = True,
    ) -> np.ndarray | tuple:
        """
        Calculate phase from I and Q signals (convenience method).

        This is a convenience method that delegates to either calc_phase_iq_atan2
        (default) or calc_phase_iq_asin2 (cross-product method) based on the iscxprod flag.

        Args:
            i_signal (np.ndarray): In-phase signal array (1D).
            q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.
            iscxprod (bool, optional): Whether to use the cross-product (asin2) method.
                If True, uses calc_phase_iq_asin2. If False (default), uses calc_phase_iq_atan2.
            isnorm (bool, optional): Whether to normalize the I and Q signals to unit circle.
                Default is True.
            isflip (bool, optional): Whether to flip the sign of the phase.
                Default is False.
            magnitude_threshold (float, optional): Minimum magnitude threshold for filtering.
                Values with magnitude below this threshold will be set to NaN.
                If None, no threshold filtering is applied. Default is None.
            interpolate_nan (bool, optional): Whether to interpolate NaN values in the phase array.
                Default is False. Uses interpolateNonFiniteValues if True.
            interpolation_method (str, optional): Interpolation method when interpolate_nan=True.
                Options: "nearest" (degree=0), "linear" (degree=1), "quadratic" (degree=2), "cubic" (degree=3).
                Default is "linear".
            interpolation_degree (int, optional): Spline degree for interpolation (0-3).
                Only used if interpolation_method is not one of the named methods.
                Default is 1.
            adjust_baseline (bool, optional): Whether to adjust the baseline of the interpolated values.
                Only applies when iscxprod=True and interpolate_nan=True.
                Default is True. If False, the interpolated values are not adjusted.
            return_magnitude_stats (bool, optional): Whether to return magnitude statistics.
                If True, returns tuple (phase, stats_dict) where stats_dict contains
                'mean_magnitude' and 'min_magnitude'. Default is True.

        Returns:
            np.ndarray or tuple: Phase array in radians (same length as input signals).
                If return_magnitude_stats=True, returns tuple (phase, stats_dict).

        Notes:
            - Default method (iscxprod=False) uses atan2, which is faster and simpler.
            - Cross-product method (iscxprod=True) is more robust to noise but slightly slower.
            - Both methods normalize signals and handle phase unwrapping/accumulation.
            - All optional parameters are passed through to the underlying method.

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5])
            q = np.array([0.0, 0.866, 0.866])
            
            # Default: atan2 method, returns tuple (phase, stats_dict)
            phase_atan2, _ = converter.calc_phase_iq(i, q, iscxprod=False)
            
            # Cross-product method with threshold and interpolation
            phase_asin2, stats = converter.calc_phase_iq(
                i, q,
                iscxprod=True,
                magnitude_threshold=0.5,
                interpolate_nan=True,
                adjust_baseline=True,
                return_magnitude_stats=True
            )
            
            # To get only phase without statistics
            phase = converter.calc_phase_iq(i, q, return_magnitude_stats=False)
            ```
        """
        if iscxprod:
            return self.calc_phase_iq_asin2(
                i_signal,
                q_signal,
                isnorm=isnorm,
                isflip=isflip,
                magnitude_threshold=magnitude_threshold,
                interpolate_nan=interpolate_nan,
                interpolation_method=interpolation_method,
                interpolation_degree=interpolation_degree,
                adjust_baseline=adjust_baseline,
                return_magnitude_stats=return_magnitude_stats,
            )
        else:
            return self.calc_phase_iq_atan2(
                i_signal,
                q_signal,
                isnorm=isnorm,
                isflip=isflip,
                magnitude_threshold=magnitude_threshold,
                interpolate_nan=interpolate_nan,
                interpolation_method=interpolation_method,
                interpolation_degree=interpolation_degree,
                return_magnitude_stats=return_magnitude_stats,
            )
    def calc_phase_cdm(
        self,
        ref_signal: np.ndarray,
        prob_signal: np.ndarray,
        fs: float,
        f_center: float,
        isbpf: bool = True,
        isconj: bool = False,
        islpf: bool = True,
        iszif: bool = False,
        isflip: bool = False,
        plot_filters: bool = False,
        magnitude_threshold: float = None,
        interpolate_nan: bool = True,
        interpolation_method: str = "nearest",
        interpolation_degree: int = 0,
        adjust_baseline: bool = True,
        return_magnitude_stats: bool = True,
    ) -> np.ndarray | tuple:
        """
        Calculate phase using Complex Demodulation (CDM) method.

        This method implements the CDM algorithm for phase extraction from interferometer
        signals. The process involves:
        1. Band-pass filtering (BPF) around the IF frequency
        2. Complex demodulation using Hilbert transform
        3. Low-pass filtering (LPF) to extract baseband phase
        4. Differential phase calculation and accumulation

        The CDM method is robust to amplitude variations and is used for recent experimental
        configurations (shot numbers >= 41542).

        Args:
            ref_signal (np.ndarray): Reference channel signal (1D array).
            prob_signal (np.ndarray): Probe channel signal (1D array), must have same length as ref_signal.
            fs (float): Sampling frequency in Hz.
            f_center (float): Center frequency of the IF signal in Hz (typically determined from STFT).
                Used to design BPF. Filter bandwidth depends on f_center:
                - f_center > 60 MHz: ±12.5 MHz stopband, ±7.5 MHz passband
                - f_center > 30 MHz: ±10.0 MHz stopband, ±5.0 MHz passband
                - f_center <= 30 MHz: ±7.0 MHz stopband, ±3.0 MHz passband
            isbpf (bool, optional): Whether to apply band-pass filter. Default is True.
            isconj (bool, optional): Whether to use conjugate of reference for demodulation.
                Default is False. Set to True for specific phase conventions.
            islpf (bool, optional): Whether to apply low-pass filter to demodulated signal.
                LPF passband: 0.5 MHz, stopband: 1.0 MHz. Default is True.
            iszif (bool, optional): Whether to use zero-IF demodulation method.
                If True, uses different phase extraction (np.angle). Default is False.
            isflip (bool, optional): Whether to flip the sign of the phase. Default is False.
            plot_filters (bool, optional): Whether to plot BPF and LPF frequency responses.
                Default is False.
            magnitude_threshold (float, optional): Minimum magnitude threshold for filtering.
                For iszif=False: Applied to demodulated signal magnitude (|demod_lpf|) before phase calculation.
                For iszif=True: Applied to demod_zif magnitude (|demod_zif|).
                Values with magnitude below this threshold will be set to NaN.
                If None and return_magnitude_stats=True, mean magnitude * 0.1 is used as default threshold.
                If None and return_magnitude_stats=False, no threshold filtering is applied. Default is None.
            interpolate_nan (bool, optional): Whether to interpolate NaN values in the phase array.
                Default is False. Uses interpolateNonFiniteValues if True.
                Applies to both differential method (iszif=False) and zero-IF method (iszif=True).
            interpolation_method (str, optional): Interpolation method when interpolate_nan=True.
                Options: "nearest" (degree=0), "linear" (degree=1), "quadratic" (degree=2), "cubic" (degree=3).
                Default is "linear".
            interpolation_degree (int, optional): Spline degree for interpolation (0-3).
                Only used if interpolation_method is not one of the named methods.
                Default is 1.
            adjust_baseline (bool, optional): Whether to adjust the baseline of the interpolated values.
                Only applies when interpolate_nan=True.
                Default is True. If False, the interpolated values are not adjusted.
            return_magnitude_stats (bool, optional): Whether to return magnitude statistics.
                If True, returns tuple (phase, stats_dict) where stats_dict contains
                'mean_magnitude' and 'min_magnitude' computed from demod_lpf (iszif=False) or
                demod_zif (iszif=True) magnitude. Default is False.

        Returns:
            np.ndarray or tuple: Phase array in radians (same length as input signals).
                Phase is accumulated from differential phase differences and baseline-corrected.
                If return_magnitude_stats=True, returns tuple (phase, stats_dict).

        Notes:
            - BPF and LPF are designed automatically based on f_center.
            - Uses scipy.signal.hilbert for complex demodulation (matches MATLAB behavior).
            - Baseline correction uses first 10000 samples (or 1000 if signal is shorter).
            - The method uses Numba-optimized functions for differential phase calculation.
            - If magnitude_threshold is specified, low-magnitude samples are set to NaN before interpolation.
            - If interpolate_nan=True and adjust_baseline=True, interpolated values are adjusted by baseline.
            - For iszif=True, magnitude threshold is applied to |demod_zif| (magnitude of demodulated signal).
            - If return_magnitude_stats=True and magnitude_threshold=None, mean magnitude * 0.1 is used as default threshold.
            - If magnitude_threshold is provided, it overrides the default threshold value.

        Examples:
            ```python
            converter = PhaseConverter()
            # Determine f_center from STFT analysis
            f_center = 15e6  # 15 MHz IF frequency
            
            # Calculate phase using CDM (default: returns tuple)
            phase, _ = converter.calc_phase_cdm(
                ref_signal, probe_signal,
                fs=50e6,
                f_center=f_center,
                isbpf=True,
                islpf=True
            )
            
            # With filter visualization
            phase, _ = converter.calc_phase_cdm(
                ref_signal, probe_signal,
                fs=50e6, f_center=f_center,
                plot_filters=True  # Shows BPF and LPF responses
            )
            
            # To get only phase without statistics
            phase = converter.calc_phase_cdm(
                ref_signal, probe_signal,
                fs=50e6, f_center=f_center,
                return_magnitude_stats=False
            )
            ```
        """
        # 1. Design filters based on f_center.
        # These values are from CDM_VEST_check.m
        if f_center > 60e6:
            fcl, f0l, f0r, fcr = (
                f_center - 12.5e6,
                f_center - 7.5e6,
                f_center + 7.5e6,
                f_center + 12.5e6,
            )
        elif f_center > 30e6:
            fcl, f0l, f0r, fcr = (
                f_center - 10.0e6,
                f_center - 5.0e6,
                f_center + 5.0e6,
                f_center + 10.0e6,
            )
        else:
            fcl, f0l, f0r, fcr = (
                f_center - 7.0e6,
                f_center - 3.0e6,
                f_center + 3.0e6,
                f_center + 7.0e6,
            )

        lpf_pass = 0.5e6
        lpf_stop = 1e6

        # Limit filter taps to prevent filtfilt issues with short signals
        # signal_length = len(ref_signal)
        # max_taps = min(signal_length // 4, 1000)  # Use at most 1/4 of signal length or 1000 taps

        bpf_coeffs = self._create_bpf(fcl, f0l, f0r, fcr, fs)
        lpf_coeffs = self._create_lpf(lpf_pass, lpf_stop, fs, max_taps=None)

        if plot_filters:
            self._plot_filter_response(
                bpf_coeffs, fs, title=f"BPF Response ({fcl / 1e6}-{fcr / 1e6} MHz)"
            )
            self._plot_filter_response(
                lpf_coeffs, fs, title=f"LPF Response ({lpf_pass / 1e6} MHz)"
            )

        # 2. Apply BPF if enabled
        ref_bpf = filtfilt(bpf_coeffs, 1, ref_signal) if isbpf else ref_signal
        prob_bpf = filtfilt(bpf_coeffs, 1, prob_signal) if isbpf else prob_signal

        # 3. Complex demodulation
        # Verified: MATLAB hilbert and scipy.signal.hilbert are identical (not conjugates)
        # MATLAB: x1_94 = hilbert(ref94); ych5 = x1_94.*ch5;
        # To match MATLAB: use hilbert() directly without conjugate
        ref_hilbert = hilbert(ref_bpf)
        # MATLAB equivalent: hilbert(ref_bpf) * prob_bpf
        # Use .conj() only if isconj=True (for specific phase convention needs)
        demod_signal = (
            ref_hilbert.conj() * prob_bpf if isconj else ref_hilbert * prob_bpf
        )

        # 4. Apply LPF to the demodulated signal
        demod_lpf = filtfilt(lpf_coeffs, 1, demod_signal) if islpf else demod_signal

        # Initialize stats_dict for magnitude statistics
        stats_dict = {}

        # 5. Calculate phase
        if not iszif:
            # The matlab script uses a differential method.
            re = np.real(demod_lpf)
            im = np.imag(demod_lpf)

            # Vectorized differential phase calculation (cross-product method)
            phase_diff = calculate_differential_phase_jit(re, im)

            # Compute magnitude statistics if requested (for differential method)
            demod_lpf_mag = np.abs(demod_lpf)
            if return_magnitude_stats or magnitude_threshold is not None:
                mean_mag, min_mag = compute_magnitude_stats_jit(demod_lpf_mag)
                stats_dict = {
                    "mean_magnitude": mean_mag,
                    "min_magnitude": min_mag,
                }
                # If return_magnitude_stats=True and magnitude_threshold=None, use mean * 0.1 as default
                if magnitude_threshold is None and return_magnitude_stats:
                    magnitude_threshold = mean_mag * 0.1

            # 6. Apply magnitude threshold filtering if specified (for differential method)
            if magnitude_threshold is not None:
                # phase_diff has length len(demod_lpf) - 1, so use demod_lpf_mag[:-1] for threshold
                mask = apply_magnitude_threshold_jit(demod_lpf_mag[:-1], magnitude_threshold)
                phase_diff[~mask] = np.nan

            # 7. Interpolate NaN values if requested (only for differential method)
            if interpolate_nan and np.any(np.isnan(phase_diff)):
                if interpolateNonFinite is None:
                    logger.warning(
                        f"{log_tag('PHI2N', 'INTERP')} interpolateNonFinite not available. Skipping NaN interpolation."
                    )
                else:
                    # Map interpolation method to degree
                    method_to_degree = {
                        "nearest": 0,
                        "linear": 1,
                        "quadratic": 2,
                        "cubic": 3,
                    }
                    degree = method_to_degree.get(interpolation_method, interpolation_degree)
                    
                    # Store original NaN positions before interpolation
                    phase_diff_original = phase_diff.copy()
                    
                    try:
                        phase_diff_interp = interpolateNonFinite(
                            phase_diff,
                            xCoords=None,  # Uniform spacing
                            maxNonFiniteNeighbors=-1,  # No limit
                            degree=degree,
                            allowExtrapolation=False,
                            smooth=0,
                        )
                        
                        # Adjust baseline: subtract the last finite value before each NaN segment
                        if adjust_baseline:
                            if degree == 0:
                                # For nearest (degree=0), set interpolated values to 0 (equivalent to baseline adjustment)
                                nan_mask = np.isnan(phase_diff_original)
                                phase_diff_interp[nan_mask] = 0.0
                                phase_diff = phase_diff_interp
                            else:
                                # For degree >= 1, adjust interpolated values by baseline
                                phase_diff = adjust_interpolated_baseline_jit(phase_diff_interp, phase_diff_original)
                        else:
                            phase_diff = phase_diff_interp
                    except Exception as e:
                        logger.warning(
                            f"{log_tag('PHI2N', 'INTERP')} Interpolation failed: {e}. Returning phase with NaN values."
                        )
            # Handle any potential NaNs, as in the MATLAB code
            elif interpolate_nan is False:
                phase_diff[np.isnan(phase_diff)] = 0.0

            # The first sample is phase_diff[0] at accumulate_phase_diff_jit, so no need to prepend a 0.
            phase_accum = accumulate_phase_diff_jit(phase_diff)
        else:
            demod_zif = ref_hilbert.conj() * hilbert(prob_bpf)
            phase_accum = np.angle(demod_zif)

            # Compute magnitude statistics if requested (for zero-IF method)
            demod_zif_mag = np.abs(demod_zif)
            if return_magnitude_stats or magnitude_threshold is not None:
                mean_mag, min_mag = compute_magnitude_stats_jit(demod_zif_mag)
                stats_dict = {
                    "mean_magnitude": mean_mag,
                    "min_magnitude": min_mag,
                }
                # If return_magnitude_stats=True and magnitude_threshold=None, use mean * 0.1 as default
                if magnitude_threshold is None and return_magnitude_stats:
                    magnitude_threshold = mean_mag * 0.1

            # 6. Apply magnitude threshold filtering if specified (for zero-IF method)
            if magnitude_threshold is not None:
                mask = apply_magnitude_threshold_jit(demod_zif_mag, magnitude_threshold)
                phase_accum[~mask] = np.nan

            # 7. Interpolate NaN values if requested (for zero-IF method)
            if interpolate_nan and np.any(np.isnan(phase_accum)):
                if interpolateNonFinite is None:
                    logger.warning(
                        f"{log_tag('PHI2N', 'INTERP')} interpolateNonFinite not available. Skipping NaN interpolation."
                    )
                else:
                    # Map interpolation method to degree
                    method_to_degree = {
                        "nearest": 0,
                        "linear": 1,
                        "quadratic": 2,
                        "cubic": 3,
                    }
                    degree = method_to_degree.get(interpolation_method, interpolation_degree)
                    
                    # Store original NaN positions before interpolation
                    phase_accum_original = phase_accum.copy()
                    
                    try:
                        phase_accum_interp = interpolateNonFinite(
                            phase_accum,
                            xCoords=None,  # Uniform spacing
                            maxNonFiniteNeighbors=-1,  # No limit
                            degree=degree,
                            allowExtrapolation=False,
                            smooth=0,
                        )
                        
                        # Adjust baseline: subtract the last finite value before each NaN segment
                        if adjust_baseline:
                            if degree == 0:
                                # For nearest (degree=0), set interpolated values to 0 (equivalent to baseline adjustment)
                                nan_mask = np.isnan(phase_accum_original)
                                phase_accum_interp[nan_mask] = 0.0
                                phase_accum = phase_accum_interp
                            else:
                                # For degree >= 1, adjust interpolated values by baseline
                                phase_accum = adjust_interpolated_baseline_jit(phase_accum_interp, phase_accum_original)
                        else:
                            phase_accum = phase_accum_interp
                    except Exception as e:
                        logger.warning(
                            f"{log_tag('PHI2N', 'INTERP')} Interpolation failed: {e}. Returning phase with NaN values."
                        )
            # Handle any potential NaNs, as in the MATLAB code
            elif interpolate_nan is False:
                phase_accum[np.isnan(phase_accum)] = 0.0

        # MATLAB uses first 10000 samples; use same or adapt based on signal length
        if len(phase_accum) > 10000:
            phase_accum -= np.mean(phase_accum[:10000])
        elif len(phase_accum) > 1000:
            phase_accum -= np.mean(phase_accum[:1000])

        if isflip:
            phase_accum *= -1

        if return_magnitude_stats:
            return phase_accum, stats_dict
        return phase_accum

    def calc_phase_fpga(
        self,
        ref_phase: np.ndarray,
        probe_phase: np.ndarray,
        time: np.ndarray,
        amp_signal: np.ndarray,
        isflip: bool = False,
    ) -> np.ndarray:
        """
        Calculate phase difference from FPGA-processed phase data.

        This method processes phase data that has already been extracted by FPGA hardware
        using the CORDIC algorithm. It simply computes the difference between probe and
        reference phases and applies baseline correction.

        This is a simplified version compared to the full MATLAB script (FPGAreadData_ver2.m).
        Future enhancements may include:
        - Moving average filtering of phase difference
        - Dynamic detection of plasma discharge window
        - Masking during low amplitude periods (beam diffraction)
        - Advanced pre/post-discharge offset correction

        Args:
            ref_phase (np.ndarray): Reference phase signal in radians (1D array).
            probe_phase (np.ndarray): Probe phase signal in radians (1D array),
                must have same length as ref_phase.
            time (np.ndarray): Time array in seconds (1D array). Currently used for logging,
                reserved for future windowing features.
            amp_signal (np.ndarray): Amplitude signal for the probe channel (1D array).
                Currently unused, reserved for future amplitude-based masking.

        Returns:
            np.ndarray: Processed phase difference in radians (same length as input signals).
                Phase difference is baseline-corrected using the first 1000 samples.

        Notes:
            - Phase difference: ?? = probe_phase - ref_phase
            - Baseline correction subtracts the mean of the first 1000 samples.
            - The FPGA method is used for shots 39302-41398.
            - Phase data from FPGA is already in radians (CORDIC output).

        Examples:
            ```python
            converter = PhaseConverter()
            # FPGA data: phases are already extracted
            ref_phase = df['CH0'].values  # Reference phase from FPGA
            probe_phase = df['CH1'].values  # Probe phase from FPGA
            time = df['TIME'].values
            
            # Calculate phase difference
            phase_diff = converter.calc_phase_fpga(
                ref_phase, probe_phase, time,
                amp_signal=df['CH17'].values  # Amplitude (for future use)
            )
            ```
        """
        phase_diff = probe_phase - ref_phase

        # Simple offset correction: subtract the mean of the initial part of the signal.
        if len(phase_diff) > 1000:
            # Using a window at the start of the signal for offset correction
            offset = np.mean(phase_diff[:1000])
            phase_diff -= offset

        if isflip:
            phase_diff *= -1

        return phase_diff

    def correct_baseline(
        self,
        density_df: pd.DataFrame,
        time_axis: np.ndarray,
        mode: str,
        shot_num: int = None,
        vest_data: pd.DataFrame = None,
        ip_column_name: str = None,
    ):
        """
        Correct baseline offset in density data.

        This method removes baseline offset from density measurements by subtracting
        the mean value in a specified time window. Two modes are supported:
        - 'ip' mode: Uses plasma current ramp-up time to define baseline window
        - 'trig' mode: Uses fixed time window (0.285-0.290 s, typical pre-trigger period)

        Args:
            density_df (pd.DataFrame): DataFrame with density data columns.
                Each column represents a probe channel's density measurements.
            time_axis (np.ndarray): Common time axis in seconds (1D array).
                Must have same length as density_df rows.
            mode (str): Baseline correction mode. Options:
                - 'ip': Use plasma current ramp-up to define baseline window (3-8 ms before ramp-up).
                - 'trig': Use fixed time window (0.285-0.290 s).
            shot_num (int, optional): Shot number for VEST data retrieval.
                Required for 'ip' mode. Default is None.
            vest_data (pd.DataFrame, optional): DataFrame containing VEST database data.
                Required for 'ip' mode. Must contain plasma current column.
                Default is None.
            ip_column_name (str, optional): Name of the plasma current column in vest_data.
                Required for 'ip' mode. Default is None.

        Returns:
            pd.DataFrame: Corrected density DataFrame with baseline offset removed.
                Same structure as input density_df.

        Raises:
            None (method returns original DataFrame with warnings if conditions not met).

        Notes:
            - 'ip' mode: Finds first time when Ip > 5 kA (or 5 A if data is in Amperes),
              then uses window 3-8 ms before ramp-up.
            - 'trig' mode: Uses fixed window [0.285, 0.290] seconds (typical pre-trigger period).
            - Baseline correction subtracts the mean value in the window from each column.
            - If no data is found in the baseline window, returns original DataFrame with warning.

        Examples:
            ```python
            converter = PhaseConverter()
            
            # Trigger-based baseline correction
            density_corrected = converter.correct_baseline(
                density_df, time_axis, mode='trig'
            )
            
            # IP-based baseline correction
            density_corrected = converter.correct_baseline(
                density_df, time_axis,
                mode='ip',
                shot_num=45821,
                vest_data=vest_df,
                ip_column_name='Ip'
            )
            ```
        """
        corrected_df = density_df.copy()

        if mode == "ip":
            if shot_num is None:
                logger.warning(
                    f"{log_tag('PHI2N', 'BLINE')} 'ip' baseline mode selected but shot_num not provided. Skipping."
                )
                return corrected_df

            if vest_data is None or vest_data.empty:
                logger.warning(
                    f"{log_tag('PHI2N', 'BLINE')} 'ip' baseline mode selected but VEST DB not available. Skipping."
                )
                return corrected_df

            if ip_column_name is None or ip_column_name not in vest_data.columns:
                logger.warning(
                    f"{log_tag('PHI2N', 'BLINE')} Plasma current column '{ip_column_name}' not in VEST DB. Skipping."
                )
                return corrected_df

            ip_data = vest_data[ip_column_name]
            # Find the ramp-up point (first time Ip > 5kA)
            # Note: Assuming Ip is in Amperes. If in kA, this threshold should be 5.
            ip_threshold = 5e3 if np.nanmax(ip_data) > 1000 else 5

            try:
                ramp_up_idxs = np.where(ip_data > ip_threshold)[0]
                if len(ramp_up_idxs) == 0:
                    raise IndexError("Threshold not exceeded")

                ramp_up_idx = ramp_up_idxs[0]
                t_rampup = ip_data.index[ramp_up_idx]

                # Define baseline window: 3 to 8 ms before ramp-up
                t_start = t_rampup - 8e-3
                t_end = t_rampup - 3e-3

            except IndexError:
                logger.warning(
                    f"{log_tag('PHI2N', 'BLINE')} Plasma current never exceeded threshold ({ip_threshold})."
                )
                logger.warning(
                    f"{log_tag('PHI2N', 'BLINE')} Cannot determine ramp-up for 'ip' baseline. Skipping."
                )
                return corrected_df

        elif mode == "trig":
            t_start, t_end = 0.285, 0.290

        else:
            logger.warning(
                f"{log_tag('PHI2N', 'BLINE')} Invalid baseline mode '{mode}'. Skipping."
            )
            return corrected_df

        # Find indices for the baseline window
        baseline_idxs = np.where((time_axis >= t_start) & (time_axis <= t_end))[0]

        if len(baseline_idxs) == 0:
            logger.warning(
                f"{log_tag('PHI2N', 'BLINE')} No data found in the baseline window [{t_start:.4f}s, {t_end:.4f}s]. Skipping."
            )
            return corrected_df

        logger.info(
            f"{log_tag('PHI2N', 'BLINE')} Correcting baseline using window [{t_start:.4f}s, {t_end:.4f}s] ({len(baseline_idxs)} points)."
        )
        for col in corrected_df.columns:
            baseline_mean = corrected_df[col].iloc[baseline_idxs].mean()
            corrected_df[col] -= baseline_mean
            logger.info(
                f"{log_tag('PHI2N', 'BLINE')} Column '{col}': Removed baseline of {baseline_mean:.2e}"
            )

        return corrected_df
