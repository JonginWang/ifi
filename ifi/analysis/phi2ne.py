#!/usr/bin/env python3
"""
Phase to Density Conversion
===========================

This module provides functions and classes for converting interferometer phase measurements
to line-integrated electron density. It supports multiple phase extraction methods (IQ, CDM, FPGA)
and includes Numba-optimized core calculations for high performance.

Key Features:
    - Multiple phase extraction methods: IQ (atan2/asin2), CDM (Complex Demodulation), FPGA
    - Numba-optimized core calculations for performance
    - Automatic parameter detection based on shot number and filename
    - FIR filter design for signal processing (LPF, BPF)
    - Baseline correction using plasma current or trigger time

Functions:
    Numba-optimized (internal):
        _normalize_iq_signals: Normalize I and Q signals to unit circle.
        _calculate_differential_phase: Calculate differential phase using cross-product method.
        _accumulate_phase_diff: Accumulate phase differences for phase reconstruction.
        _phase_to_density: Core phase-to-density conversion calculation.
    
    Public:
        get_interferometry_params: Get interferometry parameters based on shot number and filename.

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

Examples:
    ```python
    from ifi.analysis.phi2ne import PhaseConverter, get_interferometry_params
    
    # Get parameters for a shot
    params = get_interferometry_params(45821, "45821_056.csv")
    print(f"Method: {params['method']}, Frequency: {params['freq_ghz']} GHz")
    
    # Convert phase to density
    converter = PhaseConverter()
    phase = np.array([0.1, 0.2, 0.3])  # radians
    density = converter.phase_to_density(phase, freq_hz=94e9, n_path=2)
    
    # Calculate phase from I/Q signals
    i_signal = np.array([1.0, 0.5, -0.5])
    q_signal = np.array([0.0, 0.866, 0.866])
    phase = converter.calc_phase_iq(i_signal, q_signal)
    ```
"""

from pathlib import Path
import configparser
from typing import Dict, Any
import numpy as np
import numba
import pandas as pd
from scipy import constants
from scipy.signal import hilbert, remez, filtfilt, freqz
import matplotlib.pyplot as plt
try:
    from .functions.remezord import remezord  # remezord is from eegpy package, translated from MATLAB
    from .plots import plot_response
    from ..utils.common import LogManager, log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.analysis.functions.remezord import remezord
    from ifi.analysis.plots import plot_response
    from ifi.utils.common import LogManager, log_tag

logger = LogManager().get_logger(__name__)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _normalize_iq_signals(i_signal, q_signal):
    """
    Normalize I and Q signals to unit circle (Numba-optimized).

    This function normalizes I/Q signals by dividing by their magnitude, ensuring
    the resulting signals lie on the unit circle. This is useful for phase extraction
    as it removes amplitude variations while preserving phase information.

    Args:
        i_signal (np.ndarray): In-phase signal array (1D).
        q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - i_norm (np.ndarray): Normalized I signal (unit circle).
            - q_norm (np.ndarray): Normalized Q signal (unit circle).

    Notes:
        - Zero magnitude values are replaced with 1.0 to avoid division by zero.
        - The function is JIT-compiled with Numba for performance.
        - Input arrays must have the same length.

    Examples:
        ```python
        i = np.array([1.0, 0.5, 0.0])
        q = np.array([0.0, 0.866, 1.0])
        i_norm, q_norm = _normalize_iq_signals(i, q)
        # Result: i_norm and q_norm have unit magnitude
        ```
    """
    iq_mag = np.sqrt(i_signal**2 + q_signal**2)
    # Avoid division by zero
    for i in range(len(iq_mag)):
        if iq_mag[i] == 0:
            iq_mag[i] = 1.0

    i_norm = i_signal / iq_mag
    q_norm = q_signal / iq_mag
    return i_norm, q_norm


@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_differential_phase(i_norm, q_norm):
    """
    Calculate differential phase using cross-product method (Numba-optimized).

    This function computes the phase difference between consecutive samples using
    the cross-product method: Δφ = 2 * arcsin((I[n] * Q[n+1] - Q[n] * I[n+1]) / (|I[n],Q[n]| * |I[n+1],Q[n+1]|))

    The cross-product method is more robust to noise than direct phase subtraction
    and is used in the asin2 phase extraction algorithm.

    Args:
        i_norm (np.ndarray): Normalized I signal (unit circle, 1D array).
        q_norm (np.ndarray): Normalized Q signal (unit circle, 1D array), must have same length as i_norm.

    Returns:
        np.ndarray: Differential phase array (length = len(i_norm) - 1) in radians.
            Each element represents the phase difference between consecutive samples.

    Notes:
        - The output array is one element shorter than the input (n-1 differences for n samples).
        - The arcsin argument is clipped to [-1, 1] to handle floating-point errors.
        - Zero denominators are replaced with 1e-12 to avoid division by zero.
        - The function is JIT-compiled with Numba for performance.

    Examples:
        ```python
        i_norm = np.array([1.0, 0.707, 0.0])
        q_norm = np.array([0.0, 0.707, 1.0])
        phase_diff = _calculate_differential_phase(i_norm, q_norm)
        # Result: Array of phase differences between consecutive samples
        ```
    """
    # Vectorized differential phase calculation (cross-product method)
    denominator = np.sqrt(i_norm[:-1] ** 2 + q_norm[:-1] ** 2) * np.sqrt(
        i_norm[1:] ** 2 + q_norm[1:] ** 2
    )
    denominator[denominator == 0] = 1e-12  # Avoid division by zero

    # Clip argument to arcsin to handle potential floating point errors
    ratio = np.clip(
        (i_norm[:-1] * q_norm[1:] - q_norm[:-1] * i_norm[1:]) / denominator, -1.0, 1.0
    )
    phase_diff = 2.0 * np.arcsin(ratio)

    return phase_diff


@numba.jit(nopython=True, cache=True, fastmath=True)
def _accumulate_phase_diff(phase_diff):
    """
    Accumulate phase differences to reconstruct absolute phase (Numba-optimized).

    This function performs cumulative summation of phase differences to reconstruct
    the absolute phase from differential phase measurements. The CDM and asin2 methods
    produce phase differences that are one element shorter than the original signal,
    so the last value is duplicated to match the original signal length.

    Args:
        phase_diff (np.ndarray): Differential phase array (delta_phi) in radians.
            Length = n - 1 for an original signal of length n.

    Returns:
        np.ndarray: Accumulated phase array in radians, length = n (original signal length).
            The last value is duplicated from the cumulative sum to match the original length.

    Notes:
        - Uses cumulative sum: phase_accum[i] = sum(phase_diff[0:i+1])
        - The last value is duplicated: phase_accum[n-1] = phase_accum[n-2]
        - This ensures the output length matches the original signal length.
        - The function is JIT-compiled with Numba for performance.

    Examples:
        ```python
        phase_diff = np.array([0.1, 0.2, -0.1])  # 3 differences
        phase_accum = _accumulate_phase_diff(phase_diff)
        # Result: [0.1, 0.3, 0.2, 0.2]  # 4 values (last duplicated)
        ```
    """
    phase_accum = np.cumsum(phase_diff)
    # Duplicate the last value to match the original signal length
    phase_accum = np.append(phase_accum, phase_accum[-1])
    return phase_accum


@numba.jit(nopython=True, cache=True, fastmath=True)
def _phase_to_density(phase, freq, c, m_e, eps0, qe, n_path):
    """
    Convert phase to line-integrated electron density (Numba-optimized core calculation).

    This function implements the core physics formula for converting interferometer phase
    to line-integrated electron density. The formula is based on plasma dispersion relation
    and interferometry principles.

    Formula:
        n_c = m_e * eps0 * (2π * freq)² / qe²  (critical density)
        ∫n_e dl = (c * n_c / (π * freq)) * phase / n_path

    Args:
        phase (np.ndarray): Phase array in radians (1D).
        freq (float): Interferometer frequency in Hz.
        c (float): Speed of light in m/s.
        m_e (float): Electron mass in kg.
        eps0 (float): Permittivity of free space in F/m.
        qe (float): Electron charge in C.
        n_path (int): Number of interferometer passes (double-pass = 2, single-pass = 1).
            If n_path <= 0, division by n_path is skipped.

    Returns:
        np.ndarray: Line-integrated electron density in m^-2 (same length as phase).

    Notes:
        - The function is JIT-compiled with Numba for performance.
        - Physical constants should be from scipy.constants for accuracy.
        - The formula assumes small phase approximation (valid for typical plasma densities).
        - n_path accounts for multiple passes through the plasma (e.g., double-pass interferometer).

    Examples:
        ```python
        phase = np.array([0.1, 0.2, 0.3])  # radians
        from scipy import constants
        density = _phase_to_density(
            phase,
            freq=94e9,  # 94 GHz
            c=constants.c,
            m_e=constants.m_e,
            eps0=constants.epsilon_0,
            qe=constants.e,
            n_path=2  # double-pass
        )
        # Result: Line-integrated density in m^-2
        ```
    """
    # Calculate critical density
    n_c = m_e * eps0 * (2 * np.pi * freq)**2 / (qe**2)

    # Calculate line-integrated density
    nedl = (c * n_c / (np.pi * freq)) * phase

    if n_path > 0:
        nedl /= n_path

    return nedl


# The formula: \int {n_{e} dl}
#              = ( 2 * c * n_c / (2 * pi * freqRF) ) * phi (radian) / n_path
# The phase (degree) : phi * (2*pi/360)
# The phase (radians): phi
# Nedl = (2 * c * n_c / omega) * phase / n_path
# Let's re-check the formula.
# d(phi) = (omega / c) * (k_plasam - k) * dl
#        = (omega / c) * (N - 1) * dl
#        = (omega / c) * (sqrt(1 - (omega_p^2 / omega^2)) - 1) * dl
#        = (omega / c) * (sqrt(1 - (n_e / n_c)) - 1) * dl
#
# omega_p^2 = n_e * qe^2 / (m_e * eps0)
# n_c = (m_e * eps0) * omega^2 / qe^2
#
# 1) d(phi) = (omega / c) * (sqrt(1 - (omega_p^2 / omega^2)) - 1) * dl
#         --> (omega / c) * (omega_p^2 / omega^2) / 2 * dl
#           = omega_p^2 / (2 * c * omega) * dl
#           = n_e * qe^2 / (2 * c * omega * m_e * eps0) * dl
#           = n_e * (qe^2 / (omega * m_e * eps0)) / (2 * c) * dl
# -->  nedl = d(phi) * (2 * c * n_c / omega)
#           = d(phi) * (2 * c * (m_e * eps0) * omega^2 / qe^2 / omega)
#           = d(phi) * (2 * c * (m_e * eps0) * omega / qe^2)
#           = d(phi) * (2 * c * (m_e * eps0) * 2 * pi * freq / qe^2
# 2) d(phi) = (omega / c) * (sqrt(1 - (n_e / n_c)) - 1) * dl
#         --> (omega / c) * (n_e / n_c) / 2 * dl
#           = omega / (2 * c * n_c) * n_e * dl
# -->  nedl = d(phi) * (2 * c * n_c / omega)


def get_interferometry_params(
    shot_num: int, filename: str, config_path: str = None
) -> Dict:
    """
    Get interferometry parameters based on shot number and filename pattern.

    This function automatically determines the appropriate interferometry method and
    parameters based on the shot number range and filename pattern. It supports
    multiple experimental configurations that have changed over time.

    Args:
        shot_num (int): Shot number (typically 5-digit integer, e.g., 45821).
        filename (str): Filename or file path (e.g., "45821_056.csv", "45821_ALL.csv").
            The filename pattern determines frequency and channel mapping:
            - "_ALL" → 280 GHz, CDM method
            - "_0XX" (e.g., "_056", "_789") → 94 GHz, CDM method, reference channel
            - Other patterns → 94 GHz, CDM method, probe channels
        config_path (str, optional): Path to configuration file (if_config.ini).
            If None, uses default path: "ifi/analysis/if_config.ini".

    Returns:
        Dict: Dictionary containing interferometry parameters:
            - method (str): Phase extraction method ("IQ", "FPGA", "CDM", or "unknown").
            - freq (float): Frequency in Hz (for calculations).
            - freq_ghz (float): Frequency in GHz (for display).
            - ref_col (str or None): Reference channel column name (e.g., "CH0").
            - probe_cols (list): List of probe channel column names (e.g., ["CH1", "CH2"]).
            - n_path (int): Number of interferometer passes.
            - amp_ref_col (str, optional): Amplitude reference channel (FPGA method only).
            - amp_probe_cols (list, optional): Amplitude probe channels (FPGA method only).

    Shot Number Ranges:
        - 30000-39265: IQ method (94 GHz)
            - Uses I/Q signal pairs directly
            - probe_cols: [("CH0", "CH1")] (tuple format for I/Q pairs)
        - 39302-41398: FPGA method (94 GHz)
            - Uses pre-processed phase from FPGA (CORDIC algorithm)
            - Includes amplitude channels (CH16-CH23)
            - ref_col: "CH0", probe_cols: ["CH1", "CH2", "CH3", "CH4", "CH5"]
        - 41542+: CDM method (94 GHz or 280 GHz based on filename)
            - "_ALL" → 280 GHz, ref_col: "CH0", probe_cols: ["CH1"]
            - "_0XX" → 94 GHz, ref_col: "CH0", probe_cols: ["CH1", "CH2"]
            - Other → 94 GHz, ref_col: "", probe_cols: ["CH0", "CH1", "CH2"]
        - Other: Returns "unknown" method with default 94 GHz parameters

    Raises:
        FileNotFoundError: If config file is not found.
        configparser.Error: If config file cannot be parsed.

    Notes:
        - Frequency values are read from config file and can be adjusted there.
        - The function supports flexible frequency detection (future enhancement for
          non-standard frequencies like 93-95 GHz range mapping to 94 GHz).
        - Channel mappings are based on experimental setup and may vary by shot range.

    Examples:
        ```python
        # CDM method for recent shots
        params = get_interferometry_params(45821, "45821_056.csv")
        # Returns: {"method": "CDM", "freq": 94e9, "freq_ghz": 94.0, ...}
        
        # 280 GHz data
        params = get_interferometry_params(46032, "46032_ALL.csv")
        # Returns: {"method": "CDM", "freq": 280e9, "freq_ghz": 280.0, ...}
        
        # FPGA method for older shots
        params = get_interferometry_params(40000, "40000.csv")
        # Returns: {"method": "FPGA", "freq": 94e9, ...}
        
        # IQ method for oldest shots
        params = get_interferometry_params(35000, "35000.csv")
        # Returns: {"method": "IQ", "freq": 94e9, ...}
        ```
    """
    # Extract frequency from filename
    basename = Path(filename).name

    # Default parameters
    params_default = {"method": "CDM", "freq_ghz": 94.0}  # noqa: F841

    # Load config if available
    if config_path is None:
        config_path = Path(__file__).parent / "if_config.ini"

    config = configparser.ConfigParser()

    config.read(config_path)

    # Extract frequency values from config (keep as Hz for calculations, also provide GHz)
    freq_94hz = float(config.get("94GHz", "freq"))  # Keep in Hz
    freq_280hz = float(config.get("280GHz", "freq"))  # Keep in Hz
    freq_94ghz = freq_94hz / 1e9  # Convert to GHz for display
    freq_280ghz = freq_280hz / 1e9  # Convert to GHz for display
    n_path_94ghz = int(config.get("94GHz", "n_path"))
    n_path_280ghz = int(config.get("280GHz", "n_path"))
    
    # Flexible frequency detection: use ranges instead of exact values
    # 93-95 GHz range → use 94GHz flag
    # 275-285 GHz range → use 280GHz flag
    # freq_94ghz_min = freq_94ghz - 1.0  # 93 GHz
    # freq_94ghz_max = freq_94ghz + 1.0  # 95 GHz
    # freq_280ghz_min = freq_280ghz - 5.0  # 275 GHz
    # freq_280ghz_max = freq_280ghz + 5.0  # 285 GHz

    # Rule 3: Shots 41542 and above
    if shot_num >= 41542:
        if "_ALL" in basename:
            # <shot_num>_ALL.csv: 280GHz / CDM method
            # "CH0" in file containing "_0" or "_ALL" is "ref. signal"
            return {
                "method": "CDM",
                "freq": freq_280hz,  # Hz for calculations
                "freq_ghz": freq_280ghz,  # GHz for display
                "ref_col": "CH0",
                "probe_cols": ["CH1"],
                "n_path": n_path_280ghz,
            }
        elif "_0" in basename:
            # <shot_num>_0XX.csv: 94GHz / CDM method
            # "CH0" in file containing "_0" or "_ALL" is "ref. signal"
            return {
                "method": "CDM",
                "freq": freq_94hz,  # Hz for calculations
                "freq_ghz": freq_94ghz,  # GHz for display
                "ref_col": "CH0",
                "probe_cols": [
                    "CH1",
                    "CH2",
                ],  # Mapping for channels 5, 6 (corrected by user)
                "n_path": n_path_94ghz,
            }
        else:
            # <shot_num>_XXX.csv: 94GHz / CDM method
            # Other probe channels
            return {
                "method": "CDM",
                "freq": freq_94hz,  # Hz for calculations
                "freq_ghz": freq_94ghz,  # GHz for display
                "ref_col": "",
                "probe_cols": ["CH0", "CH1", "CH2"],  # Mapping for channels 7-9
                "n_path": n_path_94ghz,
            }

    # Rule 2: Shots 39302–41398
    elif 39302 <= shot_num <= 41398:
        # 94GHz / FPGA method
        # The first 8 channels: phase ref(CH0) and probes (CH1-CH7) [rad]
        # The second 8 channels: nothing
        # The last 8 channels: amplitude of ref(CH16) and probes(CH17-CH23) [V]
        return {
            "method": "FPGA",
            "freq": freq_94hz,  # Hz for calculations
            "freq_ghz": freq_94ghz,  # GHz for display
            "ref_col": "CH0",
            "probe_cols": [
                "CH1",
                "CH2",
                "CH3",
                "CH4",
                "CH5",
            ],  # ch. 5-9 of 94G interferometer
            "amp_ref_col": "CH16",
            "amp_probe_cols": [
                "CH17",
                "CH18",
                "CH19",
                "CH20",
                "CH21",
            ],  # ch. 5-9 of 94G interferometer
            "n_path": n_path_94ghz,
        }

    # Rule 1: Shots 30000–39265 (modified range as per user's file)
    elif 30000 <= shot_num <= 39265:
        # 94GHz / IQ method
        # Assuming the two columns for I and Q are named 'CH0' and 'CH1' for simplicity.
        return {
            "method": "IQ",
            "freq": freq_94hz,  # Hz for calculations
            "freq_ghz": freq_94ghz,  # GHz for display
            "ref_col": None,  # IQ method does not use a separate reference signal from another channel
            "probe_cols": [("CH0", "CH1")],
            "n_path": n_path_94ghz,
        }

    # Default case for shots outside defined ranges
    else:
        return {
            "method": "unknown",
            "freq": freq_94hz,  # Hz for calculations, default to 94GHz if unknown
            "freq_ghz": freq_94ghz,  # GHz for display
            "ref_col": None,
            "probe_cols": [],
            "n_path": n_path_94ghz,
        }


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
        phase = converter.calc_phase_iq(i_signal, q_signal, iscxprod=False)
        
        # Convert phase to density
        density = converter.phase_to_density(
            phase,
            freq_hz=params['freq'],
            n_path=params['n_path']
        )
        
        # CDM method (requires reference and probe signals)
        ref_signal = np.sin(2 * np.pi * 15e6 * t)
        probe_signal = np.sin(2 * np.pi * 15e6 * t + phase)
        phase_cdm = converter.calc_phase_cdm(
            ref_signal, probe_signal,
            fs=50e6, f_center=15e6
        )
        
        # Correct baseline
        density_corrected = converter.correct_baseline(
            density_df, time_axis, mode='trig'
        )
        ```
    """

    def __init__(self, config_path: str = "ifi/analysis/if_config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.constants = {}
        if self.config.has_section("constants"):
            for key, name in self.config["constants"].items():
                if hasattr(constants, name):
                    self.constants[key] = getattr(constants, name)
                else:
                    raise ValueError(f"Constant '{name}' not found in scipy.constants")

    def get_params(self, freq_ghz: int) -> Dict[str, Any]:
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
        if not self.config.has_section(section):
            raise ValueError(
                f"Configuration for {freq_ghz}GHz not found in config file."
            )

        params = dict(self.config[section])
        params["freq"] = float(params["freq"])
        params["n_ch"] = int(params["n_ch"])
        params["n_path"] = self.config.getint(section, "n_path")
        return params

    def get_analysis_params(self, shot_num: int, filename: str) -> Dict:
        """
        Get interferometry analysis parameters based on shot number and filename.

        This method wraps the standalone `get_interferometry_params` function and uses
        the class's configuration path. It provides better integration with other
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
            - This method uses the class's config_path (default: "ifi/analysis/if_config.ini").
            - For detailed shot number ranges and filename patterns, see `get_interferometry_params`.

        Examples:
            ```python
            converter = PhaseConverter()
            params = converter.get_analysis_params(45821, "45821_056.csv")
            # Use params with phase_to_density
            density = converter.phase_to_density(phase, analysis_params=params)
            ```
        """
        # Use the standalone function but with our config path
        params = get_interferometry_params(shot_num, filename, config_path=None)
        return params

    def phase_to_density(
        self,
        phase: np.ndarray,
        freq_hz: float = None,
        n_path: int = None,
        analysis_params: Dict = None,
        wavelength: float = None,
    ) -> np.ndarray:
        """
        Convert phase (in radians) to line-integrated electron density (m^-2).

        This method implements the plasma interferometry formula to convert phase measurements
        to line-integrated electron density. The conversion uses physical constants and
        Numba-optimized core calculations for performance.

        Formula:
            ∫n_e dl = (c * n_c / (π * freq)) * phase / n_path
            where n_c = m_e * eps0 * (2π * freq)² / qe² (critical density)

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
        return _phase_to_density(phase, freq, c, m_e, eps0, qe, passes)

    def calc_phase_iq_atan2(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        isnorm: bool = True,
        isflip: bool = False,
    ) -> np.ndarray:
        """
        Calculate phase from I and Q signals using atan2 method.

        This method normalizes I/Q signals to the unit circle and calculates the phase
        using arctangent. The phase is unwrapped to handle 2π discontinuities and
        calibrated to start at zero.

        Args:
            i_signal (np.ndarray): In-phase signal array (1D).
            q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.
            isnorm (bool, optional): Whether to normalize I/Q signals to unit circle.
                Default is True. If False, signals are used as-is (must already be normalized).
            isflip (bool, optional): Whether to flip the sign of the phase.
                Default is False. Useful for correcting phase convention.

        Returns:
            np.ndarray: Phase array in radians (same length as input signals).
                Phase is unwrapped and calibrated to start at 0.

        Notes:
            - Normalization uses Numba-optimized function for performance.
            - Phase unwrapping handles 2π jumps automatically.
            - The first phase value is always 0 (calibrated).

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5, -1.0])
            q = np.array([0.0, 0.866, 0.866, 0.0])
            phase = converter.calc_phase_iq_atan2(i, q)
            # Result: [0, π/3, 2π/3, π] (approximately)
            ```
        """
        # 1. Normalize I and Q signals (numba-optimized)
        if isnorm:
            i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)
        else:
            i_norm = i_signal
            q_norm = q_signal

        # 2. Calculate phase by arctan (direct calculation)
        phase = np.unwrap(np.arctan2(q_norm, i_norm))
        phase -= phase[0]  # Calibrate to start at 0
        if isflip:
            phase *= -1
        return phase

    def calc_phase_iq_asin2(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        isnorm: bool = True,
        isflip: bool = False,
    ) -> np.ndarray:
        """
        Calculate phase from I and Q signals using asin2 (cross-product) method.

        This method uses a differential cross-product approach to calculate phase,
        which is more robust to noise than direct atan2. It computes phase differences
        between consecutive samples and accumulates them to reconstruct absolute phase.

        The algorithm is based on MATLAB's IQPostprocessing script and uses:
        1. I/Q normalization to unit circle
        2. Differential phase calculation: Δφ = 2 * arcsin(cross_product / magnitude_product)
        3. Phase accumulation via cumulative sum

        Args:
            i_signal (np.ndarray): In-phase signal array (1D).
            q_signal (np.ndarray): Quadrature signal array (1D), must have same length as i_signal.
            isnorm (bool, optional): Whether to normalize I/Q signals to unit circle.
                Default is True. If False, signals are used as-is (must already be normalized).
            isflip (bool, optional): Whether to flip the sign of the accumulated phase.
                Default is False. Useful for correcting phase convention.

        Returns:
            np.ndarray: Phase array in radians (same length as input signals).
                Phase is accumulated from differential phase differences.

        Notes:
            - Uses Numba-optimized functions for normalization, differential calculation, and accumulation.
            - NaN values in phase differences are replaced with 0.0.
            - The cross-product method is more robust to amplitude variations than atan2.
            - The output length matches input length (last differential value is duplicated).

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5, -1.0])
            q = np.array([0.0, 0.866, 0.866, 0.0])
            phase = converter.calc_phase_iq_asin2(i, q)
            # Result: Accumulated phase from differential cross-products
            ```
        """
        # 1. Normalize I and Q signals (numba-optimized)
        if isnorm:
            i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)
        else:
            i_norm = i_signal
            q_norm = q_signal

        # 2. Calculate the differential phase (numba-optimized)
        phase_diff = _calculate_differential_phase(i_norm, q_norm)

        # Handle any potential NaNs, as in the MATLAB code
        phase_diff[np.isnan(phase_diff)] = 0.0

        # 3. Accumulate the phase differences (numba-optimized)
        phase_accum = _accumulate_phase_diff(phase_diff)

        # Match the sign from the MATLAB script
        if isflip:
            phase_accum *= -1

        return phase_accum

    def calc_phase_iq(
        self,
        i_signal: np.ndarray,
        q_signal: np.ndarray,
        iscxprod: bool = False,
        isnorm: bool = True,
        isflip: bool = False,
    ) -> np.ndarray:
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

        Returns:
            np.ndarray: Phase array in radians (same length as input signals).

        Notes:
            - Default method (iscxprod=False) uses atan2, which is faster and simpler.
            - Cross-product method (iscxprod=True) is more robust to noise but slightly slower.
            - Both methods normalize signals and handle phase unwrapping/accumulation.

        Examples:
            ```python
            converter = PhaseConverter()
            i = np.array([1.0, 0.5, -0.5])
            q = np.array([0.0, 0.866, 0.866])
            
            # Default: atan2 method
            phase_atan2 = converter.calc_phase_iq(i, q, iscxprod=False)
            
            # Cross-product method
            phase_asin2 = converter.calc_phase_iq(i, q, iscxprod=True)
            ```
        """
        if iscxprod:
            return self.calc_phase_iq_asin2(
                i_signal, q_signal, isnorm=isnorm, isflip=isflip
            )
        else:
            return self.calc_phase_iq_atan2(
                i_signal, q_signal, isnorm=isnorm, isflip=isflip
            )

    def _create_lpf(self, f_pass, f_stop, fs, approx=False, max_taps=None):
        """
        Create a low-pass FIR filter using the remez (Parks-McClellan) algorithm.

        This method designs an optimal FIR filter with specified passband and stopband
        frequencies. The filter design uses remezord to determine the optimal number
        of taps and remez to generate the filter coefficients.

        Args:
            f_pass (float): Passband edge frequency in Hz.
            f_stop (float): Stopband edge frequency in Hz. Must be > f_pass.
            fs (float): Sampling frequency in Hz.
            approx (bool, optional): Whether to use approximate tap count instead of remezord.
                Default is False. If True, uses rule-of-thumb: N ≈ 4 / (transition_width_normalized).
            max_taps (int, optional): Maximum number of filter taps to prevent filtfilt issues
                with short signals. If None, no limit is applied. Default is None.

        Returns:
            np.ndarray: FIR filter coefficients (taps) for use with scipy.signal.filtfilt.

        Notes:
            - Filter specifications: ~0.5 dB passband ripple, -80 dB stopband attenuation.
            - The number of taps is always odd (required by remez).
            - If approx=True, uses simplified tap count estimation.
            - Filter coefficients are designed for use with filtfilt (zero-phase filtering).

        Examples:
            ```python
            converter = PhaseConverter()
            # Create LPF: passband up to 0.5 MHz, stopband above 1 MHz
            lpf_taps = converter._create_lpf(
                f_pass=0.5e6,
                f_stop=1e6,
                fs=50e6
            )
            # Apply filter
            from scipy.signal import filtfilt
            filtered_signal = filtfilt(lpf_taps, 1, signal)
            ```
        """
        nyq = 0.5 * fs
        # dens from matlab is 20, which is the weight.
        # The transition width is f_stop - f_pass.
        # Let's assume a reasonable ripple.
        # Dpass = 0.0575, Dstop = 0.0001 from LPF.m
        # This translates to about 0.5 dB ripple in passband and -80 dB attenuation in stopband.
        # numtaps, bands, desired, weight = remezord([f_pass, f_stop], [1, 0], [0.0575, 0.0001], Hz=fs)
        # remezord is not in scipy, so we have to estimate the number of taps.
        # A common rule of thumb: N = 4 / (transition_width_normalized)
        numtaps_approx = int(4 / ((f_stop - f_pass) / nyq))
        if numtaps_approx % 2 == 0:
            numtaps_approx += 1

        freqs = [f_pass, f_stop]
        amps = [1, 0]
        rip_pass = 0.057501127785  # pass-band ripple (linear)
        rip_stop = 1e-4  # stop-band ripple (linear)
        rips = [rip_pass, rip_stop]

        [numtaps, bands, amps, weight] = remezord(
            freqs, amps, rips, Hz=fs, alg="herrmann"
        )

        # Limit the number of taps to prevent filtfilt issues
        if max_taps is not None:
            numtaps = min(numtaps, max_taps)
            if numtaps % 2 == 0:
                numtaps += 1

        # compare the numtaps from remezord and approximate one
        if abs(numtaps - numtaps_approx) > 50 and not approx:  # Allow some tolerance
            logger.info(
                f"{log_tag('PHI2N', 'LPF')} numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}"
            )
            logger.info(
                f"{log_tag('PHI2N', 'LPF')} Difference is significant. Using the value from remezord."
            )
            pass
        elif approx:
            logger.info(
                f"{log_tag('PHI2N', 'LPF')} Using approximate numtaps: {numtaps_approx}"
            )
            numtaps = numtaps_approx
        if all(bands) <= 1.0:
            taps = remez(
                numtaps=numtaps,
                bands=bands,
                desired=amps,
                weight=weight,
                fs=1.0,
                grid_density=20,
            )
        else:
            taps = remez(
                numtaps=numtaps,
                bands=bands,
                desired=amps,
                weight=weight,
                fs=fs,
                grid_density=20,
            )

        return taps

    def _create_bpf(self, f_stop1, f_pass1, f_pass2, f_stop2, fs, approx=False):
        """
        Create a band-pass FIR filter using the remez (Parks-McClellan) algorithm.

        This method designs an optimal FIR band-pass filter with specified stopband
        and passband frequencies. The filter design uses remezord to determine the
        optimal number of taps and remez to generate the filter coefficients.

        Args:
            f_stop1 (float): Lower stopband edge frequency in Hz (below passband).
            f_pass1 (float): Lower passband edge frequency in Hz.
            f_pass2 (float): Upper passband edge frequency in Hz.
            f_stop2 (float): Upper stopband edge frequency in Hz (above passband).
            fs (float): Sampling frequency in Hz.
            approx (bool, optional): Whether to use approximate tap count instead of remezord.
                Default is False. If True, uses rule-of-thumb estimation.

        Returns:
            np.ndarray: FIR filter coefficients (taps) for use with scipy.signal.filtfilt.

        Notes:
            - Filter specifications: ~0.5 dB passband ripple, -80 dB stopband attenuation.
            - Frequency order must be: f_stop1 < f_pass1 < f_pass2 < f_stop2.
            - The number of taps is always odd (required by remez).
            - Filter coefficients are designed for use with filtfilt (zero-phase filtering).

        Examples:
            ```python
            converter = PhaseConverter()
            # Create BPF: passband 7.5-22.5 MHz, stopbands outside
            bpf_taps = converter._create_bpf(
                f_stop1=2.5e6,   # Lower stopband
                f_pass1=7.5e6,   # Lower passband
                f_pass2=22.5e6,  # Upper passband
                f_stop2=27.5e6,  # Upper stopband
                fs=50e6
            )
            # Apply filter
            from scipy.signal import filtfilt
            filtered_signal = filtfilt(bpf_taps, 1, signal)
            ```
        """
        nyq = 0.5 * fs
        numtaps_approx = int(4 / ((f_pass1 - f_stop1) / nyq))  # estimate from one side
        if numtaps_approx % 2 == 0:
            numtaps_approx += 1

        freqs = [f_stop1, f_pass1, f_pass2, f_stop2]
        amps = [0, 1, 0]
        rip_stop1 = 1e-4  # stop-band ripple (linear)
        rip_pass = 0.057501127785  # pass-band ripple (linear)
        rip_stop2 = 1e-4  # stop-band ripple (linear)
        rips = [rip_stop1, rip_pass, rip_stop2]

        [numtaps, bands, amps, weight] = remezord(
            freqs, amps, rips, Hz=fs, alg="herrmann"
        )

        # compare the numtaps from remezord and approximate one
        if abs(numtaps - numtaps_approx) > 50 and not approx:  # Allow some tolerance
            logger.info(
                f"{log_tag('PHI2N', 'BPF')} numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}"
            )
            logger.info(
                f"{log_tag('PHI2N', 'BPF')} Difference is significant. Using the value from remezord."
            )
            pass
        elif approx:
            logger.info(
                f"{log_tag('PHI2N', 'BPF')} Using approximate numtaps: {numtaps_approx}"
            )
            numtaps = numtaps_approx

        if all(bands) <= 1.0:
            taps = remez(
                numtaps=numtaps,
                bands=bands,
                desired=amps,
                weight=weight,
                fs=1.0,
                grid_density=20,
            )
        else:
            taps = remez(
                numtaps=numtaps,
                bands=bands,
                desired=amps,
                weight=weight,
                fs=fs,
                grid_density=20,
            )

        return taps

    def _plot_filter_response(self, taps, fs=None, title="Frequency Response"):
        """
        Plot the frequency response of an FIR filter.

        This method visualizes the magnitude and phase response of a filter to help
        verify filter design and characteristics.

        Args:
            taps (np.ndarray): FIR filter coefficients (from _create_lpf or _create_bpf).
            fs (float, optional): Sampling frequency in Hz. If None, uses 2π (normalized frequency).
                Default is None.
            title (str, optional): Plot title. Default is "Frequency Response".

        Notes:
            - Uses scipy.signal.freqz to compute frequency response.
            - If fs is None, frequency axis will be in rad/sample (normalized).
            - If fs is provided, frequency axis will be in Hz.
            - Calls plt.show() to display the plot.

        Examples:
            ```python
            converter = PhaseConverter()
            lpf_taps = converter._create_lpf(0.5e6, 1e6, fs=50e6)
            converter._plot_filter_response(lpf_taps, fs=50e6, title="LPF Response")
            ```
        """
        if fs is None:
            fs = 2 * np.pi
            logger.warning(
                f"{log_tag('PHI2N', 'VFILT')} fs is not provided, using 2*pi, resulting in frequency in rad/sample"
            )
        freqs_at_response, responses = freqz(taps, worN=2048, fs=fs)
        plot_response(freqs_at_response, responses, title)
        plt.show()

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
    ) -> np.ndarray:
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

        Returns:
            np.ndarray: Phase array in radians (same length as input signals).
                Phase is accumulated from differential phase differences and baseline-corrected.

        Notes:
            - BPF and LPF are designed automatically based on f_center.
            - Uses scipy.signal.hilbert for complex demodulation (matches MATLAB behavior).
            - Baseline correction uses first 10000 samples (or 1000 if signal is shorter).
            - The method uses Numba-optimized functions for differential phase calculation.

        Examples:
            ```python
            converter = PhaseConverter()
            # Determine f_center from STFT analysis
            f_center = 15e6  # 15 MHz IF frequency
            
            # Calculate phase using CDM
            phase = converter.calc_phase_cdm(
                ref_signal, probe_signal,
                fs=50e6,
                f_center=f_center,
                isbpf=True,
                islpf=True
            )
            
            # With filter visualization
            phase = converter.calc_phase_cdm(
                ref_signal, probe_signal,
                fs=50e6, f_center=f_center,
                plot_filters=True  # Shows BPF and LPF responses
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

        # 5. Calculate phase
        if not iszif:      
        
            # The matlab script uses a differential method.
            re = np.real(demod_lpf)
            im = np.imag(demod_lpf)

            # Vectorized differential phase calculation (cross-product method)
            phase_diff = _calculate_differential_phase(re, im)

            # The first sample is phase_diff[0] at _accumulate_phase_diff, so no need to prepend a 0.
            phase_accum = _accumulate_phase_diff(phase_diff)
        else:
            demod_zif = ref_hilbert.conj() * hilbert(prob_bpf)
            phase_accum = np.angle(demod_zif)

        # MATLAB uses first 10000 samples; use same or adapt based on signal length
        if len(phase_accum) > 10000:
            phase_accum -= np.mean(phase_accum[:10000])
        elif len(phase_accum) > 1000:
            phase_accum -= np.mean(phase_accum[:1000])

        if isflip:
            phase_accum *= -1

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
            - Phase difference: Δφ = probe_phase - ref_phase
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


# if __name__ == "__main__":
#     # pc = PhaseConverter()
#     # print(pc.get_params(94))
#     pass
