#!/usr/bin/env python3
"""
Phase to Density Conversion
===========================

This module contains the functions for phase to density conversion.
The functions are optimized for performance using numba.
The functions are used in the main_analysis.py module.

Functions:
    numba-optimized functions:
        _normalize_iq_signals: Normalize I and Q signals.
        _calculate_differential_phase: Calculate differential phase using cross-product method.
        _accumulate_phase_diff: Accumulate phase difference.
        _phase_to_density: Convert phase to density.
    get_interferometry_params: Get interferometry parameters for a given shot number and filename.

Classes:
    PhaseConverter: A class for converting phase to density.
"""

import logging
from pathlib import Path
import configparser
from typing import Dict, Any
import numpy as np
import numba
import pandas as pd
from scipy import constants
from scipy.signal import hilbert, remez, filtfilt, freqz
import matplotlib.pyplot as plt

# "eegpy" is a package that contains the remezord function, translated from MATLAB
# Though remezord is not in scipy, we could import by piece of script under ifi/analysis/functions/remezord.py
from ifi.analysis.functions.remezord import remezord
from ifi.analysis.plots import plot_response
from ifi.utils.common import LogManager

logger = LogManager().get_logger(__name__)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _normalize_iq_signals(i_signal, q_signal):
    """
    Numba-optimized function to normalize I and Q signals.

    Args:
        i_signal (np.ndarray): I signal.
        q_signal (np.ndarray): Q signal.

    Returns:
        i_norm (np.ndarray): Normalized I signal.
        q_norm (np.ndarray): Normalized Q signal.
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
    Numba-optimized function to calculate differential phase using cross-product method.

    Args:
        i_norm (np.ndarray): Normalized I signal.
        q_norm (np.ndarray): Normalized Q signal.

    Returns:
        phase_diff (np.ndarray): Differential phase.
    """
    n = len(i_norm)
    phase_diff = np.zeros(n - 1)

    for i in range(n - 1):
        # Numerator: I_n(i) * Q_n(i+1) - Q_n(i) * I_n(i+1)
        numerator = i_norm[i] * q_norm[i + 1] - q_norm[i] * i_norm[i + 1]

        # Denominator: sqrt((I_n(i)+I_n(i+1))^2 + (Q_n(i)+Q_n(i+1))^2)
        denominator = np.sqrt(
            (i_norm[i] + i_norm[i + 1]) ** 2 + (q_norm[i] + q_norm[i + 1]) ** 2
        )
        if denominator == 0:
            denominator = 1.0

        # Clip ratio to [-1, 1] for arcsin
        ratio = numerator / denominator
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0

        phase_diff[i] = np.arcsin(ratio)

    return phase_diff


@numba.jit(nopython=True, cache=True, fastmath=True)
def _accumulate_phase_diff(phase_diff):
    """
    Numba-optimized cumulative sum with zero padding.

    Args:
        phase_diff (np.ndarray): Differential phase.

    Returns:
        phase_accum (np.ndarray): Accumulated phase.
    """
    n = len(phase_diff)
    phase_accum = np.zeros(n + 1)

    for i in range(n):
        phase_accum[i + 1] = phase_accum[i] + phase_diff[i]

    return phase_accum


@numba.jit(nopython=True, cache=True, fastmath=True)
def _phase_to_density(phase, freq, c, m_e, eps0, qe, n_path):
    """
    Numba-optimized core calculation for phase to density conversion.

    Args:
        phase (np.ndarray): Phase.
        freq (float): Frequency.
        c (float): Speed of light.
        m_e (float): Electron mass.
        eps0 (float): Permittivity of free space.
        qe (float): Electron charge.
        n_path (int): Number of interferometer passes.

    Returns:
        nedl (np.ndarray): Line-integrated density.
    """
    # Calculate critical density
    n_c = m_e * eps0 * (2 * np.pi * freq) ** 2 / qe**2

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
    Get interferometry parameters for a given shot number and filename.

    Args:
        shot_num (int): Shot number
        filename (str): Filename to extract frequency information
        config_path (str, optional): Path to config file. If None, uses default.

    Returns:
        Dict: Dictionary containing method and frequency information
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
    A class for converting phase to density.

    Attributes:
        config: Config parser for the configuration file.
        constants: Constants for the calculation.

    Methods:
        get_params: Get parameters for a given interferometer frequency.
        get_analysis_params: Get analysis parameters for a given shot number and filename.
        phase_to_density: Convert phase to density.
        calc_phase_iq_atan2: Calculate phase using atan2 method.
        calc_phase_iq_asin2: Calculate phase using asin2 method.
        calc_phase_iq: Calculate phase using atan2 method.
        _create_lpf: Create a low pass filter.
        _create_bpf: Create a band pass filter.
        _plot_filter_response: Plot the frequency response of a filter.
        calc_phase_cdm: Calculate phase using CDM method.
        calc_phase_fpga: Calculate phase using FPGA method.
        correct_baseline: Correct the baseline of a density dataframe.

    Example:
        ```python
        from ifi.analysis.phi2ne import PhaseConverter
        converter = PhaseConverter()
        converter.get_params(94)
        converter.get_analysis_params(45821, "45821_056.csv")
        converter.phase_to_density(np.array([0.1, 0.2, 0.3]), 94.0e9, 2)
        converter.calc_phase_iq_atan2(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
        converter.calc_phase_iq_asin2(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
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
        """Gets parameters for a given interferometer frequency."""
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
        Returns interferometry analysis parameters based on shot number and filename.
        This method uses the class's config path and provides better integration
        with phase_to_density calculations.

        Args:
            shot_num: Shot number
            filename: Name of the data file (e.g., "45821_056.csv", "45821_ALL.csv")

        Returns:
            Dictionary containing analysis parameters compatible with phase_to_density
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
        Converts phase (in radians) to line-integrated density (m^-2).

        OPTIMIZED: Uses numba JIT compilation for enhanced performance.

        Args:
            phase: Phase array in radians
            freq_hz: Frequency in Hz (takes precedence over analysis_params and wavelength)
            n_path: Number of interferometer passes (takes precedence over analysis_params)
            analysis_params: Dictionary from get_analysis_params() containing 'freq' and 'n_path'
            wavelength: Wavelength in meters (will be converted to frequency using c = λf)

        Returns:
            Line-integrated density in m^-2
        """
        # Convert wavelength to frequency if provided
        if wavelength is not None:
            if freq_hz is not None:
                import logging

                logging.warning("Both freq_hz and wavelength provided. Using freq_hz.")
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
        self, i_signal: np.ndarray, q_signal: np.ndarray, isflip: bool = False
    ) -> np.ndarray:
        """
        Calculates the phase from I and Q signals.
        Normalizes the I/Q signals to a unit circle and calculates the accumulated phase.
        """
        # Normalize
        iq_mag = np.sqrt(i_signal**2 + q_signal**2)
        iq_mag[iq_mag == 0] = 1.0
        i_norm = i_signal / iq_mag
        q_norm = q_signal / iq_mag

        # Calculate phase
        phase = np.unwrap(np.arctan2(q_norm, i_norm))
        phase -= phase[0]  # Calibrate to start at 0
        if isflip:
            phase *= -1
        return phase

    def calc_phase_iq_asin2(
        self, i_signal: np.ndarray, q_signal: np.ndarray, isflip: bool = False
    ) -> np.ndarray:
        """
        Calculates the phase from I and Q signals using a differential cross-product method.
        This is based on the IQPostprocessing(...).m script.

        OPTIMIZED: Uses numba JIT compilation for enhanced performance.
        """
        # 1. Normalize I and Q signals (numba-optimized)
        i_norm, q_norm = _normalize_iq_signals(i_signal, q_signal)

        # 2. Calculate the differential phase (numba-optimized)
        del_phase = _calculate_differential_phase(i_norm, q_norm)

        # Handle any potential NaNs, as in the MATLAB code
        del_phase[np.isnan(del_phase)] = 0.0

        # Scale by factor of 2 as in original algorithm
        del_phase *= 2.0

        # 3. Accumulate the phase differences (numba-optimized)
        accumulated_phase = _accumulate_phase_diff(del_phase)

        # Match the sign from the MATLAB script
        if isflip:
            accumulated_phase *= -1

        return accumulated_phase

    def calc_phase_iq(
        self, i_signal: np.ndarray, q_signal: np.ndarray, isflip: bool = False
    ) -> np.ndarray:
        """
        Calculates phase from I and Q signals using the default atan2 method.

        This is a convenience method that calls calc_phase_iq_atan2.

        Args:
            i_signal: In-phase signal array
            q_signal: Quadrature signal array
            isflip: Whether to flip the sign of the result

        Returns:
            Phase array in radians
        """
        return self.calc_phase_iq_asin2(i_signal, q_signal, isflip=isflip)

    def _create_lpf(self, f_pass, f_stop, fs, approx=False, max_taps=None):
        """Creates a low-pass FIR filter using the remez algorithm."""
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
            logging.info(
                f"    - [LPF] numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}"
            )
            logging.info(
                "    - [LPF] Difference is significant. Using the value from remezord."
            )
            pass
        elif approx:
            logging.info(f"    - [LPF] Using approximate numtaps: {numtaps_approx}")
            numtaps = numtaps_approx
        if all(bands) <= 1.0:
            taps = remez(numtaps, bands, amps, weight, fs=1.0, grid_density=20)
        else:
            taps = remez(numtaps, bands, amps, weight, fs=fs, grid_density=20)

        return taps

    def _create_bpf(self, f_stop1, f_pass1, f_pass2, f_stop2, fs, approx=False):
        """Creates a band-pass FIR filter using the remez algorithm."""
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
            logging.info(
                f"    - [BPF] numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}"
            )
            logging.info(
                "    - [BPF] Difference is significant. Using the value from remezord."
            )
            pass
        elif approx:
            logging.info(f"    - [BPF] Using approximate numtaps: {numtaps_approx}")
            numtaps = numtaps_approx

        if all(bands) <= 1.0:
            taps = remez(numtaps, bands, amps, weight, fs=1.0, grid_density=20)
        else:
            taps = remez(numtaps, bands, amps, weight, fs=fs, grid_density=20)

        return taps

    def _plot_filter_response(self, taps, fs=None, title="Frequency Response"):
        """Plots the frequency response of the FIR filter."""
        if fs is None:
            fs = 2 * np.pi
            logging.warning(
                "    ! [Filter Plot] fs is not provided, using 2*pi, resulting in frequency in rad/sample"
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
        isold: bool = False,
        isflip: bool = False,
        plot_filters: bool = False,
    ) -> np.ndarray:
        """
        Calculates phase using complex demodulation.
        This involves BPF, Hilbert transform, LPF, and phase extraction.

        Args:
            ref_signal: The reference channel signal.
            prob_signal: The probe channel signal.
            fs: The sampling frequency.
            f_center: The center frequency of the IF signal, determined from STFT.
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
        ref_hilbert = hilbert(ref_bpf)
        # Using conjugate on reference aligns with convention for phase difference (phi_prob - phi_ref)
        demod_signal = (
            ref_hilbert.conj() * prob_bpf if isconj else ref_hilbert * prob_bpf
        )

        # 4. Apply LPF to the demodulated signal
        demod_lpf = filtfilt(lpf_coeffs, 1, demod_signal)

        # 5. Calculate phase
        # The matlab script uses a differential method.
        if isold:
            re = filtfilt(lpf_coeffs, 1, np.real(demod_signal))
            im = filtfilt(lpf_coeffs, 1, np.imag(demod_signal))
        else:
            re = np.real(demod_lpf)
            im = np.imag(demod_lpf)

        # Vectorized differential phase calculation (cross-product method)
        denominator = np.sqrt(re[:-1] ** 2 + im[:-1] ** 2) * np.sqrt(
            re[1:] ** 2 + im[1:] ** 2
        )
        denominator[denominator == 0] = 1e-12  # Avoid division by zero

        # Clip argument to arcsin to handle potential floating point errors
        ratio = np.clip((re[:-1] * im[1:] - im[:-1] * re[1:]) / denominator, -1.0, 1.0)
        d_phase = np.arcsin(ratio)

        # The first sample is lost in differentiation, so prepend a 0.
        phase_accum = np.concatenate(([0], _accumulate_phase_diff(d_phase)))

        # Calibrate to start at 0, assuming first 1000 samples are pre-plasma
        if len(phase_accum) > 1000:
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
        Processes phase from FPGA data.

        This is a simplified version of the logic in FPGAreadData_ver2.m.
        It calculates the phase difference and performs a basic offset correction.

        The full MATLAB script includes more advanced features like:
        - Moving average filtering of the phase difference.
        - Dynamic detection of the plasma discharge window.
        - Masking of the phase during periods of low amplitude (beam diffraction).
        - More sophisticated pre/post-discharge offset correction.

        These can be implemented here later if needed.

        Args:
            ref_phase: The reference phase signal.
            probe_phase: The probe phase signal for a specific channel.
            time: The time array, used for potential windowing in future.
            amp_signal: The amplitude signal for the probe channel, for future use.

        Returns:
            The processed phase difference in radians.
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
        Corrects the baseline of the calculated density.

        Args:
            density_df: DataFrame with density data columns.
            time_axis: The common time axis for the data.
            mode: The baseline correction mode ('ip' or 'trig').
            shot_num: Shot number for VEST data retrieval (required for 'ip' mode).
            vest_data: DataFrame containing VEST data, required for 'ip' mode.
            ip_column_name: The name of the plasma current column in vest_data.
        """
        corrected_df = density_df.copy()

        if mode == "ip":
            if shot_num is None:
                logging.warning(
                    "    ! [Baseline Correction] 'ip' baseline mode selected but shot_num not provided. Skipping."
                )
                return corrected_df

            if vest_data is None or vest_data.empty:
                logging.warning(
                    "    ! [Baseline Correction] 'ip' baseline mode selected but VEST DB not available. Skipping."
                )
                return corrected_df

            if ip_column_name is None or ip_column_name not in vest_data.columns:
                logging.warning(
                    f"    ! [Baseline Correction] Plasma current column '{ip_column_name}' not in VEST DB. Skipping."
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
                logging.warning(
                    f"    ! [Baseline Correction] Plasma current never exceeded threshold ({ip_threshold})."
                )
                logging.warning(
                    "    ! Cannot determine ramp-up for 'ip' baseline. Skipping."
                )
                return corrected_df

        elif mode == "trig":
            t_start, t_end = 0.285, 0.290

        else:
            logging.warning(
                f"    ! [Baseline Correction] Invalid baseline mode '{mode}'. Skipping."
            )
            return corrected_df

        # Find indices for the baseline window
        baseline_idxs = np.where((time_axis >= t_start) & (time_axis <= t_end))[0]

        if len(baseline_idxs) == 0:
            logging.warning(
                f"    ! [Baseline Correction] No data found in the baseline window [{t_start:.4f}s, {t_end:.4f}s]. Skipping."
            )
            return corrected_df

        logging.info(
            f"    - [Baseline Correction] Correcting baseline using window [{t_start:.4f}s, {t_end:.4f}s] ({len(baseline_idxs)} points)."
        )
        for col in corrected_df.columns:
            baseline_mean = corrected_df[col].iloc[baseline_idxs].mean()
            corrected_df[col] -= baseline_mean
            logging.info(
                f"    - [Baseline Correction] Column '{col}': Removed baseline of {baseline_mean:.2e}"
            )

        return corrected_df


if __name__ == "__main__":
    # pc = PhaseConverter()
    # print(pc.get_params(94))
    pass
