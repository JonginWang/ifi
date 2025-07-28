import numpy as np
import configparser
from scipy import constants
from scipy.signal import hilbert, firwin, remez, filtfilt, freqz
import matplotlib.pyplot as plt
from typing import Dict, Any
import pandas as pd
import logging

# "eegpy" is a package that contains the remezord function, translated from MATLAB
# Though remezord is not in scipy, we could import by piece of script under ifi/analysis/utils/remezord.py
from ifi.analysis.utils.remezord import remezord
from ifi.analysis.plots import plot_response


class PhaseConverter:
    def __init__(self, config_path: str = 'ifi/analysis/if_config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.constants = {}
        if self.config.has_section('constants'):
            for key, name in self.config['constants'].items():
                if hasattr(constants, name):
                    self.constants[key] = getattr(constants, name)
                else:
                    raise ValueError(f"Constant '{name}' not found in scipy.constants")

    def get_interferometer_params(self, freq_ghz: int) -> Dict[str, Any]:
        """Gets parameters for a given interferometer frequency."""
        section = f"{freq_ghz}GHz"
        if not self.config.has_section(section):
            raise ValueError(f"Configuration for {freq_ghz}GHz not found in config file.")
        
        params = dict(self.config[section])
        params['freq'] = float(params['freq'])
        params['n_ch'] = int(params['n_ch'])
        params['n_path'] = self.config.getint(section, 'n_path')
        return params

    def phase_to_density(self, phase: np.ndarray, freq_ghz: int) -> np.ndarray:
        """
        Converts phase (in radians) to line-integrated density (m^-2).
        """
        params = self.get_interferometer_params(freq_ghz)
        freq = params['freq']
        
        m_e = self.constants['m_e']
        eps0 = self.constants['eps0']
        qe = self.constants['qe']
        c = self.constants['c']

        n_c = m_e * eps0 * (2 * np.pi * freq)**2 / qe**2
        
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
        nedl = (c * n_c / (np.pi * freq)) * phase

        if params['n_path'] > 0:
            nedl /= params['n_path']
            
        return nedl

    def calc_phase_iq_atan2(self, i_signal: np.ndarray, q_signal: np.ndarray, isflip: bool = False) -> np.ndarray:
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
        phase -= phase[0] # Calibrate to start at 0
        if isflip:
            phase *= -1
        return phase

    def calc_phase_iq_asin2(self, i_signal: np.ndarray, q_signal: np.ndarray, isflip: bool = False) -> np.ndarray:
        """
        Calculates the phase from I and Q signals using a differential cross-product method.
        This is based on the IQPostprocessing(...).m script.
        """
        # 1. Normalize I and Q signals
        iq_mag = np.sqrt(i_signal**2 + q_signal**2)
        # Avoid division by zero, set magnitude to 1 if it's 0.
        iq_mag[iq_mag == 0] = 1.0
        i_norm = i_signal / iq_mag
        q_norm = q_signal / iq_mag

        # 2. Calculate the phase difference between consecutive points (vectorized)
        # Numerator of the arcsin argument: I_n(i) * Q_n(i+1) - Q_n(i) * I_n(i+1)
        numerator = i_norm[:-1] * q_norm[1:] - q_norm[:-1] * i_norm[1:]

        # Denominator of the arcsin argument: sqrt((I_n(i)+I_n(i+1))^2 + (Q_n(i)+Q_n(i+1))^2)
        denominator = np.sqrt((i_norm[:-1] + i_norm[1:])**2 + (q_norm[:-1] + q_norm[1:])**2)
        denominator[denominator == 0] = 1.0 # Avoid division by zero

        # The argument for arcsin can be > 1 or < -1 due to floating point inaccuracies. Clip it.
        ratio = np.clip(numerator / denominator, -1.0, 1.0)

        # Calculate the differential phase in radians
        del_phase = 2 * np.arcsin(ratio)

        # Handle any potential NaNs, as in the MATLAB code
        del_phase[np.isnan(del_phase)] = 0.0

        # 3. Accumulate the phase differences
        # The first phase value is 0 since it's a differential measurement.
        accumulated_phase = np.concatenate(([0.0], np.cumsum(del_phase)))

        # Match the sign from the MATLAB script
        if isflip:
            accumulated_phase *= -1

        return accumulated_phase

    def _create_lpf(self, f_pass, f_stop, fs, approx=False):
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
        rip_pass = 0.057501127785   # pass-band ripple (linear)
        rip_stop = 1e-4             # stop-band ripple (linear)
        rips = [rip_pass, rip_stop]

        [numtaps, bands, amps, weight] = remezord(freqs, amps, rips, Hz=fs, alg='herrmann')
        
        # compare the numtaps from remezord and approximate one
        if abs(numtaps - numtaps_approx) > 50 and not approx: # Allow some tolerance
            logging.info(f"numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}")
            logging.info("Difference is significant. Using the value from remezord.")
            pass
        elif approx:
            logging.info(f"Using approximate numtaps: {numtaps_approx}")
            numtaps = numtaps_approx

        taps = remez(numtaps, bands, amps, weight, fs=fs, grid_density=20)

        return taps

    def _create_bpf(self, f_stop1, f_pass1, f_pass2, f_stop2, fs, approx=False):
        """Creates a band-pass FIR filter using the remez algorithm."""
        nyq = 0.5 * fs
        numtaps_approx = int(4 / ((f_pass1 - f_stop1) / nyq)) # estimate from one side
        if numtaps_approx % 2 == 0:
            numtaps_approx += 1

        freqs = [f_stop1, f_pass1, f_pass2, f_stop2]
        amps = [0, 1, 0]
        rip_stop1 = 1e-4             # stop-band ripple (linear)
        rip_pass = 0.057501127785    # pass-band ripple (linear)
        rip_stop2 = 1e-4             # stop-band ripple (linear)
        rips = [rip_stop1, rip_pass, rip_stop2]

        [numtaps, bands, amps, weight] = remezord(freqs, amps, rips, Hz=fs, alg='herrmann')
        
        # compare the numtaps from remezord and approximate one
        if abs(numtaps - numtaps_approx) > 50 and not approx: # Allow some tolerance
            logging.info(f"numtaps from remezord: {numtaps}, numtaps_approx: {numtaps_approx}")
            logging.info("Difference is significant. Using the value from remezord.")
            pass
        elif approx:
            logging.info(f"Using approximate numtaps: {numtaps_approx}")
            numtaps = numtaps_approx

        taps = remez(numtaps, bands, amps, weight, fs=fs, grid_density=20)
        return taps

    def _plot_filter_response(self, taps, fs=None, title="Frequency Response"):
        """Plots the frequency response of the FIR filter."""
        if fs is None:
            fs = 2*np.pi
            logging.warning(f"fs is not provided, using 2*pi, resulting in frequency in rad/sample")
        freqs_at_response, responses = freqz(taps, worN=2048, fs=fs)
        plot_response(freqs_at_response, responses, title)
        plt.show()

    def calc_phase_cdm(self, ref_signal: np.ndarray, prob_signal: np.ndarray, fs: float, f_center: float, isbpf: bool = True, isconj: bool = False, isold: bool = False, isflip: bool = False, plot_filters: bool = False) -> np.ndarray:
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
        if f_center > 65e6:
            fcl, f0l, f0r, fcr = 55e6, 60e6, 75e6, 80e6
        elif f_center > 35e6:
            fcl, f0l, f0r, fcr = 30e6, 35e6, 45e6, 50e6
        else:
            fcl, f0l, f0r, fcr = 3e6, 7e6, 13e6, 17e6
            
        lpf_pass = 0.5e6
        lpf_stop = 1e6
        
        bpf_coeffs = self._create_bpf(fcl, f0l, f0r, fcr, fs)
        lpf_coeffs = self._create_lpf(lpf_pass, lpf_stop, fs)

        if plot_filters:
            self._plot_filter_response(bpf_coeffs, fs, title=f"BPF Response ({fcl/1e6}-{fcr/1e6} MHz)")
            self._plot_filter_response(lpf_coeffs, fs, title=f"LPF Response ({lpf_pass/1e6} MHz)")

        # 2. Apply BPF if enabled
        ref_bpf = filtfilt(bpf_coeffs, 1, ref_signal) if isbpf else ref_signal
        prob_bpf = filtfilt(bpf_coeffs, 1, prob_signal) if isbpf else prob_signal
        
        # 3. Complex demodulation
        ref_hilbert = hilbert(ref_bpf)
        # Using conjugate on reference aligns with convention for phase difference (phi_prob - phi_ref)
        demod_signal = ref_hilbert.conj() * prob_bpf if isconj else ref_hilbert * prob_bpf

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
        denominator = (np.sqrt(re[:-1]**2 + im[:-1]**2) * np.sqrt(re[1:]**2 + im[1:]**2))
        denominator[denominator == 0] = 1e-12 # Avoid division by zero

        # Clip argument to arcsin to handle potential floating point errors
        ratio = np.clip((re[:-1] * im[1:] - im[:-1] * re[1:]) / denominator, -1.0, 1.0)
        d_phase = np.arcsin(ratio)
        
        # The first sample is lost in differentiation, so prepend a 0.
        phase_accum = np.concatenate(([0], np.cumsum(d_phase)))

        # Calibrate to start at 0, assuming first 1000 samples are pre-plasma
        if len(phase_accum) > 1000:
            phase_accum -= np.mean(phase_accum[:1000]) 

        if isflip:
            phase_accum *= -1

        return phase_accum

    def calc_phase_fpga(self, ref_phase: np.ndarray, probe_phase: np.ndarray, time: np.ndarray, amp_signal: np.ndarray, isflip: bool = False) -> np.ndarray:
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

    def correct_baseline(self, density_df: pd.DataFrame, time_axis: np.ndarray, mode: str, vest_data: pd.DataFrame = None, ip_column_name: str = None):
        """
        Corrects the baseline of the calculated density.

        Args:
            density_df: DataFrame with density data columns.
            time_axis: The common time axis for the data.
            mode: The baseline correction mode ('ip' or 'trig').
            vest_data: DataFrame containing VEST data, required for 'ip' mode.
            ip_column_name: The name of the plasma current column in vest_data.
        """
        corrected_df = density_df.copy()

        if mode == 'ip':
            if vest_data is None or vest_data.empty:
                logging.warning("Warning: 'ip' baseline mode selected but VEST data is not available. Skipping correction.")
                return corrected_df
            
            if ip_column_name is None or ip_column_name not in vest_data.columns:
                logging.warning(f"Warning: Plasma current column '{ip_column_name}' not in VEST data. Skipping 'ip' baseline correction.")
                return corrected_df
            
            ip_data = vest_data[ip_column_name]
            # Find the ramp-up point (first time Ip > 5kA)
            # Note: Assuming Ip is in Amperes. If in kA, this threshold should be 5.
            ip_threshold = 5e3 if np.nanmax(ip_data) > 1000 else 5

            try:
                ramp_up_indices = np.where(ip_data > ip_threshold)[0]
                if len(ramp_up_indices) == 0:
                    raise IndexError("Threshold not exceeded")
                
                ramp_up_index = ramp_up_indices[0]
                t_rampup = ip_data.index[ramp_up_index]
                
                # Define baseline window: 3 to 8 ms before ramp-up
                t_start = t_rampup - 8e-3
                t_end = t_rampup - 3e-3

            except IndexError:
                logging.warning(f"Warning: Plasma current never exceeded threshold ({ip_threshold}). Cannot determine ramp-up for 'ip' baseline. Skipping.")
                return corrected_df

        elif mode == 'trig':
            t_start, t_end = 0.285, 0.290
        
        else:
            logging.warning(f"Warning: Invalid baseline mode '{mode}'. Skipping correction.")
            return corrected_df

        # Find indices for the baseline window
        baseline_indices = np.where((time_axis >= t_start) & (time_axis <= t_end))[0]

        if len(baseline_indices) == 0:
            logging.warning(f"Warning: No data found in the baseline window [{t_start:.4f}s, {t_end:.4f}s]. Skipping correction.")
            return corrected_df

        logging.info(f"Correcting baseline using window [{t_start:.4f}s, {t_end:.4f}s] ({len(baseline_indices)} points).")
        for col in corrected_df.columns:
            baseline_mean = corrected_df[col].iloc[baseline_indices].mean()
            corrected_df[col] -= baseline_mean
            logging.info(f"  - Column '{col}': Removed baseline of {baseline_mean:.2e}")
            
        return corrected_df


if __name__ == '__main__':
    # pc = PhaseConverter()
    # print(pc.get_interferometer_params(94))
    pass
