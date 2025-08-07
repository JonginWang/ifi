import os

# Set a project-local cache directory for numba to avoid permission errors.
# This must be set before importing ssqueezepy to prevent caching issues.
numba_cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'cache', 'numba_cache')
os.makedirs(numba_cache_dir, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir

import numpy as np
import pandas as pd
from scipy import signal as spsig
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import ssqueezepy as ssqpy
from ssqueezepy.experimental import scale_to_freq
import pywt
from typing import Tuple, Union
from collections import defaultdict
import logging

from ifi.utils.common import assign_kwargs
from ifi.utils.common import LogManager

LogManager()

class SpectrumAnalysis:
    def __init__(self):
        self.kwargs_stft_fallback = {
            'window': ('kaiser', 8), 'nperseg': 10000, 'noverlap': 5000, 'mfft': None,
            'dual_win': None, 'scale_to': 'magnitude', 'phase_shift': 0, 'padtype': 'reflect',
            'padding': 'even'
        }
        self._cached_stft_kwargs = {}
        self._cached_window = None
        self._cached_kwargs_full = None

    def _get_stft_kwargs(self, **kwargs):
        # Only recompute window if kwargs have changed or cache is empty
        if kwargs != self._cached_stft_kwargs or self._cached_kwargs_full is None:
            self._cached_stft_kwargs = kwargs
            
            final_kwargs = self.kwargs_stft_fallback.copy()
            final_kwargs.update(kwargs)

            nperseg = final_kwargs['nperseg']
            window_spec = final_kwargs['window']
            
            if isinstance(window_spec, (str, tuple)):
                self._cached_window = spsig.get_window(window_spec, nperseg)
            elif isinstance(window_spec, np.ndarray) and len(window_spec) == nperseg:
                self._cached_window = window_spec
            else:
                raise TypeError(f"Unsupported window type or length: {window_spec}")
            
            final_kwargs['win'] = self._cached_window
            final_kwargs['hop'] = nperseg - final_kwargs['noverlap']
            final_kwargs['mfft'] = nperseg if final_kwargs['mfft'] is None else final_kwargs['mfft']
            self._cached_kwargs_full = final_kwargs
        
        return self._cached_kwargs_full

    def _translate_kwargs(self, kwargs, target_lib='scipy'):
        """Translates kwargs for different libraries."""
        translated = kwargs.copy()
        if target_lib == 'ssqueezepy':
            if 'mfft' in translated:
                translated['n_fft'] = translated.pop('mfft')
            if 'hop' in translated:
                translated['hop_len'] = translated.pop('hop')
        elif target_lib == 'scipy':
            if 'n_fft' in translated:
                translated['mfft'] = translated.pop('n_fft')
            if 'hop_len' in translated:
                translated['hop'] = translated.pop('hop_len')
        return translated

    def compute_stft(self, signal: np.ndarray, fs: float, t_start: float = 0.0, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_kwargs = self._get_stft_kwargs(**kwargs)
        scipy_kwargs = self._translate_kwargs(all_kwargs, 'scipy')
        
        SFT = spsig.ShortTimeFFT(scipy_kwargs['win'], scipy_kwargs['hop'], fs=fs, mfft=scipy_kwargs['mfft'],
                                     dual_win=scipy_kwargs['dual_win'], scale_to=scipy_kwargs['scale_to'],
                                     phase_shift=scipy_kwargs['phase_shift'])
        Zxx = SFT.stft(signal, p0=0, p1=None, k_offset=0, padding=scipy_kwargs['padding'], axis=-1)
        return SFT.f, SFT.t(signal.shape[-1]) + t_start, Zxx

    def compute_stft_sqpy(self, signal: np.ndarray, fs: float, t_start: float = 0.0, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_kwargs = self._get_stft_kwargs(**kwargs)
        sqpy_kwargs = self._translate_kwargs(all_kwargs, 'ssqueezepy')

        Sxx = ssqpy.stft(signal, window=sqpy_kwargs['win'], n_fft=sqpy_kwargs['n_fft'], 
                         hop_len=sqpy_kwargs['hop_len'], fs=fs, padtype=sqpy_kwargs['padtype'])
        
        # Create frequency and time axes manually
        n_fft = sqpy_kwargs['n_fft']
        hop_len = sqpy_kwargs['hop_len']
        freqs_stft = np.fft.rfftfreq(n_fft, 1/fs)  # Only positive frequencies
        
        # Time axis with hopping consideration and original start time
        n_frames = Sxx.shape[1]
        time_stft = np.arange(0, (len(signal) - 1)//hop_len + 1) / fs + t_start
        
        return freqs_stft, time_stft, Sxx

    def compute_cwt(self, signal: np.ndarray, fs: float, wavelet: str = "gmw", 
                    f_min: float = None, f_max: float = None,
                   nv: int = 32, scales: str = 'log-piecewise', **kwargs):
        """
        Compute Continuous Wavelet Transform using ssqueezepy.
        
        Args:
            signal: Input signal
            fs: Sampling frequency  
            wavelet: Wavelet type
            f_min: Minimum frequency (Hz)
            f_max: Maximum frequency (Hz)
            nv: Number of voices (wavelets per octave)
            scales: Scale distribution ('log', 'log-piecewise', 'linear' or array)
            **kwargs: Additional arguments for ssqpy.cwt
        """
        logging.debug(f"Computing CWT with ssqueezepy using '{wavelet}' wavelet.")
        wav = ssqpy.Wavelet(wavelet)
        
        # Convert frequency range to scales if provided
        if f_min is not None and f_max is not None:
            # Calculate appropriate scales for frequency range
            scale_min = fs / (2 * f_max)  # Higher freq -> smaller scale
            scale_max = fs / (2 * f_min)  # Lower freq -> larger scale
            
            # Generate logarithmically spaced scales
            n_scales = int(nv * np.log2(scale_max / scale_min))
            scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
            
            Wx, scales_out = ssqpy.cwt(signal, fs=fs, wavelet=wav, scales=scales, **kwargs)
        else:
            # Use default scaling
            Wx, scales_out = ssqpy.cwt(signal, fs=fs, wavelet=wav, nv=nv, **kwargs)
        
        # Convert scales to frequencies
        freqs_cwt = scale_to_freq(scales_out, wav, len(signal), fs=fs)

        return freqs_cwt, Wx

    def find_freq_ridge(self, Zxx: np.ndarray, f: np.ndarray, prominence: float = None, distance: float = None) -> np.ndarray:
        """
        Finds the frequency "ridge" in a spectrogram by locating the frequency
        with the maximum power for each time slice.

        Args:
            Zxx (np.ndarray): The complex matrix from the STFT (frequencies x times).
            f (np.ndarray): The array of sample frequencies.
            prominence (float, optional): Ignored. Kept for signature compatibility.
            distance (float, optional): Ignored. Kept for signature compatibility.

        Returns:
            np.ndarray: A 1D array containing the frequency with the highest power for each time segment.
        """
        if Zxx.ndim != 2:
            raise ValueError(f"`Zxx` must be a 2-D array, but got shape {Zxx.shape}")

        # Get the magnitude squared (power)
        power_spectrum = np.abs(Zxx)**2
        
        # Find the index of the maximum power for each time slice (along the frequency axis=0)
        peak_indices = np.argmax(power_spectrum, axis=0)
        
        # Map the indices to frequencies
        frequency_ridge = f[peak_indices]
        
        return frequency_ridge

    def find_center_frequency_fft(self, signal: np.ndarray, fs: float) -> float:
        """
        Finds the center frequency of a signal using FFT.

        This method computes the FFT of the signal and identifies the frequency
        with the highest magnitude, considered as the center frequency. To avoid
        low-frequency noise, it only searches above a certain threshold.

        Args:
            signal (np.ndarray): The input signal array.
            fs (float): The sampling frequency of the signal.

        Returns:
            float: The estimated center frequency in Hz.
        """
        n = len(signal)
        if n == 0:
            return 0.0

        # Compute FFT
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(n, 1 / fs)

        # We only care about the positive frequencies
        xf_positive = xf[:n//2]
        yf_positive = 2.0/n * np.abs(yf[0:n//2])

        # Define a minimum frequency threshold to ignore DC and low-freq noise
        # Threshold: Min(5% of Nyquist frequency, 5MHz)
        nyquist = fs / 2
        min_freq_threshold = min(nyquist * 0.05, 5e6)
        
        # Find the index where frequency is just above the threshold
        try:
            search_start_idx = np.where(xf_positive > min_freq_threshold)[0][0]
        except IndexError:
            # This happens if all frequencies are below the threshold.
            # In this case, just search from the beginning of positive freqs.
            search_start_idx = 0

        # Find the peak frequency above the threshold
        if search_start_idx >= len(yf_positive):
             # If search start index is out of bounds, there's no valid range to search
            logging.warning("No valid range to search for center frequency. Returning 0.0 Hz.")
            return 0.0
            
        peak_idx = np.argmax(yf_positive[search_start_idx:]) + search_start_idx
        f_center = xf_positive[peak_idx]

        logging.info(f"Center frequency: {f_center/1e6:.2f} MHz")

        return f_center

if __name__ == '__main__':
    # Example usage:
    analyzer = SpectrumAnalysis()
    fs1 = 5e6
    fs2 = 8e6
    fs = 50e6
    t = np.arange(0, 1, 1/fs)
    signal1 = np.sin(2 * np.pi * fs1 * t)
    signal2 = np.sin(2 * np.pi * fs2 * t)
    signal = signal1 + 0.5 *signal2 + 0.1 * np.random.randn(len(t))

    f_center = analyzer.find_center_frequency_fft(signal, fs)
    logging.info(f"Center frequency: {f_center/1e6:.2f} MHz")

    # Call with default parameters using SciPy
    f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
    ridge = analyzer.find_freq_ridge(Zxx, f)
    logging.info("--- SciPy STFT ---")
    logging.info(f"STFT shape: {Zxx.shape}")
    logging.info(f"Ridge frequency: {np.mean(ridge)}")
    
    # Call using ssqueezepy
    f_ssq, t_ssq, Sxx = analyzer.compute_stft_sqpy(signal, fs, n_fft=1024, hop_len=512)
    ridge_ssq = analyzer.find_freq_ridge(Sxx, f_ssq)
    logging.info("--- ssqueezepy STFT ---")
    logging.info(f"STFT shape: {Sxx.shape}")
    logging.info(f"Ridge frequency: {np.mean(ridge_ssq)}")
    pass
