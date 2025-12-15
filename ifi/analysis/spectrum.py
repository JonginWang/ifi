#!/usr/bin/env python3
"""
Spectrum Analysis
=================

This module contains functions for analyzing the spectrum of a signal. It
includes functions for computing the Short-Time Fourier Transform (STFT),
Continuous Wavelet Transform (CWT), and finding the frequency ridge.

Classes:
    SpectrumAnalysis: A class for analyzing the spectrum of a signal.
"""

import numpy as np
from scipy import signal as spsig
from typing import Tuple, Optional
from functools import lru_cache
import ssqueezepy as ssqpy
from ssqueezepy.experimental import scale_to_freq
try:
    from ..utils.common import LogManager, log_tag
    from ..utils.validation import validate_signal, validate_frequency
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag
    from ifi.utils.validation import validate_signal, validate_frequency
    
logger = LogManager().get_logger(__name__)


# Lazy-loaded getter for MATLAB-compatible ridge tracker without globals
@lru_cache(maxsize=1)
def _get_extract_ridges_matlab():
    """Return `extract_fridges`, importing once on first use (cached)."""
    from .functions.ridge_tracking import extract_fridges

    return extract_fridges


class SpectrumAnalysis:
    """Spectrum Analysis Class

    This class provides methods for analyzing the spectrum of a signal.

    Attributes:
        kwargs_stft_fallback: Default kwargs for STFT
        _cached_stft_kwargs: Cached kwargs for STFT
        _cached_window: Cached window for STFT
        _cached_kwargs_full: Cached full kwargs for STFT

    Methods:
        _get_stft_kwargs: Get STFT kwargs
        _translate_kwargs: Translate kwargs for different libraries
        compute_stft: Compute STFT
        compute_stft_sqpy: Compute STFT with ssqueezepy
        compute_cwt: Compute Continuous Wavelet Transform
        find_freq_ridge: Find frequency ridge
        find_center_frequency_fft: Find center frequency of signal

    Examples:
        ```python
        import numpy as np
        from ifi.analysis.spectrum import SpectrumAnalysis

        analyzer = SpectrumAnalysis()
        fs = 50e6
        t = np.arange(0, 1, 1/fs)
        signal = (np.sin(2 * np.pi * 10e6 * t)
                  + np.sin(2 * np.pi * 20e6 * t)
                  + 0.1 * np.random.randn(len(t)))
        
        # Compute STFT
        f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
        
        # Find frequency ridge
        ridge = analyzer.find_freq_ridge(Zxx, f, method='stft')
        
        # Find center frequency
        f_center = analyzer.find_center_frequency_fft(signal, fs)
        
        print(f"Center Frequency: {f_center / 1e6:.2f} MHz")
        print(f"Ridge Frequency: {np.mean(ridge) / 1e6:.2f} MHz")
        print(f"STFT Shape: {Zxx.shape}")
        print(f"STFT Time: {t_stft[0]} to {t_stft[-1]}")
        
        # Compute CWT with memory optimization
        freqs_cwt, Wx = analyzer.compute_cwt(
            signal, fs, 
            f_center=15e6, 
            f_deviation=0.1,
            decimation_factor=4
        )
        print(f"CWT Shape: {Wx.shape}")
        print(f"CWT Frequency Range: [{freqs_cwt.min()/1e6:.2f}, {freqs_cwt.max()/1e6:.2f}] MHz")
        ```
    """

    def __init__(self):
        """Initialize SpectrumAnalysis class"""
        # Default kwargs for STFT
        self.kwargs_stft_fallback = {
            "window": ("kaiser", 8),
            # nperseg/noverlap/mfft are chosen dynamically when not provided
            "nperseg": None,
            "noverlap": None,  # 10000
            "mfft": None,      # 5000
            "dual_win": None,
            "scale_to": "magnitude",
            "phase_shift": 0,
            "padtype": "reflect",
            "padding": "even",
        }
        self._cached_stft_kwargs = {}
        self._cached_window = None
        self._cached_kwargs_full = None

    def _get_stft_kwargs(self, **kwargs):
        """
        Get STFT kwargs from kwargs and fallback kwargs
        for mutually compatible usage of scipy and ssqueezepy.

        Args:
            **kwargs: Additional arguments for STFT

        Returns:
            dict: Dictionary containing the STFT kwargs

        Raises:
            TypeError: If the window type or length is not supported
            ValueError: If the Zxx matrix is not a 2-D array
        """
        # Only recompute window if kwargs have changed or cache is empty
        if kwargs != self._cached_stft_kwargs or self._cached_kwargs_full is None:
            self._cached_stft_kwargs = kwargs

            final_kwargs = self.kwargs_stft_fallback.copy()
            final_kwargs.update(kwargs)

            nperseg = final_kwargs["nperseg"]
            window_spec = final_kwargs["window"]

            if isinstance(window_spec, (str, tuple)):
                self._cached_window = spsig.get_window(window_spec, nperseg)
            elif isinstance(window_spec, np.ndarray) and len(window_spec) == nperseg:
                self._cached_window = window_spec
            else:
                raise TypeError(f"Unsupported window type or length: {window_spec}")

            final_kwargs["win"] = self._cached_window
            final_kwargs["hop"] = nperseg - final_kwargs["noverlap"]
            final_kwargs["mfft"] = (
                nperseg if final_kwargs["mfft"] is None else final_kwargs["mfft"]
            )
            self._cached_kwargs_full = final_kwargs

        return self._cached_kwargs_full

    def _translate_kwargs(self, kwargs, target_lib="scipy"):
        """
        Translate kwargs for different libraries.

        Args:
            kwargs (dict): Dictionary containing the STFT kwargs
            target_lib (str, optional): Target library. Default is "scipy".

        Returns:
            dict: Dictionary containing the translated kwargs

        Raises:
            ValueError: If the target library is not supported
            TypeError: If the kwargs are not a dictionary
        """
        if not isinstance(kwargs, dict):
            logger.error(f"{log_tag('SPECR','KWARG')} kwargs must be a dictionary, but got {type(kwargs)}")
            raise TypeError("kwargs must be a dictionary")

        translated = kwargs.copy()
        if target_lib == "ssqueezepy":
            if "mfft" in translated:
                translated["n_fft"] = translated.pop("mfft")
            if "hop" in translated:
                translated["hop_len"] = translated.pop("hop")
        elif target_lib == "scipy":
            if "n_fft" in translated:
                translated["mfft"] = translated.pop("n_fft")
            if "hop_len" in translated:
                translated["hop"] = translated.pop("hop_len")
        return translated

    def compute_stft(
        self, signal: np.ndarray, fs: float, t_start: float = 0.0, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT) using scipy.

        This method uses scipy.signal.ShortTimeFFT for STFT computation with automatic
        parameter selection if not provided. The method dynamically determines optimal
        window size, overlap, and FFT length based on signal length.

        Args:
            signal (np.ndarray): Input signal array (1D).
            fs (float): Sampling frequency in Hz.
            t_start (float, optional): Start time offset in seconds. This value is added
                to the time axis. Default is 0.0.
            **kwargs: Additional arguments for STFT:
                - nperseg (int, optional): Length of each segment. If None, automatically
                  determined as ~1/64 of signal length, clamped between 512 and 8192,
                  rounded to nearest power of 2.
                - noverlap (int, optional): Number of points to overlap between segments.
                  If None, automatically determined to achieve time resolution <= 250 μs.
                - mfft (int, optional): Length of FFT. If None, automatically determined
                  to achieve frequency resolution <= 200 kHz.
                - window (str, tuple, or np.ndarray, optional): Window function.
                  Default is ("kaiser", 8).
                - scale_to (str, optional): Scaling mode. Default is "magnitude".
                - phase_shift (float, optional): Phase shift. Default is 0.
                - padding (str, optional): Padding mode. Default is "even".

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - frequencies (np.ndarray): Array of frequencies in Hz.
                - times (np.ndarray): Array of time points in seconds (with t_start offset).
                - STFT_matrix (np.ndarray): Complex STFT matrix with shape (n_freqs, n_times).

        Raises:
            TypeError: If the window type or length is not supported.
            ValueError: If signal is empty or invalid.

        Notes:
            - Window size (nperseg) is automatically optimized for signal length if not provided.
              Defaults to ~1/64 of signal length, clamped between 512 and 8192, rounded to nearest power of 2.
            - FFT length (mfft) is automatically determined to achieve frequency resolution <= 200 kHz.
              Calculated as ceil(fs / 200 kHz) rounded to nearest power of 2, but at least nperseg.
            - Overlap (noverlap) is automatically determined to achieve time resolution <= 250 μs.
              Calculated from hop_length = fs * 250 μs, ensuring at least 25% overlap for smoothness.

        Examples:
            ```python
            # Basic STFT with automatic parameters
            freqs, times, Zxx = analyzer.compute_stft(signal, fs=50e6)
            
            # STFT with custom window size
            freqs, times, Zxx = analyzer.compute_stft(
                signal, fs=50e6,
                nperseg=4096,
                noverlap=2048
            )
            
            # STFT with time offset
            freqs, times, Zxx = analyzer.compute_stft(
                signal, fs=50e6,
                t_start=0.290  # Add trigger time offset
            )
            ```
        """
        logger.debug(f"{log_tag('SPECR','STFT')} Computing STFT with scipy.")

        # Dynamically determine STFT sizing if not provided
        dynamic_kwargs = dict(kwargs)
        
        # Helper function to round to nearest power-of-two
        def _round_pow2(x):
            from math import log2
            p = int(round(log2(x)))
            return max(256, 1 << p)
        
        # Determine nperseg (window size) based on signal length
        if dynamic_kwargs.get("nperseg") is None:
            n = signal.shape[-1]
            # Aim ~1/64 of signal length, clamp between 512 and 8192
            nperseg_dyn = max(1<<9, min(1<<16, max(1<<8, n // 1<<6)))
            nperseg_dyn = _round_pow2(nperseg_dyn)
            dynamic_kwargs["nperseg"] = nperseg_dyn
        
        # Determine mfft (FFT size) based on frequency resolution requirement
        # Target: df <= 200 kHz
        if dynamic_kwargs.get("mfft") is None:
            # Calculate minimum mfft to achieve df <= 200 kHz
            target_df_max = 200e3  # 200 kHz
            mfft_min = int(np.ceil(fs / target_df_max))
            # Round to nearest power-of-two for FFT efficiency
            mfft_dyn = _round_pow2(mfft_min)
            # Ensure mfft is at least nperseg (if nperseg is already set)
            if dynamic_kwargs.get("nperseg") is not None:
                mfft_dyn = max(mfft_dyn, dynamic_kwargs["nperseg"])
            dynamic_kwargs["mfft"] = int(mfft_dyn)
        
        # Determine hop_length based on time resolution requirement
        # Target: dt <= 250 μs
        if dynamic_kwargs.get("noverlap") is None and dynamic_kwargs.get("nperseg") is not None:
            target_dt_max = 250e-6  # 250 μs
            hop_length_target = int(np.floor(fs * target_dt_max))
            nperseg = dynamic_kwargs["nperseg"]
            
            # hop_length cannot exceed nperseg (must be at least 1)
            # If hop_length_target > nperseg, we need to increase nperseg or accept worse time resolution
            # For now, we'll use the maximum hop_length possible (nperseg - 1) and ensure minimum overlap
            hop_length_max = min(hop_length_target, nperseg - 1)
            
            # Calculate noverlap from hop_length
            # hop_length = nperseg - noverlap, so noverlap = nperseg - hop_length
            noverlap_dyn = max(0, nperseg - hop_length_max)
            
            # Ensure reasonable overlap (at least 25% overlap for smoothness)
            min_overlap = int(nperseg * 0.25)
            noverlap_dyn = max(noverlap_dyn, min_overlap)
            
            # Recalculate hop_length from final noverlap to ensure consistency
            final_hop_length = nperseg - noverlap_dyn
            actual_dt = final_hop_length / fs
            
            # Log warning if time resolution requirement cannot be met
            if actual_dt > target_dt_max * 1.1:  # 10% tolerance
                logger.warning(
                    f"{log_tag('SPECR','STFT')} Time resolution {actual_dt*1e6:.2f} μs exceeds target "
                    f"{target_dt_max*1e6:.2f} μs. Consider increasing nperseg or accepting lower resolution."
                )
            
            dynamic_kwargs["noverlap"] = int(noverlap_dyn)

        all_kwargs = self._get_stft_kwargs(**dynamic_kwargs)
        scipy_kwargs = self._translate_kwargs(all_kwargs, "scipy")

        SFT = spsig.ShortTimeFFT(
            scipy_kwargs["win"],
            scipy_kwargs["hop"],
            fs=fs,
            mfft=scipy_kwargs["mfft"],
            dual_win=scipy_kwargs["dual_win"],
            scale_to=scipy_kwargs["scale_to"],
            phase_shift=scipy_kwargs["phase_shift"],
        )
        Zxx = SFT.stft(
            signal, p0=0, p1=None, k_offset=0, padding=scipy_kwargs["padding"], axis=-1
        )
        return SFT.f, SFT.t(signal.shape[-1]) + t_start, Zxx

    def compute_stft_sqpy(
        self, signal: np.ndarray, fs: float, t_start: float = 0.0, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT) using ssqueezepy.

        This method uses ssqueezepy.stft for STFT computation with automatic parameter
        selection if not provided. Similar to compute_stft but uses ssqueezepy library
        which may provide different performance characteristics.

        Args:
            signal (np.ndarray): Input signal array (1D).
            fs (float): Sampling frequency in Hz.
            t_start (float, optional): Start time offset in seconds. This value is added
                to the time axis. Default is 0.0.
            **kwargs: Additional arguments for STFT:
                - nperseg (int, optional): Length of each segment. If None, automatically
                  determined as ~1/64 of signal length, clamped between 512 and 8192,
                  rounded to nearest power of 2.
                - noverlap (int, optional): Number of points to overlap between segments.
                  If None, automatically determined to achieve time resolution <= 250 μs.
                - mfft (int, optional): Length of FFT. If None, automatically determined
                  to achieve frequency resolution <= 200 kHz.
                - window (str, tuple, or np.ndarray, optional): Window function.
                  Default is ("kaiser", 8).
                - padtype (str, optional): Padding type. Default is "reflect".

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - frequencies (np.ndarray): Array of frequencies in Hz (positive frequencies only).
                - times (np.ndarray): Array of time points in seconds (with t_start offset).
                - STFT_matrix (np.ndarray): Complex STFT matrix with shape (n_freqs, n_times).

        Raises:
            ValueError: If the kwargs are not a dictionary or signal is invalid.
            TypeError: If the window type or length is not supported.

        Notes:
            - This method returns only positive frequencies (unlike compute_stft which may
              return full spectrum depending on scipy version).
            - Window size (nperseg) is automatically optimized for signal length if not provided.
              Defaults to ~1/64 of signal length, clamped between 512 and 8192, rounded to nearest power of 2.
            - FFT length (mfft) is automatically determined to achieve frequency resolution <= 200 kHz.
              Calculated as ceil(fs / 200 kHz) rounded to nearest power of 2, but at least nperseg.
            - Overlap (noverlap) is automatically determined to achieve time resolution <= 250 μs.
              Calculated from hop_length = fs * 250 μs, ensuring at least 25% overlap for smoothness.
            - Uses ssqueezepy's STFT implementation which may have different numerical
              characteristics compared to scipy.

        Examples:
            ```python
            # Basic STFT with ssqueezepy
            freqs, times, Sxx = analyzer.compute_stft_sqpy(signal, fs=50e6)
            
            # STFT with custom parameters
            freqs, times, Sxx = analyzer.compute_stft_sqpy(
                signal, fs=50e6,
                nperseg=4096,
                noverlap=2048
            )
            ```
        """
        logger.debug(f"{log_tag('SPECR','SSTFT')} Computing STFT with ssqueezepy.")

        # Dynamically determine STFT sizing if not provided
        dynamic_kwargs = dict(kwargs)
        
        # Helper function to round to nearest power-of-two
        def _round_pow2(x):
            from math import log2
            p = int(round(log2(x)))
            return max(256, 1 << p)
        
        # Determine nperseg (window size) based on signal length
        if dynamic_kwargs.get("nperseg") is None:
            n = signal.shape[-1]
            nperseg_dyn = max(1<<9, min(1<<16, max(1<<8, n // 1<<6)))
            nperseg_dyn = _round_pow2(nperseg_dyn)
            dynamic_kwargs["nperseg"] = nperseg_dyn
        
        # Determine mfft (FFT size) based on frequency resolution requirement
        # Target: df <= 200 kHz
        if dynamic_kwargs.get("mfft") is None:
            # Calculate minimum mfft to achieve df <= 200 kHz
            target_df_max = 200e3  # 200 kHz
            mfft_min = int(np.ceil(fs / target_df_max))
            # Round to nearest power-of-two for FFT efficiency
            mfft_dyn = _round_pow2(mfft_min)
            # Ensure mfft is at least nperseg (if nperseg is already set)
            if dynamic_kwargs.get("nperseg") is not None:
                mfft_dyn = max(mfft_dyn, dynamic_kwargs["nperseg"])
            dynamic_kwargs["mfft"] = int(mfft_dyn)
        
        # Determine hop_length based on time resolution requirement
        # Target: dt <= 250 μs
        if dynamic_kwargs.get("noverlap") is None and dynamic_kwargs.get("nperseg") is not None:
            target_dt_max = 250e-6  # 250 μs
            hop_length_target = int(np.floor(fs * target_dt_max))
            nperseg = dynamic_kwargs["nperseg"]
            
            # hop_length cannot exceed nperseg (must be at least 1)
            # If hop_length_target > nperseg, we need to increase nperseg or accept worse time resolution
            # For now, we'll use the maximum hop_length possible (nperseg - 1) and ensure minimum overlap
            hop_length_max = min(hop_length_target, nperseg - 1)
            
            # Calculate noverlap from hop_length
            # hop_length = nperseg - noverlap, so noverlap = nperseg - hop_length
            noverlap_dyn = max(0, nperseg - hop_length_max)
            
            # Ensure reasonable overlap (at least 25% overlap for smoothness)
            min_overlap = int(nperseg * 0.25)
            noverlap_dyn = max(noverlap_dyn, min_overlap)
            
            # Recalculate hop_length from final noverlap to ensure consistency
            final_hop_length = nperseg - noverlap_dyn
            actual_dt = final_hop_length / fs
            
            # Log warning if time resolution requirement cannot be met
            if actual_dt > target_dt_max * 1.1:  # 10% tolerance
                logger.warning(
                    f"{log_tag('SPECR','SSTFT')} Time resolution {actual_dt*1e6:.2f} μs exceeds target "
                    f"{target_dt_max*1e6:.2f} μs. Consider increasing nperseg or accepting lower resolution."
                )
            
            dynamic_kwargs["noverlap"] = int(noverlap_dyn)

        all_kwargs = self._get_stft_kwargs(**dynamic_kwargs)
        sqpy_kwargs = self._translate_kwargs(all_kwargs, "ssqueezepy")

        Sxx = ssqpy.stft(
            signal,
            window=sqpy_kwargs["win"],
            n_fft=sqpy_kwargs["n_fft"],
            hop_len=sqpy_kwargs["hop_len"],
            fs=fs,
            padtype=sqpy_kwargs["padtype"],
        )

        # Create frequency and time axes manually
        n_fft = sqpy_kwargs["n_fft"]
        hop_len = sqpy_kwargs["hop_len"]
        freqs_stft = np.fft.rfftfreq(n_fft, 1 / fs)  # Only positive frequencies

        # Time axis with hopping consideration and original start time
        n_frames = Sxx.shape[1]  # noqa: F841
        time_stft = np.arange(0, (len(signal) - 1) // hop_len + 1) / fs + t_start

        return freqs_stft, time_stft, Sxx

    def compute_cwt(
        self,
        signal: np.ndarray,
        fs: float,
        wavelet: str = "gmw",
        f_min: float = None,
        f_max: float = None,
        f_center: float = None,
        f_deviation: float = 0.1,
        nv: int = 32,
        scales: str = "log-piecewise",
        decimation_factor: int = 1,
        **kwargs,
    ):
        """
        Compute Continuous Wavelet Transform using ssqueezepy with memory optimization.

        This method computes the CWT of a signal with optional memory optimizations:
        - Decimation: Reduces time-axis data points by downsampling the signal
        - Frequency range limiting: Restricts analysis to a specific frequency range
        - Center frequency mode: Automatically sets frequency range around a center frequency

        Args:
            signal (np.ndarray): Input signal array (1D).
            fs (float): Sampling frequency in Hz.
            wavelet (str, optional): Wavelet type. Default is "gmw" (Generalized Morse Wavelet).
                Other options depend on ssqueezepy support.
            f_min (float, optional): Minimum frequency in Hz. If None and f_center is None,
                uses full frequency range. Default is None.
            f_max (float, optional): Maximum frequency in Hz. If None and f_center is None,
                uses full frequency range. Default is None.
            f_center (float, optional): Center frequency in Hz. If provided, frequency range
                will be automatically set to f_center * (1 ± f_deviation). This overrides
                f_min and f_max. Default is None.
            f_deviation (float, optional): Frequency deviation factor for f_center mode.
                Range will be f_center * (1 ± f_deviation). Default is 0.1 (10% deviation).
            nv (int, optional): Number of voices (wavelets per octave). Higher values provide
                better frequency resolution but increase computation time. Default is 32.
            scales (str, optional): Scale distribution method. Options: 'log', 'log-piecewise',
                'linear', or a numpy array of scales. Default is "log-piecewise".
            decimation_factor (int, optional): Decimation factor to reduce time axis data points.
                Signal will be downsampled by this factor using FIR anti-aliasing filter before CWT.
                This reduces memory usage and computation time. Default is 1 (no decimation).
                Note: After decimation, fs becomes fs / decimation_factor.
            **kwargs: Additional arguments passed to ssqueezepy.cwt.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - freqs_cwt (np.ndarray): Array of frequencies in Hz corresponding to each scale.
                - Wx (np.ndarray): CWT matrix with shape (n_scales, n_times).
                  If decimation_factor > 1, n_times will be reduced accordingly.

        Raises:
            ValueError: If decimation_factor is not a positive integer.
            ValueError: If frequency range is invalid (f_min >= f_max or exceeds Nyquist).
            ValueError: If wavelet type is not supported by ssqueezepy.

        Notes:
            - When using decimation, the effective sampling frequency becomes fs / decimation_factor.
            - Frequency ranges are automatically clamped to valid values (0 to Nyquist).
            - If f_center is provided, it takes precedence over f_min and f_max.
            - The method uses scipy.signal.decimate for anti-aliasing during decimation.

        Examples:
            ```python
            # Basic CWT
            freqs, Wx = analyzer.compute_cwt(signal, fs=50e6)
            
            # CWT with center frequency mode (10% deviation around 15 MHz)
            freqs, Wx = analyzer.compute_cwt(
                signal, fs=50e6,
                f_center=15e6,
                f_deviation=0.1
            )
            
            # CWT with decimation (reduce time points by factor of 4)
            freqs, Wx = analyzer.compute_cwt(
                signal, fs=50e6,
                decimation_factor=4
            )
            
            # Combined: center frequency + decimation
            freqs, Wx = analyzer.compute_cwt(
                signal, fs=50e6,
                f_center=15e6,
                f_deviation=0.1,
                decimation_factor=4
            )
            ```
        """
        logger.debug(
            f"{log_tag('SPECR','SCWT')} Computing CWT with ssqueezepy using '{wavelet}' wavelet."
        )
        
        # Validate decimation_factor
        if decimation_factor < 1 or not isinstance(decimation_factor, int):
            raise ValueError(f"decimation_factor must be a positive integer, got {decimation_factor}")
        
        # Determine frequency range BEFORE decimation (if f_center is provided)
        # This ensures we use the original frequency range even after decimation
        original_fs = fs  # noqa: F841
        original_nyquist = fs / 2  # noqa: F841
        if f_center is not None and f_center > 0:
            # Calculate frequency range around f_center using original fs
            f_range_min = f_center * (1 - f_deviation)
            f_range_max = f_center * (1 + f_deviation)
            logger.info(
                f"{log_tag('SPECR','SCWT')} Using f_center mode: "
                f"f_center={f_center/1e6:.2f} MHz, "
                f"range=[{f_range_min/1e6:.2f}, {f_range_max/1e6:.2f}] MHz "
                f"(±{f_deviation*100:.1f}%)"
            )
            f_min = f_range_min
            f_max = f_range_max
        elif f_min is None or f_max is None:
            # If no frequency range specified, use default (full range)
            logger.debug(
                f"{log_tag('SPECR','SCWT')} No frequency range specified, using full range"
            )
        
        # Apply decimation to reduce time axis data points
        if decimation_factor > 1:
            logger.info(
                f"{log_tag('SPECR','SCWT')} Applying decimation factor {decimation_factor} "
                f"(reducing signal length from {len(signal)} to {len(signal) // decimation_factor})"
            )
            # Use scipy.signal.decimate for anti-aliasing
            from scipy import signal as spsig
            signal = spsig.decimate(signal, decimation_factor, ftype='fir')
            fs = fs / decimation_factor
            logger.debug(
                f"{log_tag('SPECR','SCWT')} Decimated signal: length={len(signal)}, fs={fs/1e6:.2f} MHz"
            )
            
            # Adjust frequency range if it exceeds new Nyquist frequency
            nyquist = fs / 2
            if f_min is not None and f_max is not None:
                if f_max > nyquist:
                    logger.warning(
                        f"{log_tag('SPECR','SCWT')} f_max ({f_max/1e6:.2f} MHz) exceeds decimated "
                        f"Nyquist frequency ({nyquist/1e6:.2f} MHz). "
                        f"Clamping to {nyquist*0.90/1e6:.2f} MHz."
                    )
                    f_max = nyquist * 0.90
                if f_min > nyquist:
                    logger.warning(
                        f"{log_tag('SPECR','SCWT')} f_min ({f_min/1e6:.2f} MHz) exceeds decimated "
                        f"Nyquist frequency ({nyquist/1e6:.2f} MHz). "
                        f"Setting to 05% of Nyquist."
                    )
                    f_min = nyquist * 0.05
        
        wav = ssqpy.Wavelet(wavelet)
        
        # Convert frequency range to scales if provided
        if f_min is not None and f_max is not None:
            # Validate frequency range against current fs (after decimation)
            nyquist = fs / 2
            if f_max > nyquist:
                logger.warning(
                    f"{log_tag('SPECR','SCWT')} f_max ({f_max/1e6:.2f} MHz) exceeds Nyquist "
                    f"frequency ({nyquist/1e6:.2f} MHz). Clamping to 90% of Nyquist."
                )
                f_max = nyquist * 0.90  # Slightly below Nyquist to avoid issues
            if f_min < 0:
                logger.warning(
                    f"{log_tag('SPECR','SCWT')} f_min ({f_min/1e6:.2f} MHz) is negative. "
                    f"Setting to 05% of Nyquist."
                )
                f_min = nyquist * 0.05
            elif f_min >= nyquist:
                logger.warning(
                    f"{log_tag('SPECR','SCWT')} f_min ({f_min/1e6:.2f} MHz) exceeds Nyquist. "
                    f"Setting to 05% of Nyquist."
                )
                f_min = nyquist * 0.05
            
            # Ensure f_min < f_max
            if f_min >= f_max:
                logger.warning(
                    f"{log_tag('SPECR','SCWT')} Invalid frequency range: f_min >= f_max. "
                    f"Using default range."
                )
                f_min = None
                f_max = None
        
        # Convert frequency range to scales if provided
        if f_min is not None and f_max is not None and f_min < f_max:
            # Calculate appropriate scales for frequency range
            # Scale formula: scale = fs / (2 * freq)
            scale_min = fs / (2 * f_max)  # Higher freq -> smaller scale
            scale_max = fs / (2 * f_min)  # Lower freq -> larger scale
            
            # Generate logarithmically spaced scales
            # Number of octaves = log2(scale_max / scale_min)
            # Number of scales = nv * number_of_octaves
            n_octaves = np.log2(scale_max / scale_min)
            if n_octaves <= 0:
                logger.warning(
                    f"{log_tag('SPECR','SCWT')} Invalid octave range ({n_octaves:.2f}). "
                    f"Using default scaling."
                )
                Wx, scales_out = ssqpy.cwt(signal, fs=fs, wavelet=wav, nv=nv, **kwargs)
            else:
                n_scales = max(int(nv * n_octaves), 10)  # Minimum 10 scales
                scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
                
                logger.debug(
                    f"{log_tag('SPECR','SCWT')} Frequency range: [{f_min/1e6:.2f}, {f_max/1e6:.2f}] MHz, "
                    f"Scales: {len(scales)} scales, {n_octaves:.2f} octaves"
                )
                
                Wx, scales_out = ssqpy.cwt(
                    signal, fs=fs, wavelet=wav, scales=scales, **kwargs
                )
        else:
            # Use default scaling
            logger.debug(
                f"{log_tag('SPECR','SCWT')} Using default scaling with nv={nv}"
            )
            Wx, scales_out = ssqpy.cwt(signal, fs=fs, wavelet=wav, nv=nv, **kwargs)

        # Convert scales to frequencies
        freqs_cwt = scale_to_freq(scales_out, wav, len(signal), fs=fs)
        
        logger.info(
            f"{log_tag('SPECR','SCWT')} CWT computed: "
            f"shape={Wx.shape}, freq_range=[{freqs_cwt.min()/1e6:.2f}, {freqs_cwt.max()/1e6:.2f}] MHz"
        )

        return freqs_cwt, Wx

    def find_freq_ridge(
        self,
        Zxx: np.ndarray,
        f: np.ndarray,
        penalty: float = 2,
        n_ridges: int = 1,
        n_bin: int = 15,
        method: str = "stft",
        library: str = "ssqueezepy",
        return_idx: bool = False,
    ) -> np.ndarray:
        """
        Find the frequency ridge(s) in a spectrogram using ridge extraction algorithms.

        This method locates the dominant frequency component(s) over time in a time-frequency
        representation (STFT or CWT). It uses advanced ridge tracking algorithms that consider
        temporal continuity and penalize sudden frequency jumps.

        Args:
            Zxx (np.ndarray): The complex or magnitude matrix from STFT/CWT with shape
                (n_frequencies, n_times). Can be complex or real-valued.
            f (np.ndarray): Array of sample frequencies or scales in Hz (1D array).
            penalty (float, optional): Penalty factor for ridge extraction. Higher values
                penalize sudden frequency jumps more strongly. Typical values: 0.5, 2, 5, 20, 40.
                Default is 2. Higher penalty -> smoother ridges, lower penalty -> more sensitive.
            n_ridges (int, optional): Number of ridges to extract. Default is 1.
                For multiple ridges, use smaller n_bin values (~4).
            n_bin (int, optional): Number of bins for ridge tracking. Larger values provide
                better frequency resolution but may miss rapid changes. Recommended:
                - Single ridge: > 15 (default is 15)
                - Multiple ridges: ~4
                Default is 15.
            method (str, optional): Transform method used. Options: 'stft' or 'cwt'.
                Default is 'stft'.
            library (str, optional): Library to use for ridge extraction.
                Options: 'ssqueezepy' (uses ssqueezepy.extract_ridges) or 'MATLAB'
                (uses MATLAB-compatible extract_fridges). Default is 'ssqueezepy'.
            return_idx (bool, optional): Whether to return frequency indices along with values.
                If True, returns tuple (freq_ridge, ridge_idxs). If False, returns only
                freq_ridge. Default is False.
                
        Returns:
            np.ndarray | Tuple[np.ndarray, np.ndarray]: 
                - If return_idx=False: 1D array of frequencies in Hz for each time point.
                  Shape: (n_times,) for single ridge, (n_ridges, n_times) for multiple ridges.
                - If return_idx=True: Tuple of (freq_ridge, ridge_idxs) where:
                  - freq_ridge: Frequency values in Hz (float64)
                  - ridge_idxs: Frequency indices (int64)

        Raises:
            ValueError: If Zxx is not a 2-D array.
            ValueError: If f is not a 1-D array.
            ValueError: If method is not 'stft' or 'cwt'.
            ValueError: If library is not 'ssqueezepy' or 'MATLAB'.

        Notes:
            - The method uses penalty-based ridge tracking to ensure temporal continuity.
            - For noisy signals, higher penalty values (5-20) work better.
            - For signals with rapid frequency changes, lower penalty values (0.5-2) are preferred.
            - Multiple ridges are extracted in order of power/importance.

        Examples:
            ```python
            # Find single ridge from STFT
            freqs, times, Zxx = analyzer.compute_stft(signal, fs)
            ridge = analyzer.find_freq_ridge(Zxx, freqs, method='stft')
            
            # Find multiple ridges with custom parameters
            ridges = analyzer.find_freq_ridge(
                Zxx, freqs,
                n_ridges=3,
                penalty=5,
                n_bin=4
            )
            
            # Get ridge with indices
            ridge, indices = analyzer.find_freq_ridge(
                Zxx, freqs,
                return_idx=True
            )
            
            # Use MATLAB-compatible algorithm
            ridge = analyzer.find_freq_ridge(
                Zxx, freqs,
                library='MATLAB'
            )
            ```
        """
        logger.debug(f"{log_tag('SPECR','RIDGE')} Finding frequency ridge.")
        if Zxx.ndim != 2:
            raise ValueError(f"`Zxx` must be a 2-D array, but got shape {Zxx.shape}")

        if library == "ssqueezepy":
            ridge_idxs = ssqpy.extract_ridges(
                Zxx,
                f,
                penalty=penalty,
                n_ridges=n_ridges,
                bw=n_bin,
                transform=method.lower(),
            )
        elif library == "MATLAB":
            extract_ridges_matlab = _get_extract_ridges_matlab()
            _, ridge_idxs, _ = extract_ridges_matlab(
                Zxx, f, penalty=penalty, num_ridges=n_ridges, BW=n_bin
            )
        # Map the indices to frequencies
        freq_ridge = f[ridge_idxs]

        return (freq_ridge, ridge_idxs) if return_idx else freq_ridge

    def find_center_frequency_fft(
        self, 
        signal: np.ndarray, 
        fs: float,
        f_range: Optional[Tuple[float, Optional[float]]] = None
    ) -> float:
        """
        Find the center frequency of a signal using FFT peak detection.

        This method computes the FFT of the signal and identifies the frequency with the
        highest magnitude, which is considered the center (dominant) frequency. To avoid
        low-frequency noise and DC components, it filters out frequencies below a threshold
        or uses a custom frequency range.

        Args:
            signal (np.ndarray): Input signal array (1D).
            fs (float): Sampling frequency in Hz.
            f_range (Optional[Tuple[float, Optional[float]]], optional): Custom frequency range
                to search for the center frequency. Tuple format: (min_f, max_f).
                - If None: Uses default threshold (5% of Nyquist or 5 MHz, whichever is smaller)
                  as minimum, and Nyquist frequency as maximum.
                - If max_f is None: Uses Nyquist frequency as maximum.
                - If provided: Only searches within the specified range.
                Default is None.

        Returns:
            float: The estimated center frequency in Hz (frequency with maximum magnitude).

        Raises:
            ValueError: If f_range is invalid (min_f >= max_f or out of bounds).
            ValueError: If no frequencies are found in the specified range.
            ValueError: If signal is empty or invalid.

        Notes:
            - The method uses magnitude (not power) for peak detection.
            - Low frequencies are filtered to avoid DC offset and low-frequency noise.
            - If multiple peaks exist, returns the one with highest magnitude.
            - The frequency resolution depends on signal length: df = fs / n_samples.

        Examples:
            ```python
            # Find center frequency with default threshold
            f_center = analyzer.find_center_frequency_fft(signal, fs=50e6)
            
            # Find center frequency in specific range (10-20 MHz)
            f_center = analyzer.find_center_frequency_fft(
                signal, fs=50e6,
                f_range=(10e6, 20e6)
            )
            
            # Find center frequency above 5 MHz (up to Nyquist)
            f_center = analyzer.find_center_frequency_fft(
                signal, fs=50e6,
                f_range=(5e6, None)
            )
            ```
        """
        logger.debug(f"{log_tag('SPECR','FCENT')} Finding center frequency of signal.")

        # Validate input signal
        try:
            signal = validate_signal(signal, "signal", min_length=2)
        except (ValueError, TypeError):
            # Fallback for backward compatibility if validation fails
            n = len(signal)
            if n == 0:
                logger.warning(
                    f"{log_tag('SPECR','FCENT')} Empty signal provided. Returning 0.0 Hz."
                )
                return 0.0
        else:
            n = len(signal)

        # Validate and process frequency range
        if f_range is not None:
            if len(f_range) != 2:
                raise ValueError("f_range must be a tuple of (min_f, max_f)")
            
            # Set default maximum frequency to Nyquist frequency
            nyquist = fs / 2
            if f_range[1] is None:
                f_range_processed = (
                    validate_frequency(f_range[0], "f_range[0]"), 
                    nyquist
                )
            else:
                f_range_processed = (
                    validate_frequency(f_range[0], "f_range[0]"),
                    validate_frequency(f_range[1], "f_range[1]"),
                )
            
            if f_range_processed[0] >= f_range_processed[1]:
                raise ValueError(
                    f"f_range[0] ({f_range_processed[0]}) must be less than "
                    f"f_range[1] ({f_range_processed[1]})"
                )
            
            f_min = f_range_processed[0]
            f_max = f_range_processed[1]
            
            logger.debug(
                f"{log_tag('SPECR','FCENT')} Using custom frequency range: "
                f"[{f_min/1e6:.2f}, {f_max/1e6:.2f}] MHz"
            )
        else:
            # Use default threshold (backward compatibility)
            nyquist = fs / 2
            f_min = min(nyquist * 0.05, 5e6)
            f_max = nyquist
            
            logger.debug(
                f"{log_tag('SPECR','FCENT')} Using default threshold: "
                f"min={f_min/1e6:.2f} MHz (5% of Nyquist or 5MHz)"
            )

        # Compute FFT
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(n, 1 / fs)

        # We only care about the positive frequencies
        xf_positive = xf[: n // 2]
        yf_positive = 2.0 / n * np.abs(yf[0 : n // 2])

        # Filter to frequency range
        if f_range is not None:
            # Use specified range
            mask = (xf_positive >= f_min) & (xf_positive <= f_max)
            freqs_filtered = xf_positive[mask]
            fft_vals_filtered = yf_positive[mask]
            
            if len(freqs_filtered) == 0:
                raise ValueError(
                    f"No frequencies found in range [{f_min/1e6:.2f}, {f_max/1e6:.2f}] MHz"
                )
            
            peak_idx = np.argmax(fft_vals_filtered)
            f_center = freqs_filtered[peak_idx]
        else:
            # Use default threshold behavior (backward compatibility)
            try:
                search_start_idx = np.where(xf_positive > f_min)[0][0]
            except IndexError:
                # This happens if all frequencies are below the threshold.
                # In this case, just search from the beginning of positive freqs.
                search_start_idx = 0

            # Find the peak frequency above the threshold
            if search_start_idx >= len(yf_positive):
                # If search start index is out of bounds, there's no valid range to search
                logger.warning(
                    f"{log_tag('SPECR','FCENT')} No valid range to search for center frequency. Returning 0.0 Hz."
                )
                return 0.0

            peak_idx = np.argmax(yf_positive[search_start_idx:]) + search_start_idx
            f_center = xf_positive[peak_idx]

        logger.info(f"{log_tag('SPECR','FCENT')} fc: {f_center / 1e6:.2f} MHz")

        return f_center


## Main-guard test code removed. See archived copy in `ifi/olds/spectrum_old_20251030.py`.
