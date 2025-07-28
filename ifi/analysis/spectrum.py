import numpy as np
from scipy.signal import get_window, ShortTimeFFT, cwt, ricker, find_peaks
import pywt
from typing import Tuple, Union
from ssqueezepy import stft
import logging

from ifi.utils import assign_kwargs


class SpectrumAnalysis:
    def __init__(self):
        self.kwargs_fallback = {
            'window': ('kaiser', 8),
            'nperseg': 10000,
            'noverlap': 5000,
            'mfft': 10000,
            'dual_win': None,
            'scale_to': 'magnitude',
            'phase_shift': 0
        }

    # @assign_kwargs(['window', 'nperseg', 'noverlap', 'mfft', 'dual_win', 'scale_to', 'phase_shift'])
    # def compute_stft(self, signal: np.ndarray, fs: float, window: Union[str, tuple, np.ndarray], 
    #                  nperseg: int, noverlap: int, mfft: int, 
    #                  dual_win, scale_to: str, phase_shift: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def compute_stft(self, signal: np.ndarray, fs: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Short-Time Fourier Transform (STFT) of a signal.
        
        ###### Revise version 0.1 ######
        Args:
            signal (np.ndarray): The input signal.
            fs (float): The sampling frequency.
            window (str, tuple, or np.ndarray): The window function to use.
                - If str or tuple, it's passed to `scipy.signal.get_window`.
                - If np.ndarray, it's used directly as the window.
            nperseg (int): Length of each segment.
            noverlap (int): Number of points to overlap between segments.
            mfft (int): The number of points for the FFT. Defaults to nperseg.
            dual_win: The dual window for the synthesis (inverse) STFT.
            scale_to (str): The scaling of the STFT ('magnitude' or 'psd').
            phase_shift (int): The phase shift applied to the window.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, times, and the complex STFT matrix.
        
        ###### Revise version 0.2 ######
        # Reviewed 2025-07-22
        Accepts keyword arguments to override defaults defined in `self.kwargs_fallback`.
        """
        # Merge provided kwargs with fallback defaults
        final_kwargs = self.kwargs_fallback.copy()
        final_kwargs.update(kwargs)

        window = final_kwargs['window']
        nperseg = final_kwargs['nperseg']
        noverlap = final_kwargs['noverlap']
        mfft = final_kwargs['mfft']
        dual_win = final_kwargs['dual_win']
        scale_to = final_kwargs['scale_to']
        phase_shift = final_kwargs['phase_shift']
        
        if isinstance(window, (str, tuple)):
            win = get_window(window, nperseg)
        elif isinstance(window, np.ndarray):
            # If the window is already a numpy array, use it directly.
            if len(window) > 1 and len(window) != nperseg:
                raise ValueError(f"Provided window is length {len(window)}. To use Kaiser window, put a number (e.g., 8) for the beta parameter. To use a pre-designed filter, use a numpy array of length {nperseg}.")
            win = window
        else:
            raise TypeError(f"Unsupported window type: {type(window)}. Must be str, tuple, or np.ndarray.")

        hop = nperseg - noverlap
        mfft = nperseg if mfft is None else mfft

        SFT = ShortTimeFFT(win, hop, fs=fs, mfft=mfft, dual_win=dual_win, scale_to=scale_to, phase_shift=phase_shift)
        
        # The stft method of a ShortTimeFFT object returns only the Zxx matrix.
        # Frequencies (f) and times (t) are properties of the object itself.
        Zxx = SFT.stft(signal)
        f = SFT.f
        t = SFT.t(signal.shape[-1])
        # ShortTimeFFT
        # f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        # nfft=mfft, return_onesided=True if fft_mode is 'onesided' else False)
        return f, t, Zxx

    def compute_stft_ssq(self, signal: np.ndarray, fs: float, window: str = 'hann',
                         n_fft: int = 1024, hop_len: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Short-Time Fourier Transform (STFT) of a signal using the ssqueezepy library.

        Args:
            signal (np.ndarray): The input signal.
            fs (float): The sampling frequency.
            window (str): The window function to use. Defaults to 'hann'.
            n_fft (int): Length of each segment. Defaults to 1024.
            hop_len (int): Hop length between segments. Defaults to 512.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, times, and the complex STFT matrix.
        """
        # ssqueezepy returns STFT, frequencies, and times in a different order.
        Sxx, f, t = stft(signal, window=window, n_fft=n_fft, hop_len=hop_len, fs=fs)
        return f, t, Sxx

    def compute_cwt(self, signal, fs, widths=None):
        """
        Computes the Continuous Wavelet Transform (CWT) of a signal.

        Args:
            signal (np.ndarray): The input signal (1D array).
            fs (float): The sampling frequency of the signal.
            widths (np.ndarray, optional): The widths to use for the CWT.
                                           If None, a default range is used.

        Returns:
            tuple: A tuple containing:
                - cwt_matrix (np.ndarray): The CWT matrix.
                - freqs (np.ndarray): The corresponding frequencies.
        """
        from scipy.signal import cwt, ricker

        if widths is None:
            widths = np.arange(1, 101)

        cwt_matrix = cwt(signal, ricker, widths)

        # For the ricker wavelet, the center frequency is approx. fs / (2 * width * pi)
        # A common approximation for plotting is f = fs / width
        # A slightly better one for ricker is:
        freqs = (5 * fs) / (2 * np.pi * widths)

        return cwt_matrix, freqs

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
    fs = 1e7
    t = np.arange(0, 1, 1/fs)
    signal = np.sin(2 * np.pi * 1e6 * t)
    
    # Call with default parameters using SciPy
    f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
    ridge = analyzer.find_freq_ridge(Zxx, f)
    logging.info("--- SciPy STFT ---")
    logging.info(f"STFT shape: {Zxx.shape}")
    logging.info(f"Ridge frequency: {np.mean(ridge)}")
    
    # Call using ssqueezepy
    f_ssq, t_ssq, Sxx = analyzer.compute_stft_ssq(signal, fs, n_fft=1024, hop_len=512)
    ridge_ssq = analyzer.find_freq_ridge(Sxx, f_ssq)
    logging.info("--- ssqueezepy STFT ---")
    logging.info(f"STFT shape: {Sxx.shape}")
    logging.info(f"Ridge frequency: {np.mean(ridge_ssq)}")
    pass
