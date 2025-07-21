import numpy as np
from scipy.signal import get_window, ShortTimeFFT, cwt, ricker, find_peaks
import pywt
from typing import Tuple
from ssqueezepy import stft

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

    @assign_kwargs(['window', 'nperseg', 'noverlap', 'mfft', 'dual_win', 'scale_to', 'phase_shift'])
    def compute_stft(self, signal: np.ndarray, fs: float, window: str, 
                     nperseg: int, noverlap: int, mfft: int, 
                     dual_win, scale_to: str, phase_shift: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Short-Time Fourier Transform (STFT) of a signal.

        Args:
            signal (np.ndarray): The input signal.
            fs (float): The sampling frequency.
            window (str): The window function to use.
            nperseg (int): Length of each segment.
            noverlap (int): Number of points to overlap between segments.
            mfft (int): The number of points for the FFT. Defaults to nperseg.
            dual_win: The dual window for the synthesis (inverse) STFT.
            scale_to (str): The scaling of the STFT ('magnitude' or 'psd').
            phase_shift (int): The phase shift applied to the window.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, times, and the complex STFT matrix.
        """
        win = get_window(window, nperseg)
        hop = nperseg - noverlap
        mfft = nperseg if mfft is None else mfft

        SFT = ShortTimeFFT(win, hop, fs=fs, mfft=mfft, dual_win=dual_win, scale_to=scale_to, phase_shift=phase_shift)
        f, t, Zxx = SFT.stft(signal)
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
        Finds the frequency "ridge" in a spectrogram (abs(Zxx)).

        Args:
            Zxx (np.ndarray): The complex matrix from the STFT.
            f (np.ndarray): The array of sample frequencies.
            prominence (float, optional): Required prominence of peaks.
            distance (float, optional): Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.

        Returns:
            np.ndarray: The frequency ridge.
        """
        # Get the magnitude squared (power)
        power_spectrum = np.abs(Zxx)**2
        
        # Find the peaks in the power spectrum
        peaks, _ = find_peaks(power_spectrum, prominence=prominence, distance=distance)
        
        # Map the indices to frequencies
        frequency_ridge = f[peaks]
        
        return frequency_ridge

if __name__ == '__main__':
    # Example usage:
    analyzer = SpectrumAnalysis()
    fs = 1e7
    t = np.arange(0, 1, 1/fs)
    signal = np.sin(2 * np.pi * 1e6 * t)
    
    # Call with default parameters using SciPy
    f, t_stft, Zxx = analyzer.compute_stft(signal, fs)
    ridge = analyzer.find_freq_ridge(Zxx, f)
    print("--- SciPy STFT ---")
    print(f"STFT shape: {Zxx.shape}")
    print(f"Ridge frequency: {np.mean(ridge)}")
    
    # Call using ssqueezepy
    f_ssq, t_ssq, Sxx = analyzer.compute_stft_ssq(signal, fs, n_fft=1024, hop_len=512)
    ridge_ssq = analyzer.find_freq_ridge(Sxx, f_ssq)
    print("\n--- ssqueezepy STFT ---")
    print(f"STFT shape: {Sxx.shape}")
    print(f"Ridge frequency: {np.mean(ridge_ssq)}")
    pass
