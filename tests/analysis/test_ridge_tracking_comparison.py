#!/usr/bin/env python3
"""
Ridge Tracking Algorithm Comparison Test
========================================

This script compares three implementations of ridge extraction:
1. ridge_tracking_orig.py: Original MATLAB-based implementation (double loops)
2. ridge_tracking.py: Vectorized optimization
3. ssqueezepy.ridge_extraction: Numba JIT optimized version

It generates multi-chirp signals and visualizes the extracted ridges.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Import ridge tracking implementations
from ifi.analysis.functions.ridge_tracking import extract_fridges as extract_fridges_optimized
from ifi.analysis.functions.ridge_tracking_orig import extract_fridges as extract_fridges_orig

try:
    from ssqueezepy.ridge_extraction import extract_ridges as extract_ridges_ssqueeze
    SSQUEEZE_AVAILABLE = True
except ImportError:
    SSQUEEZE_AVAILABLE = False
    print("Warning: ssqueezepy not available. Skipping ssqueezepy comparison.")


def generate_multi_chirp_signal(
    duration=1.0,
    sampling_rate=1000,
    chirp_params=None,
    noise_level=0.1,
    random_seed=42
):
    """
    Generate a multi-component chirp signal.
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling frequency in Hz
        chirp_params: List of dicts with keys:
            - 'f0': Start frequency (Hz)
            - 'f1': End frequency (Hz)
            - 'amplitude': Signal amplitude
            - 't_start': Start time (s)
            - 't_end': End time (s)
        noise_level: Standard deviation of white noise
        random_seed: Random seed for reproducibility
        
    Returns:
        t: Time array
        signal: Generated signal
        true_ridges: List of true frequency ridges (for validation)
    """
    if chirp_params is None:
        # Default: 3 overlapping chirps
        chirp_params = [
            {'f0': 50, 'f1': 200, 'amplitude': 1.0, 't_start': 0.0, 't_end': 0.5},
            {'f0': 100, 'f1': 300, 'amplitude': 0.8, 't_start': 0.2, 't_end': 0.7},
            {'f0': 150, 'f1': 250, 'amplitude': 0.6, 't_start': 0.4, 't_end': 1.0},
        ]
    
    np.random.seed(random_seed)
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal_data = np.zeros_like(t)
    true_ridges = []
    
    for params in chirp_params:
        t_start_idx = int(params['t_start'] * sampling_rate)
        t_end_idx = int(params['t_end'] * sampling_rate)
        t_chirp = t[t_start_idx:t_end_idx]
        
        # Linear chirp: f(t) = f0 + (f1 - f0) * (t - t_start) / (t_end - t_start)
        if len(t_chirp) > 0:
            freq_instantaneous = params['f0'] + (params['f1'] - params['f0']) * (
                (t_chirp - params['t_start']) / (params['t_end'] - params['t_start'])
            )
            
            # Generate chirp signal
            phase = 2 * np.pi * np.cumsum(freq_instantaneous) / sampling_rate
            chirp_signal = params['amplitude'] * np.cos(phase)
            
            signal_data[t_start_idx:t_end_idx] += chirp_signal
            
            # Store true ridge for visualization
            true_ridge = np.full(len(t), np.nan)
            true_ridge[t_start_idx:t_end_idx] = freq_instantaneous
            true_ridges.append(true_ridge)
    
    # Add white noise
    noise = np.random.normal(0, noise_level, len(signal_data))
    signal_data += noise
    
    return t, signal_data, true_ridges


def compute_stft_for_ridge_extraction(signal_data, sampling_rate, nperseg=256, noverlap=None):
    """
    Compute STFT for ridge extraction.
    
    Args:
        signal_data: Input signal
        sampling_rate: Sampling frequency
        nperseg: Length of each segment
        noverlap: Number of overlapping points
        
    Returns:
        f: Frequency array
        t: Time array
        Zxx: STFT matrix (complex)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    f, t, Zxx = signal.stft(
        signal_data,
        sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )
    
    # Use only positive frequencies
    positive_freq_idx = f >= 0
    f = f[positive_freq_idx]
    Zxx = Zxx[positive_freq_idx, :]
    
    return f, t, Zxx


def compare_ridge_extraction_methods(
    tf_transf,
    frequency_scales,
    num_ridges=3,
    penalty=2.0,
    BW=15
):
    """
    Compare three ridge extraction methods.
    
    Returns:
        results: Dict with results from each method
        timings: Dict with execution times
    """
    results = {}
    timings = {}
    
    # Method 1: Original (double loops)
    print("Running original implementation...")
    start_time = time.time()
    max_energy_orig, ridge_idx_orig, fridge_orig = extract_fridges_orig(
        tf_transf, frequency_scales, penalty=penalty, num_ridges=num_ridges, BW=BW
    )
    timings['orig'] = time.time() - start_time
    results['orig'] = {
        'max_energy': max_energy_orig,
        'ridge_idx': ridge_idx_orig,
        'fridge': fridge_orig
    }
    
    # Method 2: Optimized (vectorized)
    print("Running optimized implementation...")
    start_time = time.time()
    max_energy_opt, ridge_idx_opt, fridge_opt = extract_fridges_optimized(
        tf_transf, frequency_scales, penalty=penalty, num_ridges=num_ridges, BW=BW
    )
    timings['optimized'] = time.time() - start_time
    results['optimized'] = {
        'max_energy': max_energy_opt,
        'ridge_idx': ridge_idx_opt,
        'fridge': fridge_opt
    }
    
    # Method 3: ssqueezepy (if available)
    if SSQUEEZE_AVAILABLE:
        print("Running ssqueezepy implementation...")
        start_time = time.time()
        ridge_output = extract_ridges_ssqueeze(
            tf_transf,
            frequency_scales,
            penalty=penalty,
            n_ridges=num_ridges,
            bw=BW,
            transform='stft',
            get_params=True,
            parallel=True
        )
        timings['ssqueezepy'] = time.time() - start_time
        
        if isinstance(ridge_output, tuple):
            ridge_idx_ssq, fridge_ssq, max_energy_ssq = ridge_output
        else:
            # If get_params=False, compute frequencies and energies manually
            ridge_idx_ssq = ridge_output
            energy = np.abs(tf_transf)**2
            n_timeshifts = tf_transf.shape[1]
            fridge_ssq = np.zeros((n_timeshifts, num_ridges))
            max_energy_ssq = np.zeros((n_timeshifts, num_ridges))
            for i in range(num_ridges):
                if ridge_idx_ssq.ndim > 1:
                    fridge_ssq[:, i] = frequency_scales[ridge_idx_ssq[:, i]]
                    max_energy_ssq[:, i] = energy[ridge_idx_ssq[:, i], np.arange(n_timeshifts)]
                else:
                    fridge_ssq[:, i] = frequency_scales[ridge_idx_ssq]
                    max_energy_ssq[:, i] = energy[ridge_idx_ssq, np.arange(n_timeshifts)]
        
        # Ensure consistent shape: (n_timeshifts, n_ridges)
        if ridge_idx_ssq.ndim == 1:
            ridge_idx_ssq = ridge_idx_ssq[:, np.newaxis]
        if fridge_ssq.ndim == 1:
            fridge_ssq = fridge_ssq[:, np.newaxis]
        if max_energy_ssq.ndim == 1:
            max_energy_ssq = max_energy_ssq[:, np.newaxis]
        
        results['ssqueezepy'] = {
            'max_energy': max_energy_ssq,
            'ridge_idx': ridge_idx_ssq,
            'fridge': fridge_ssq
        }
    
    return results, timings


def visualize_results(
    t,
    signal_data,
    f,
    t_stft,
    tf_transf,
    results,
    timings,
    true_ridges=None,
    save_path=None
):
    """
    Visualize ridge extraction results.
    
    Args:
        t: Time array for signal
        signal_data: Original signal
        f: Frequency array
        t_stft: Time array for STFT
        tf_transf: STFT matrix
        results: Dict with extraction results
        timings: Dict with execution times
        true_ridges: List of true frequency ridges
        save_path: Path to save figure
    """
    n_methods = len(results)
    fig = plt.figure(figsize=(16, 4 * (n_methods + 1)))
    
    # Plot 1: Original signal
    ax1 = plt.subplot(n_methods + 1, 1, 1)
    ax1.plot(t, signal_data, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Multi-Chirp Signal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-N: STFT with extracted ridges for each method
    method_names = list(results.keys())
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for idx, method_name in enumerate(method_names):
        ax = plt.subplot(n_methods + 1, 1, idx + 2)
        
        # Plot STFT magnitude
        magnitude = np.abs(tf_transf)
        im = ax.pcolormesh(
            t_stft, f, 20 * np.log10(magnitude + 1e-10),
            shading='gouraud', cmap='viridis'
        )
        plt.colorbar(im, ax=ax, label='Magnitude [dB]')
        
        # Plot extracted ridges
        fridge = results[method_name]['fridge']
        n_ridges = fridge.shape[1] if fridge.ndim > 1 else 1
        
        for ridge_idx in range(n_ridges):
            if fridge.ndim > 1:
                ridge_freq = fridge[:, ridge_idx]
            else:
                ridge_freq = fridge
            
            # Ensure ridge_freq has same length as t_stft
            if len(ridge_freq) != len(t_stft):
                # Interpolate if needed
                from scipy.interpolate import interp1d
                t_orig = np.linspace(0, 1, len(ridge_freq))
                interp_func = interp1d(
                    t_orig, ridge_freq, kind='linear',
                    bounds_error=False, fill_value=np.nan
                )
                ridge_freq = interp_func(np.linspace(0, 1, len(t_stft)))
            
            # Remove NaN values for plotting
            valid_mask = ~np.isnan(ridge_freq)
            if np.any(valid_mask) and len(ridge_freq) == len(t_stft):
                ax.plot(
                    t_stft[valid_mask],
                    ridge_freq[valid_mask],
                    color=colors[ridge_idx % len(colors)],
                    linewidth=2,
                    label=f'Ridge {ridge_idx + 1}',
                    linestyle='--'
                )
        
        # Plot true ridges if available
        if true_ridges is not None:
            for true_ridge in true_ridges:
                valid_mask = ~np.isnan(true_ridge)
                if np.any(valid_mask):
                    # Interpolate to STFT time grid
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(
                        t[valid_mask],
                        true_ridge[valid_mask],
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    true_ridge_stft = interp_func(t_stft)
                    valid_stft = ~np.isnan(true_ridge_stft)
                    if np.any(valid_stft):
                        ax.plot(
                            t_stft[valid_stft],
                            true_ridge_stft[valid_stft],
                            color='white',
                            linewidth=1.5,
                            alpha=0.7,
                            label='True Ridge',
                            linestyle=':'
                        )
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        timing_str = f" ({timings[method_name]:.4f}s)" if method_name in timings else ""
        ax.set_title(f'{method_name.capitalize()} Method{timing_str}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def main():
    """Main test function."""
    print("=" * 80)
    print("Ridge Tracking Algorithm Comparison Test")
    print("=" * 80)
    
    # Generate multi-chirp signal
    print("\n1. Generating multi-chirp signal...")
    t, signal_data, true_ridges = generate_multi_chirp_signal(
        duration=1.0,
        sampling_rate=1000,
        noise_level=0.05,
        chirp_params=[
            {'f0': 50, 'f1': 200, 'amplitude': 1.0, 't_start': 0.0, 't_end': 0.5},
            {'f0': 100, 'f1': 300, 'amplitude': 0.8, 't_start': 0.2, 't_end': 0.7},
            {'f0': 150, 'f1': 250, 'amplitude': 0.6, 't_start': 0.4, 't_end': 1.0},
        ]
    )
    print(f"   Signal length: {len(signal_data)} samples")
    print(f"   Number of chirp components: {len(true_ridges)}")
    
    # Compute STFT
    print("\n2. Computing STFT...")
    f, t_stft, Zxx = compute_stft_for_ridge_extraction(
        signal_data, sampling_rate=1000, nperseg=256, noverlap=128
    )
    print(f"   STFT shape: {Zxx.shape}")
    print(f"   Frequency range: {f[0]:.2f} - {f[-1]:.2f} Hz")
    print(f"   Time range: {t_stft[0]:.2f} - {t_stft[-1]:.2f} s")
    
    # Compare ridge extraction methods
    print("\n3. Comparing ridge extraction methods...")
    results, timings = compare_ridge_extraction_methods(
        Zxx,
        f,
        num_ridges=3,
        penalty=2.0,
        BW=15
    )
    
    # Print timing results
    print("\n4. Performance Comparison:")
    print("-" * 80)
    for method_name, timing in timings.items():
        print(f"   {method_name:15s}: {timing:.6f} seconds")
    if len(timings) > 1:
        baseline = min(timings.values())
        print("\n   Speedup relative to fastest:")
        for method_name, timing in timings.items():
            speedup = baseline / timing
            print(f"   {method_name:15s}: {speedup:.2f}x")
    
    # Visualize results
    print("\n5. Visualizing results...")
    output_dir = Path(project_root) / "tests" / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "ridge_tracking_comparison.png"
    
    fig = visualize_results(
        t, signal_data, f, t_stft, Zxx, results, timings,
        true_ridges=true_ridges, save_path=save_path
    )
    
    print("\n6. Test completed successfully!")
    print(f"   Results saved to: {save_path}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()

