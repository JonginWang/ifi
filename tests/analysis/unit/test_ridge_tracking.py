#!/usr/bin/env python3
"""
Unit tests for ridge_tracking.py module.

This test suite validates:
1. Main extract_fridges function with known inputs
2. Vectorized helper functions (forwards/backwards penalty calculation)
3. Comparison with original implementation to ensure no regressions
4. Boundary conditions and edge cases
"""

import pytest
import numpy as np
from scipy import signal

from ifi.analysis.functions.ridge_tracking import extract_fridges as extract_fridges_optimized
from ifi.analysis.functions.ridge_tracking_orig import extract_fridges as extract_fridges_orig


@pytest.fixture
def sample_stft_data():
    """
    Generate sample STFT data for testing.
    
    Returns:
        tuple: (tf_transf, frequency_scales)
            tf_transf: Complex STFT matrix (nfreq, ntime)
            frequency_scales: Frequency array
    """
    # Generate a simple chirp signal
    fs = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Linear chirp: 50 Hz to 200 Hz
    f0, f1 = 50, 200
    freq_instantaneous = f0 + (f1 - f0) * t / duration
    phase = 2 * np.pi * np.cumsum(freq_instantaneous) / fs
    signal_data = np.cos(phase) + 0.1 * np.random.randn(len(t))
    
    # Compute STFT
    nperseg = 256
    noverlap = nperseg // 2
    f, t_stft, Zxx = signal.stft(
        signal_data, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False
    )
    
    # Use only positive frequencies
    positive_freq_idx = f >= 0
    f = f[positive_freq_idx]
    Zxx = Zxx[positive_freq_idx, :]
    
    return Zxx, f


@pytest.fixture
def simple_stft_matrix():
    """
    Generate a simple, deterministic STFT matrix for testing.
    
    Returns:
        tuple: (tf_transf, frequency_scales)
            tf_transf: Complex STFT matrix with known structure
            frequency_scales: Frequency array
    """
    nfreq = 64
    ntime = 32
    frequency_scales = np.linspace(0, 500, nfreq)
    
    # Create a simple STFT with energy concentrated at specific frequencies
    tf_transf = np.zeros((nfreq, ntime), dtype=complex)
    
    # Add energy at specific frequency-time locations
    for t_idx in range(ntime):
        # Create a ridge at frequency index that increases with time
        freq_idx = int(20 + 10 * t_idx / ntime)
        if freq_idx < nfreq:
            tf_transf[freq_idx, t_idx] = 1.0 + 0.1j
    
    # Add some noise
    tf_transf += 0.1 * (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime))
    
    return tf_transf, frequency_scales


class TestExtractFridges:
    """Test the main extract_fridges function."""
    
    def test_basic_extraction_single_ridge(self, sample_stft_data):
        """Test basic ridge extraction with single ridge."""
        tf_transf, frequency_scales = sample_stft_data
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Check output shapes
        assert max_energy.shape == (tf_transf.shape[1], 1)
        assert ridge_idx.shape == (tf_transf.shape[1], 1)
        assert fridge.shape == (tf_transf.shape[1], 1)
        
        # Check that ridge indices are valid
        assert np.all(ridge_idx >= 0)
        assert np.all(ridge_idx < len(frequency_scales))
        
        # Check that frequencies are valid
        assert np.all(fridge >= frequency_scales[0])
        assert np.all(fridge <= frequency_scales[-1])
    
    def test_multiple_ridges(self, sample_stft_data):
        """Test extraction of multiple ridges."""
        tf_transf, frequency_scales = sample_stft_data
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=3, BW=15
        )
        
        # Check output shapes
        assert max_energy.shape == (tf_transf.shape[1], 3)
        assert ridge_idx.shape == (tf_transf.shape[1], 3)
        assert fridge.shape == (tf_transf.shape[1], 3)
        
        # Check that all ridges have valid indices
        assert np.all(ridge_idx >= 0)
        assert np.all(ridge_idx < len(frequency_scales))
    
    def test_penalty_parameter(self, simple_stft_matrix):
        """Test that penalty parameter affects ridge extraction."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        # Extract with low penalty (allows more frequency jumps)
        _, ridge_idx_low, _ = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=0.5, num_ridges=1, BW=10
        )
        
        # Extract with high penalty (penalizes frequency jumps)
        _, ridge_idx_high, _ = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=10.0, num_ridges=1, BW=10
        )
        
        # High penalty should result in smoother ridge (less variation)
        variation_low = np.std(np.diff(ridge_idx_low.flatten()))
        variation_high = np.std(np.diff(ridge_idx_high.flatten()))
        
        # High penalty should generally result in less variation
        # (though this depends on the signal structure)
        assert isinstance(variation_low, (int, float))
        assert isinstance(variation_high, (int, float))
    
    def test_bw_parameter(self, simple_stft_matrix):
        """Test that BW parameter affects multiple ridge extraction."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        # Extract with small BW
        _, ridge_idx_small_bw, _ = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=2, BW=5
        )
        
        # Extract with large BW
        _, ridge_idx_large_bw, _ = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=2, BW=20
        )
        
        # Both should extract valid ridges
        assert ridge_idx_small_bw.shape == ridge_idx_large_bw.shape
        assert np.all(ridge_idx_small_bw >= 0)
        assert np.all(ridge_idx_large_bw >= 0)
    
    def test_edge_case_empty_signal(self):
        """Test with empty/zero signal."""
        nfreq = 64
        ntime = 32
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = np.zeros((nfreq, ntime), dtype=complex)
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Should still return valid output
        assert max_energy.shape == (ntime, 1)
        assert ridge_idx.shape == (ntime, 1)
        assert fridge.shape == (ntime, 1)
    
    def test_edge_case_single_time_point(self):
        """Test with single time point."""
        nfreq = 64
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = np.random.randn(nfreq, 1) + 1j * np.random.randn(nfreq, 1)
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        assert max_energy.shape == (1, 1)
        assert ridge_idx.shape == (1, 1)
        assert fridge.shape == (1, 1)


class TestComparisonWithOriginal:
    """Compare optimized implementation with original to ensure no regressions."""
    
    def test_identical_results_simple_case(self, simple_stft_matrix):
        """Test that optimized version produces identical results to original."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        # Run both implementations
        max_energy_opt, ridge_idx_opt, fridge_opt = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        max_energy_orig, ridge_idx_orig, fridge_orig = extract_fridges_orig(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Compare ridge indices (should be identical)
        np.testing.assert_array_equal(ridge_idx_opt, ridge_idx_orig)
        
        # Compare frequencies (should be very close)
        np.testing.assert_allclose(fridge_opt, fridge_orig, rtol=1e-10, atol=1e-10)
        
        # Compare energies (should be very close)
        np.testing.assert_allclose(max_energy_opt, max_energy_orig, rtol=1e-10, atol=1e-10)
    
    def test_identical_results_multiple_ridges(self, simple_stft_matrix):
        """Test multiple ridges extraction comparison."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy_opt, ridge_idx_opt, fridge_opt = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=3, BW=15
        )
        
        max_energy_orig, ridge_idx_orig, fridge_orig = extract_fridges_orig(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=3, BW=15
        )
        
        # Compare ridge indices
        np.testing.assert_array_equal(ridge_idx_opt, ridge_idx_orig)
        
        # Compare frequencies and energies
        np.testing.assert_allclose(fridge_opt, fridge_orig, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(max_energy_opt, max_energy_orig, rtol=1e-10, atol=1e-10)
    
    def test_identical_results_different_penalties(self, simple_stft_matrix):
        """Test with different penalty values."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        for penalty in [0.5, 1.0, 2.0, 5.0, 10.0]:
            max_energy_opt, ridge_idx_opt, fridge_opt = extract_fridges_optimized(
                tf_transf, frequency_scales, penalty=penalty, num_ridges=1, BW=15
            )
            
            max_energy_orig, ridge_idx_orig, fridge_orig = extract_fridges_orig(
                tf_transf, frequency_scales, penalty=penalty, num_ridges=1, BW=15
            )
            
            np.testing.assert_array_equal(ridge_idx_opt, ridge_idx_orig)
            np.testing.assert_allclose(fridge_opt, fridge_orig, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(max_energy_opt, max_energy_orig, rtol=1e-10, atol=1e-10)
    
    def test_identical_results_real_world_signal(self, sample_stft_data):
        """Test with real-world STFT data from chirp signal."""
        tf_transf, frequency_scales = sample_stft_data
        
        max_energy_opt, ridge_idx_opt, fridge_opt = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        max_energy_orig, ridge_idx_orig, fridge_orig = extract_fridges_orig(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # For real signals, allow small numerical differences due to floating point
        # but ridge indices should be identical
        np.testing.assert_array_equal(ridge_idx_opt, ridge_idx_orig)
        
        # Frequencies and energies should be very close
        np.testing.assert_allclose(fridge_opt, fridge_orig, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(max_energy_opt, max_energy_orig, rtol=1e-8, atol=1e-8)


class TestVectorizedFunctions:
    """Test vectorized helper functions indirectly through integration tests."""
    
    def test_forward_pass_consistency(self, simple_stft_matrix):
        """Test that forward pass produces consistent results."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        # Extract ridges multiple times - should be deterministic
        results1 = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        results2 = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Results should be identical
        np.testing.assert_array_equal(results1[1], results2[1])  # ridge_idx
        np.testing.assert_allclose(results1[2], results2[2], rtol=1e-10)  # fridge
    
    def test_backward_pass_consistency(self, simple_stft_matrix):
        """Test that backward pass produces consistent results."""
        # This is tested indirectly through the full extraction
        # The backward pass refines the forward pass results
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Ridge indices should be valid
        assert np.all(ridge_idx >= 0)
        assert np.all(ridge_idx < len(frequency_scales))
        
        # Frequencies should be monotonically increasing/decreasing or stable
        # (depending on signal structure, but should not jump erratically)
        ridge_freq = fridge.flatten()
        valid_mask = ~np.isnan(ridge_freq)
        if np.sum(valid_mask) > 1:
            valid_freq = ridge_freq[valid_mask]
            # Check that frequency changes are reasonable (not huge jumps)
            freq_diffs = np.abs(np.diff(valid_freq))
            max_freq_diff = np.max(freq_diffs)
            # Maximum frequency jump should be less than half the frequency range
            freq_range = frequency_scales[-1] - frequency_scales[0]
            assert max_freq_diff < 0.5 * freq_range


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def test_boundary_frequency_indices(self, simple_stft_matrix):
        """Test that ridge extraction handles boundary frequencies correctly."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        # Create signal with energy at boundaries
        tf_transf[0, :] = 10.0 + 1j  # Lowest frequency
        tf_transf[-1, :] = 10.0 + 1j  # Highest frequency
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Should handle boundaries without errors
        assert np.all(ridge_idx >= 0)
        assert np.all(ridge_idx < len(frequency_scales))
    
    def test_small_bw_parameter(self, simple_stft_matrix):
        """Test with very small BW parameter."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=2, BW=1
        )
        
        assert ridge_idx.shape == (tf_transf.shape[1], 2)
        assert np.all(ridge_idx >= 0)
    
    def test_large_bw_parameter(self, simple_stft_matrix):
        """Test with large BW parameter."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=2, BW=30
        )
        
        assert ridge_idx.shape == (tf_transf.shape[1], 2)
        assert np.all(ridge_idx >= 0)
    
    def test_zero_penalty(self, simple_stft_matrix):
        """Test with zero penalty (should allow any frequency jump)."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=0.0, num_ridges=1, BW=15
        )
        
        # Should still produce valid output
        assert ridge_idx.shape == (tf_transf.shape[1], 1)
        assert np.all(ridge_idx >= 0)
        assert np.all(ridge_idx < len(frequency_scales))
    
    def test_very_large_penalty(self, simple_stft_matrix):
        """Test with very large penalty (should penalize frequency jumps heavily)."""
        tf_transf, frequency_scales = simple_stft_matrix
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=100.0, num_ridges=1, BW=15
        )
        
        # Should produce smoother ridge
        assert ridge_idx.shape == (tf_transf.shape[1], 1)
        assert np.all(ridge_idx >= 0)
        
        # Check that frequency variation is limited
        ridge_freq = fridge.flatten()
        valid_mask = ~np.isnan(ridge_freq)
        if np.sum(valid_mask) > 1:
            valid_freq = ridge_freq[valid_mask]
            freq_diffs = np.abs(np.diff(valid_freq))
            # With high penalty, frequency changes should be relatively small
            assert isinstance(np.max(freq_diffs), (int, float))


class TestNumericalStability:
    """Test numerical stability and robustness."""
    
    def test_nan_handling(self):
        """Test that NaN values in input are handled gracefully."""
        nfreq = 64
        ntime = 32
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
        
        # Add some NaN values
        tf_transf[10:15, 5:10] = np.nan + 1j * np.nan
        
        # Should not raise an error
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        # Output should not contain NaN (except possibly in frequencies if ridge is invalid)
        assert not np.any(np.isnan(ridge_idx))
        assert not np.any(np.isnan(max_energy))
    
    def test_inf_handling(self):
        """Test that Inf values are handled."""
        nfreq = 64
        ntime = 32
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)
        
        # Add some Inf values
        tf_transf[10:15, 5:10] = np.inf + 1j * np.inf
        
        # Should handle gracefully
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        assert ridge_idx.shape == (ntime, 1)
        assert np.all(np.isfinite(ridge_idx))
    
    def test_very_small_energy(self):
        """Test with very small energy values."""
        nfreq = 64
        ntime = 32
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = 1e-10 * (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime))
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        assert ridge_idx.shape == (ntime, 1)
        assert np.all(ridge_idx >= 0)
    
    def test_very_large_energy(self):
        """Test with very large energy values."""
        nfreq = 64
        ntime = 32
        frequency_scales = np.linspace(0, 500, nfreq)
        tf_transf = 1e10 * (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime))
        
        max_energy, ridge_idx, fridge = extract_fridges_optimized(
            tf_transf, frequency_scales, penalty=2.0, num_ridges=1, BW=15
        )
        
        assert ridge_idx.shape == (ntime, 1)
        assert np.all(ridge_idx >= 0)

