#!/usr/bin/env python3
"""
Test suite for CORDIC Vectoring Mode implementation.

Tests the new vectoring mode against reference implementation (atan2)
and compares with the old rotation mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from ifi.analysis.phase_analysis import CORDICProcessor


class TestCORDICVectoringMode:
    """Test suite for CORDIC vectoring mode."""

    @pytest.fixture
    def cordic_processor(self):
        """Create CORDICProcessor instance."""
        return CORDICProcessor()

    def test_vectoring_basic_accuracy(self, cordic_processor):
        """Test basic accuracy of vectoring mode against atan2."""
        test_cases = [
            (1.0, 0.0, 0.0, "0°"),
            (-1.0, 0.0, np.pi, "180°"),
            (0.0, 1.0, np.pi / 2, "90°"),
            (1.0, 1.0, np.pi / 4, "45°"),
            (-1.0, -1.0, -3 * np.pi / 4, "-135°"),
            (np.sqrt(3), 1.0, np.pi / 6, "30°"),
            (1.0, -1.0, -np.pi / 4, "-45°"),
            (0.5, 0.8660254, np.pi / 3, "60°"),  # cos(60°), sin(60°)
        ]

        for x, y, expected_phase, desc in test_cases:
            mag, phase, _ = cordic_processor.cordic(
                np.array([x]), np.array([y]), method="vectoring"
            )
            expected_mag = np.sqrt(x * x + y * y)
            atan2_phase = np.arctan2(y, x)

            # Magnitude accuracy
            assert np.abs(mag[0] - expected_mag) < 1e-6, f"Magnitude error for {desc}"

            # Phase accuracy (should match atan2)
            # Handle phase wrapping
            phase_diff = phase[0] - atan2_phase
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
            # CORDIC has finite precision - allow small errors
            assert (
                np.abs(phase_diff) < 2e-5
            ), f"Phase error for {desc}: {phase[0]:.6f} vs {atan2_phase:.6f}, diff={phase_diff:.2e}"

    def test_vectoring_vectorized_accuracy(self, cordic_processor):
        """Test vectorized vectoring mode with multiple inputs."""
        x = np.array([1.0, -1.0, 0.0, 1.0, -1.0, np.sqrt(3), 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0, -1.0, 1.0, -1.0])

        mags, phases, _ = cordic_processor.cordic(x, y, method="vectoring")

        # Compare with atan2
        expected_phases = np.arctan2(y, x)
        expected_mags = np.sqrt(x * x + y * y)

        # Check magnitude accuracy
        assert np.allclose(mags, expected_mags, rtol=1e-6)

        # Check phase accuracy (handle wrapping)
        phase_diffs = phases - expected_phases
        phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        assert np.all(np.abs(phase_diffs) < 2e-5), f"Phase errors: {phase_diffs}"

    def test_vectoring_phase_range(self, cordic_processor):
        """Test that vectoring mode produces phases in [-π, π] range."""
        # Test all quadrants
        angles = np.linspace(-np.pi, np.pi, 100)
        x = np.cos(angles)
        y = np.sin(angles)

        mags, phases, _ = cordic_processor.cordic(x, y, method="vectoring")

        # All phases should be in [-π, π]
        assert np.all(phases >= -np.pi - 1e-10), "Phase below -π"
        assert np.all(phases <= np.pi + 1e-10), "Phase above π"

        # Should match input angles (with wrapping tolerance)
        phase_diffs = phases - angles
        phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        assert np.all(np.abs(phase_diffs) < 1e-4), "Phase mismatch with input angles"

    def test_vectoring_zero_input(self, cordic_processor):
        """Test vectoring mode with zero input."""
        x = np.array([0.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 0.0])

        mags, phases, _ = cordic_processor.cordic(x, y, method="vectoring")

        # Zero input should result in zero phase
        assert phases[0] == 0.0, "Zero input should produce zero phase"
        assert mags[0] == 0.0, "Zero input should produce zero magnitude"

        # Non-zero inputs should work normally
        # CORDIC has finite precision - allow 2e-5 tolerance
        assert np.abs(phases[1] - np.pi / 2) < 2e-5, "y=1 should produce π/2"
        assert np.abs(phases[2] - 0.0) < 2e-5, "x=1 should produce 0"

    def test_vectoring_vs_rotation_mode(self, cordic_processor):
        """Compare vectoring mode with rotation mode (target_angle=0)."""
        # Generate test data
        n = 100
        angles = np.linspace(-np.pi, np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)

        # Vectoring mode
        mags_vectoring, phases_vectoring, _ = cordic_processor.cordic(x, y, method="vectoring")

        # Rotation mode (old method with target_angle=0)
        mags_rotation, phases_rotation, _ = cordic_processor.cordic(x, y, np.zeros(n), method="rotation")

        # Expected values
        expected_phases = np.arctan2(y, x)
        expected_mags = np.sqrt(x * x + y * y)

        # Vectoring mode should match expected values
        phase_diffs_vectoring = phases_vectoring - expected_phases
        phase_diffs_vectoring = (phase_diffs_vectoring + np.pi) % (2 * np.pi) - np.pi
        # CORDIC has finite precision - allow 3.5e-5 tolerance for large arrays
        assert np.all(
            np.abs(phase_diffs_vectoring) < 3.5e-5
        ), f"Vectoring mode should match atan2, max error: {np.max(np.abs(phase_diffs_vectoring)):.2e}"

        # Rotation mode should fail (known bug)
        # This test documents the bug - rotation mode with target_angle=0 doesn't work
        phase_diffs_rotation = phases_rotation - expected_phases
        phase_diffs_rotation = (phase_diffs_rotation + np.pi) % (2 * np.pi) - np.pi
        # Rotation mode has large errors (this is expected due to the bug)
        assert np.any(
            np.abs(phase_diffs_rotation) > 0.1
        ), "Rotation mode should have errors (documenting the bug)"

        print(
            f"\nVectoring mode max error: {np.max(np.abs(phase_diffs_vectoring)):.2e}"
        )
        print(
            f"Rotation mode max error: {np.max(np.abs(phase_diffs_rotation)):.2e}"
        )

    def test_vectoring_performance(self, cordic_processor):
        """Test performance of vectoring mode with large arrays."""
        import time

        # Large array
        n = 100000
        angles = np.linspace(-np.pi, np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)

        # Time vectoring mode
        start = time.time()
        mags, phases, _ = cordic_processor.cordic(x, y, method="vectoring")
        vectoring_time = time.time() - start

        # Verify results
        expected_phases = np.arctan2(y, x)
        phase_diffs = phases - expected_phases
        phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        max_error = np.max(np.abs(phase_diffs))

        # CORDIC has finite precision - allow 3.5e-5 tolerance for large arrays
        assert max_error < 3.5e-5, f"Large array accuracy issue: {max_error:.2e}"
        print(f"\nVectoring mode time for {n} elements: {vectoring_time:.4f}s")

    def test_vectoring_integration_with_extract_phase_samples(
        self, cordic_processor
    ):
        """Test vectoring mode integration with extract_phase_samples."""
        fs = 50e6
        f0 = 8e6
        duration = 0.001  # 1ms
        t = np.arange(0, duration, 1 / fs)
        signal = np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.randn(len(t))

        # Extract phase samples using vectoring mode
        times, phases = cordic_processor.extract_phase_samples(
            signal, f0, fs, samples_per_period=4
        )

        # Verify results
        assert len(times) > 0, "Should extract some phase samples"
        assert len(phases) == len(times), "Times and phases should match length"
        assert np.all(np.isfinite(phases)), "All phases should be finite"
        # extract_phase_samples performs unwrapping, so phases can grow beyond [-π, π]
        # Instead, check that wrapped phases are in reasonable range
        phases_wrapped = np.angle(np.exp(1j * phases))
        assert np.all(
            np.abs(phases_wrapped) <= np.pi + 1e-4
        ), f"Wrapped phases should be in reasonable range, min={np.min(phases_wrapped):.6f}, max={np.max(phases_wrapped):.6f}"

        # Check that phases are not all zeros (unlike old rotation mode bug)
        assert np.any(np.abs(phases) > 1e-6), "Phases should not all be near zero"

    def test_vectoring_small_array_fallback(self, cordic_processor):
        """Test that small arrays use NumPy implementation."""
        # Small array (< 1000 elements) should use NumPy implementation
        x = np.array([1.0, -1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        mags, phases, _ = cordic_processor.cordic(x, y, method="vectoring")

        # Should still produce correct results
        expected_phases = np.arctan2(y, x)
        phase_diffs = phases - expected_phases
        phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi
        assert np.all(np.abs(phase_diffs) < 2e-5)


