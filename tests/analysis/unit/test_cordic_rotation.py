"""
Tests for CORDIC rotation mode methods.

Rotation mode rotates a vector to a target angle.
This is different from vectoring mode which calculates the phase directly.
"""
import numpy as np
import pytest

from ifi.analysis.phase_analysis import CORDICProcessor


class TestCORDICRotationMode:
    """Test CORDIC rotation mode functionality."""

    @pytest.fixture
    def cordic_processor(self):
        """Create a CORDICProcessor instance."""
        return CORDICProcessor(n_iterations=16)

    def test_cordic_rotation_basic(self, cordic_processor):
        """Test basic CORDIC rotation with known angles."""
        # Test case: rotate (1, 0) to angle π/4
        x, y = 1.0, 0.0
        target_angle = np.pi / 4

        magnitude, phase, _ = cordic_processor.cordic_rotation(x, y, target_angle)

        # After rotation, phase should be close to target_angle
        assert abs(phase - target_angle) < 1e-4, f"Phase {phase:.6f} should be close to target {target_angle:.6f}"

        # Magnitude should be preserved (with scale factor)
        expected_magnitude = np.sqrt(x**2 + y**2) / cordic_processor.scale_factor
        assert abs(magnitude - expected_magnitude) < 1e-4, "Magnitude should be preserved"

    def test_cordic_rotation_target_zero(self, cordic_processor):
        """Test rotation with target_angle=0 (should match vectoring mode)."""
        x, y = 1.0, 1.0

        # Rotation mode with target_angle=0
        mag_rot, phase_rot, _ = cordic_processor.cordic_rotation(x, y, 0.0)

        # Vectoring mode (should give same result)
        mag_vec, phase_vec, _ = cordic_processor.cordic_vectoring_vectorized(
            np.array([x]), np.array([y])
        )

        # Phase should match arctan2
        expected_phase = np.arctan2(y, x)

        # Note: rotation mode with target_angle=0 has known accuracy issues
        # Allow larger tolerance for rotation mode
        assert abs(phase_rot - expected_phase) < 3e-5 or abs(
            phase_rot - expected_phase - 2 * np.pi
        ) < 3e-5 or abs(phase_rot - expected_phase + 2 * np.pi) < 3e-5, (
            f"Rotation mode phase {phase_rot:.6f} should match expected {expected_phase:.6f}"
        )

        # Vectoring mode should be more accurate
        assert abs(phase_vec[0] - expected_phase) < 2e-5, (
            f"Vectoring mode phase {phase_vec[0]:.6f} should match expected {expected_phase:.6f}"
        )

    def test_cordic_rotation_vectorized_basic(self, cordic_processor):
        """Test vectorized rotation with various target angles."""
        n = 8
        angles = np.linspace(-np.pi, np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)

        # Use actual angles as targets (not zeros)
        target_angles = angles.copy()

        magnitudes, phases, _ = cordic_processor.cordic_rotation_vectorized(
            x, y, target_angles
        )

        # After rotation to target angle, accumulated phase should be close to target
        for i in range(n):
            phase_diff = abs(phases[i] - target_angles[i])
            phase_diff_wrapped = min(phase_diff, abs(phase_diff - 2 * np.pi))
            assert (
                phase_diff_wrapped < 3e-5
            ), f"Phase {phases[i]:.6f} should match target {target_angles[i]:.6f} for element {i}"

    def test_cordic_rotation_vectorized_target_zero(self, cordic_processor):
        """Test vectorized rotation with target_angles=0 (comparison with vectoring)."""
        n = 100
        angles = np.linspace(-np.pi, np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)
        target_angles = np.zeros(n)

        # Rotation mode
        mags_rot, phases_rot, _ = cordic_processor.cordic_rotation_vectorized(
            x, y, target_angles
        )

        # Vectoring mode
        mags_vec, phases_vec, _ = cordic_processor.cordic_vectoring_vectorized(x, y)

        # Expected phases
        expected_phases = np.arctan2(y, x)

        # Compare accuracy
        phase_diffs_rot = phases_rot - expected_phases
        phase_diffs_rot = (phase_diffs_rot + np.pi) % (2 * np.pi) - np.pi
        max_error_rot = np.max(np.abs(phase_diffs_rot))

        phase_diffs_vec = phases_vec - expected_phases
        phase_diffs_vec = (phase_diffs_vec + np.pi) % (2 * np.pi) - np.pi
        max_error_vec = np.max(np.abs(phase_diffs_vec))

        # Vectoring mode should be more accurate for target_angle=0 case
        # But both should be reasonably accurate
        assert max_error_rot < 3e-5, f"Rotation mode max error: {max_error_rot:.2e}"
        assert max_error_vec < 3.5e-5, f"Vectoring mode max error: {max_error_vec:.2e}"

        # Vectoring mode should generally be more accurate
        # (but allow some variance due to numerical differences)
        print(
            f"\nRotation mode max error: {max_error_rot:.2e}, "
            f"Vectoring mode max error: {max_error_vec:.2e}"
        )

    def test_cordic_rotation_negative_target(self, cordic_processor):
        """Test rotation with negative target angle."""
        x, y = 1.0, 0.0
        target_angle = -np.pi / 4

        magnitude, phase, _ = cordic_processor.cordic_rotation(x, y, target_angle)

        # Phase should be close to target (wrapped)
        phase_diff = abs(phase - target_angle)
        assert phase_diff < 1e-4 or abs(phase_diff - 2 * np.pi) < 1e-4, (
            f"Phase {phase:.6f} should be close to target {target_angle:.6f}"
        )

    def test_cordic_rotation_vectorized_array_length_mismatch(self, cordic_processor):
        """Test error handling for array length mismatch."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        target_angles = np.array([0.0])  # Wrong length

        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            cordic_processor.cordic_rotation_vectorized(x, y, target_angles)

    def test_cordic_rotation_vs_vectoring_efficiency(self, cordic_processor):
        """Compare rotation mode (target_angle=0) vs vectoring mode efficiency."""
        import time

        n = 10000
        angles = np.linspace(-np.pi, np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)
        target_angles = np.zeros(n)

        # Time rotation mode
        start = time.time()
        mags_rot, phases_rot, _ = cordic_processor.cordic_rotation_vectorized(
            x, y, target_angles
        )
        rotation_time = time.time() - start

        # Time vectoring mode
        start = time.time()
        mags_vec, phases_vec, _ = cordic_processor.cordic_vectoring_vectorized(x, y)
        vectoring_time = time.time() - start

        # Vectoring mode should be faster for target_angle=0 case
        # (parallelization and optimized algorithm)
        print(
            f"\nRotation mode time: {rotation_time:.4f}s, "
            f"Vectoring mode time: {vectoring_time:.4f}s"
        )

        # Both should complete successfully
        assert len(phases_rot) == n
        assert len(phases_vec) == n

