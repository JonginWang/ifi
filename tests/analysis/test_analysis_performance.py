#!/usr/bin/env python3
"""
Performance benchmarks for core time-frequency analysis.

Use `pytest -q` for a light run, or `pytest -q --run-heavy` to enable
heavier dataset sizes. If `pytest-benchmark` is installed, it will provide
timed results; otherwise, we fall back to simple timing assertions.
"""

from __future__ import annotations

import os
import time

import numpy as np  # noqa: F401
import pytest

from ifi.analysis.spectrum import SpectrumAnalysis

from tests.analysis.fixtures.synthetic_signals import sine_signal, linear_chirp


def _heavy_enabled() -> bool:
    """Check if heavy tests should run via environment variable or pytest config."""
    # Check environment variable (available at import time)
    env_enabled = os.environ.get("IFI_RUN_HEAVY", "0") == "1"
    # Check pytest config (set via pytest_configure hook in conftest.py)
    pytest_enabled = os.environ.get("_PYTEST_RUN_HEAVY", "0") == "1"
    return env_enabled or pytest_enabled


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("light", id="light"),
        pytest.param("heavy", id="heavy", marks=pytest.mark.skipif(not _heavy_enabled(), reason="set IFI_RUN_HEAVY=1 or use --run-heavy")),
    ],
)
def test_stft_performance(mode):
    fs = 50e6 if mode == "light" else 100e6
    duration = 0.010 if mode == "light" else 0.050
    f0 = 10e6
    _, x = sine_signal(fs=fs, freq=f0, duration=duration)

    analyzer = SpectrumAnalysis()
    t0 = time.time()
    f, t, Z = analyzer.compute_stft(x, fs)
    elapsed = time.time() - t0

    # Basic sanity
    assert f.size > 0 and t.size > 0 and Z.size > 0

    # Loose upper bound to catch regressions (tuned for CI-friendly values)
    assert elapsed < (1.0 if mode == "light" else 3.0)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("light", id="light"),
        pytest.param("heavy", id="heavy", marks=pytest.mark.skipif(not _heavy_enabled(), reason="set IFI_RUN_HEAVY=1 or use --run-heavy")),
    ],
)
def test_cwt_performance(mode):
    fs = 20e6 if mode == "light" else 40e6
    duration = 0.010 if mode == "light" else 0.040
    f0, f1 = 2e6, 8e6
    _, x = linear_chirp(fs=fs, f0=f0, f1=f1, duration=duration)

    analyzer = SpectrumAnalysis()
    t0 = time.time()
    freqs_cwt, Wx = analyzer.compute_cwt(x, fs, wavelet="gmw", nv=32)
    elapsed = time.time() - t0

    assert freqs_cwt.size > 0 and Wx.size > 0
    assert elapsed < (1.5 if mode == "light" else 5.0)


