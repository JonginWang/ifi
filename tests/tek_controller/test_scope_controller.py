#!/usr/bin/env python3
"""
Tests for TekScopeController and its internal state machine.

These tests mock out the tm-devices DeviceManager and scope objects so that
no real hardware is required. The focus is on verifying state transitions and
basic method behaviour.

TODO:
    Add an optional integration test suite that runs only when a real
    Tektronix scope is available and a dedicated test configuration is set.
    This suite should:
        - Attempt a real connection using a known VISA/tm-devices address.
        - Perform a short acquisition on a test channel (e.g. CH1) and verify
          that non-empty waveform data is returned.
        - Validate that the reported trigger state and identification string
          match expectations for the laboratory setup.
    The hardware-backed tests must be clearly marked (e.g. with a pytest
    marker) so that they are skipped by default in CI or environments
    without connected instruments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import ifi.tek_controller.scope as scope_mod
from ifi.tek_controller.scope import ScopeState, TekScopeController


class DummyScopeBase:
    """Base class used to stand in for MDO3/MSO5 in tests."""


class DummyWaveform:
    """Simple waveform container with x and y arrays."""

    def __init__(self) -> None:
        self.x = np.array([0.0, 1.0], dtype=float)
        self.y = np.array([0.1, 0.2], dtype=float)


class DummyCurve:
    """Dummy curve command interface."""

    def get(self) -> DummyWaveform:
        return DummyWaveform()


class DummyDataSource:
    """Dummy data source command interface."""

    def __init__(self) -> None:
        self.last_channel: str | None = None

    def write(self, channel: str) -> None:
        self.last_channel = channel


class DummyCommands:
    """Aggregate dummy command interfaces to mimic tm-devices API."""

    def __init__(self) -> None:
        self.data = type("Data", (), {"source": DummyDataSource()})()
        self.curve = DummyCurve()


class DummyDevice(DummyScopeBase):
    """Dummy scope device used for testing without hardware."""

    def __init__(self) -> None:
        self.model = "MDO3"
        self.serial = "1234"
        self.idn_string = "TEKTRONIX,MDO3,1234,FW:1.0"
        self.device_name = "DummyScope"
        self.trigger = type("Trigger", (), {"state": "READY"})()
        self.commands = DummyCommands()

    def close(self) -> None:
        """Simulate closing a connection to the device."""


class DummyDeviceManager:
    """
    Dummy replacement for tm_devices.DeviceManager.

    It acts as a context manager and exposes a `devices` list along with a
    `get_device` helper similar to the real implementation.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.devices = [DummyDevice()]
        self.setup_cleanup_enabled = False
        self.teardown_cleanup_enabled = False
        self.visa_library: Any = None

    def __enter__(self) -> "DummyDeviceManager":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        return None

    def get_device(self, serial: str) -> DummyDevice:
        # In tests we simply ignore the serial and always return a dummy scope.
        return DummyDevice()


@pytest.fixture(autouse=True)
def patch_tm_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically patch tm-devices classes used inside TekScopeController.

    This ensures that no real hardware or drivers are required for the tests.
    """

    # Replace DeviceManager with our dummy implementation
    monkeypatch.setattr(scope_mod, "DeviceManager", DummyDeviceManager)

    # Replace MDO3/MSO5 with a common dummy base so isinstance checks succeed
    monkeypatch.setattr(scope_mod, "MDO3", DummyScopeBase)
    monkeypatch.setattr(scope_mod, "MSO5", DummyScopeBase)


def test_initial_state_is_idle() -> None:
    """Controller should start in the IDLE state with no connected scope."""
    controller = TekScopeController()
    assert controller.state == ScopeState.IDLE
    assert controller.scope is None


def test_connect_success_sets_state_idle() -> None:
    """Successful connect should end in IDLE state with a valid scope."""
    controller = TekScopeController()
    ok = controller.connect("MDO3 - 1234")
    assert ok is True
    assert isinstance(controller.scope, DummyScopeBase)
    assert controller.state == ScopeState.IDLE


def test_connect_failure_sets_state_error() -> None:
    """
    A malformed identifier should cause connect() to fail and set ERROR state.
    """
    controller = TekScopeController()
    ok = controller.connect("INVALID_IDENTIFIER")
    assert ok is False
    assert controller.scope is None
    assert controller.state == ScopeState.ERROR


def test_get_waveform_data_updates_state_and_returns_arrays() -> None:
    """
    get_waveform_data should transition to ACQUIRING, then back to IDLE and
    return time and voltage arrays when a scope is connected.
    """
    controller = TekScopeController()
    assert controller.connect("MDO3 - 1234")

    time_values, voltage_values = controller.get_waveform_data(channel="CH1")

    assert controller.state == ScopeState.IDLE
    assert isinstance(time_values, np.ndarray)
    assert isinstance(voltage_values, np.ndarray)
    assert time_values.shape == voltage_values.shape
    assert time_values.size > 0


def test_get_waveform_data_without_scope_sets_error_and_returns_none() -> None:
    """If no scope is connected, get_waveform_data should set ERROR and return None."""
    controller = TekScopeController()
    controller.scope = None
    result = controller.get_waveform_data(channel="CH1")
    assert result is None
    assert controller.state == ScopeState.ERROR


def test_is_idle_and_is_busy_helpers() -> None:
    """Smoke test for is_idle and is_busy helper methods."""
    controller = TekScopeController()
    assert controller.is_idle() is True
    assert controller.is_busy() is False

    controller.state = ScopeState.ACQUIRING
    assert controller.is_idle() is False
    assert controller.is_busy() is True


def test_acquire_data_returns_dict_for_multiple_channels() -> None:
    """acquire_data should return a mapping from channel names to (time, voltage)."""
    controller = TekScopeController()
    assert controller.connect("MDO3 - 1234")

    channels = ["CH1", "CH2"]
    result = controller.acquire_data(channels)

    assert controller.state == ScopeState.IDLE
    assert set(result.keys()) == set(channels)
    for ch in channels:
        t, v = result[ch]
        assert isinstance(t, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert t.shape == v.shape
        assert t.size > 0


def test_save_data_creates_csv_and_resets_state(tmp_path) -> None:
    """
    save_data should create a CSV file with expected columns and return to IDLE.
    """
    controller = TekScopeController()
    # Simulate an already-connected scope; save_data does not require it directly.

    data = {
        "TIME": np.array([0.0, 1.0], dtype=float),
        "CH1": np.array([0.1, 0.2], dtype=float),
    }

    shot_code = 45821
    suffix = "_056"

    filepath = controller.save_data(tmp_path, shot_code, suffix, data, file_format="CSV")

    assert controller.state == ScopeState.IDLE
    assert filepath is not None
    assert filepath.exists()

    # Verify file contents via pandas
    df = pd.read_csv(filepath)
    assert list(df.columns) == ["TIME", "CH1"]
    assert len(df) == 2


def test_save_data_hdf5_format_uses_h5_extension(tmp_path) -> None:
    """save_data with file_format='HDF5' should create an .h5 file."""
    controller = TekScopeController()

    data = {
        "TIME": np.array([0.0, 1.0], dtype=float),
        "CH1": np.array([0.1, 0.2], dtype=float),
    }

    shot_code = 45821
    suffix = "_ALL"

    filepath = controller.save_data(tmp_path, shot_code, suffix, data, file_format="HDF5")

    assert controller.state == ScopeState.IDLE
    assert filepath is not None
    assert filepath.suffix.lower() == ".h5"


