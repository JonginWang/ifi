#!/usr/bin/env python3
"""
Tests for HDF5 schema validation utilities.

These tests focus on minimal structural checks for the IFI result file schema:
- Presence of required groups and metadata attributes.
- Basic handling of the `/signals` group and optional groups.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ifi.utils.file_io import save_results_to_hdf5
from ifi.utils.h5_schema import SchemaValidationError, load_h5_data, validate_h5_schema


@pytest.fixture
def valid_h5_file(tmp_path):
    """
    Create a minimal but schema-compliant HDF5 file for testing.
    """
    h5_path = tmp_path / "45821.h5"
    with h5py.File(h5_path, "w") as hf:
        # /metadata
        metadata = hf.create_group("metadata")
        metadata.attrs["shot_number"] = 45821
        metadata.attrs["created_at"] = "2025-11-26T00:00:00"
        metadata.attrs["ifi_version"] = "1.0"

        # /signals with one simple DataFrame-equivalent group
        signals = hf.create_group("signals")
        sig_group = signals.create_group("test_signal")
        sig_group.create_dataset("TIME", data=np.linspace(0, 1, 10))
        sig_group.create_dataset("CH0", data=np.random.randn(10))

    return h5_path


def test_validate_h5_schema_valid_file(valid_h5_file):
    """
    validate_h5_schema should pass silently for a well-formed file.
    """
    validate_h5_schema(valid_h5_file)


def test_validate_h5_schema_missing_metadata(tmp_path):
    """
    Missing /metadata group should raise SchemaValidationError.
    """
    h5_path = tmp_path / "missing_meta.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_group("signals")

    with pytest.raises(SchemaValidationError):
        validate_h5_schema(h5_path)


def test_validate_h5_schema_missing_required_attr(tmp_path):
    """
    Missing a required metadata attribute should raise SchemaValidationError.
    """
    h5_path = tmp_path / "missing_attr.h5"
    with h5py.File(h5_path, "w") as hf:
        metadata = hf.create_group("metadata")
        # Intentionally omit 'ifi_version'
        metadata.attrs["shot_number"] = 1
        metadata.attrs["created_at"] = "2025-11-26T00:00:00"
        hf.create_group("signals")

    with pytest.raises(SchemaValidationError):
        validate_h5_schema(h5_path)


def test_validate_h5_schema_signals_empty_flag(tmp_path):
    """
    When /signals is marked empty, it may have no child groups.
    """
    h5_path = tmp_path / "empty_signals.h5"
    with h5py.File(h5_path, "w") as hf:
        meta = hf.create_group("metadata")
        meta.attrs["shot_number"] = 1
        meta.attrs["created_at"] = "2025-11-26T00:00:00"
        meta.attrs["ifi_version"] = "1.0"

        signals = hf.create_group("signals")
        signals.attrs["empty"] = True

    validate_h5_schema(h5_path)


def test_validate_h5_schema_signals_non_empty_without_children(tmp_path):
    """
    If /signals is not marked empty and has no children, validation should fail.
    """
    h5_path = tmp_path / "bad_signals.h5"
    with h5py.File(h5_path, "w") as hf:
        meta = hf.create_group("metadata")
        meta.attrs["shot_number"] = 1
        meta.attrs["created_at"] = "2025-11-26T00:00:00"
        meta.attrs["ifi_version"] = "1.0"
        hf.create_group("signals")

    with pytest.raises(SchemaValidationError):
        validate_h5_schema(h5_path)


def test_load_h5_data_integration_with_save_results(tmp_path):
    """
    Integration test:
    - Use save_results_to_hdf5 to create a realistic file under base_dir/shot_num/.
    - Call load_h5_data on the actual .h5 path.
    - Ensure that schema validation passes and loaded data matches expectations.
    """
    base_dir = tmp_path / "results"
    shot_num = 12345
    results_dir = base_dir / str(shot_num)

    # Create simple signal and density/vest data
    time = np.linspace(0, 1, 10)
    df_signals = pd.DataFrame({"TIME": time, "CH0": np.random.randn(10)})
    density_df = pd.DataFrame({"ne_CH0_test": np.random.randn(10)})
    vest_df = pd.DataFrame({"ip": np.random.randn(5), "time": np.linspace(0, 1, 5)})

    saved_path = save_results_to_hdf5(
        str(results_dir),
        shot_num,
        {"test_file.csv": df_signals},
        {},
        {},
        density_df,
        vest_df,
    )

    assert saved_path is not None
    h5_path = Path(saved_path)
    assert h5_path.exists()

    # load_h5_data should validate schema and then load via load_results_from_hdf5
    loaded = load_h5_data(h5_path)
    assert loaded is not None
    assert "metadata" in loaded
    assert loaded["metadata"]["shot_number"] == shot_num
    assert "signals" in loaded
    assert "test_file.csv" in loaded["signals"]
    loaded_df = loaded["signals"]["test_file.csv"]
    assert "TIME" in loaded_df.columns
    assert "CH0" in loaded_df.columns


