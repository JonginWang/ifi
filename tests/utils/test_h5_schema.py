#!/usr/bin/env python3
"""
Tests for HDF5 schema validation utilities.

These tests focus on minimal structural checks for the IFI result file schema:
- Presence of required root metadata attrs.
- Basic handling of the `/rawdata` group and optional groups.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ifi.utils.io_utils import save_results_to_hdf5
from ifi.utils.io_h5_inspect import (
    SchemaValidationError,
    load_h5_data,
    validate_h5_schema,
)
from ifi.utils.io_process_common import build_raw_cache_file_path, make_raw_cache_group_name


@pytest.fixture
def valid_h5_file(tmp_path):
    """
    Create a minimal but schema-compliant HDF5 file for testing.
    """
    h5_path = tmp_path / "45821.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.attrs["shot_number"] = 45821
        hf.attrs["created_at"] = "2025-11-26T00:00:00"
        hf.attrs["ifi_version"] = "1.0"

        rawdata = hf.create_group("rawdata")
        sig_group = rawdata.create_group("test_signal")
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
    Missing required root attrs should raise SchemaValidationError.
    """
    h5_path = tmp_path / "missing_meta.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.create_group("rawdata")

    with pytest.raises(SchemaValidationError):
        validate_h5_schema(h5_path)


def test_validate_h5_schema_missing_required_attr(tmp_path):
    """
    Missing a required metadata attribute should raise SchemaValidationError.
    """
    h5_path = tmp_path / "missing_attr.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.attrs["shot_number"] = 1
        hf.attrs["created_at"] = "2025-11-26T00:00:00"
        hf.create_group("rawdata")

    with pytest.raises(SchemaValidationError):
        validate_h5_schema(h5_path)


def test_validate_h5_schema_rawdata_empty_flag(tmp_path):
    """
    When /rawdata is marked empty, it may have no child groups.
    """
    h5_path = tmp_path / "empty_signals.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.attrs["shot_number"] = 1
        hf.attrs["created_at"] = "2025-11-26T00:00:00"
        hf.attrs["ifi_version"] = "1.0"

        rawdata = hf.create_group("rawdata")
        rawdata.attrs["empty"] = True

    validate_h5_schema(h5_path)


def test_validate_h5_schema_rawdata_non_empty_without_children(tmp_path):
    """
    If /rawdata is not marked empty and has no children, validation should fail.
    """
    h5_path = tmp_path / "bad_signals.h5"
    with h5py.File(h5_path, "w") as hf:
        hf.attrs["shot_number"] = 1
        hf.attrs["created_at"] = "2025-11-26T00:00:00"
        hf.attrs["ifi_version"] = "1.0"
        hf.create_group("rawdata")

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
    assert "rawdata" in loaded
    assert "test_file.csv" in loaded["rawdata"]
    loaded_df = loaded["rawdata"]["test_file.csv"]
    assert "TIME" in loaded_df.columns
    assert "CH0" in loaded_df.columns


def test_save_results_links_rawdata_to_existing_source_cache(tmp_path):
    """When per-source raw cache exists, canonical results should link to it."""
    base_dir = tmp_path / "results"
    shot_num = 12345
    results_dir = base_dir / str(shot_num)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_signals = pd.DataFrame(
        {"TIME": np.linspace(0, 1, 8), "CH0": np.random.randn(8)}
    )
    source_name = "test_file.csv"
    cache_file = build_raw_cache_file_path(results_dir, source_name, shot_num=shot_num)
    cache_group_name = make_raw_cache_group_name(source_name)

    with h5py.File(cache_file, "w") as cache_hf:
        rawdata = cache_hf.create_group("rawdata")
        sig_group = rawdata.create_group(cache_group_name)
        sig_group.attrs["original_name"] = source_name
        sig_group.attrs["canonical_name"] = source_name
        sig_group.create_dataset("TIME", data=df_signals["TIME"].to_numpy())
        ch0 = sig_group.create_dataset("CH0", data=df_signals["CH0"].to_numpy())
        ch0.attrs["original_name"] = "CH0"

    saved_path = save_results_to_hdf5(
        str(results_dir),
        shot_num,
        {source_name: df_signals},
        {},
        {},
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert saved_path is not None
    with h5py.File(saved_path, "r") as hf:
        rawdata = hf["rawdata"]
        assert len(rawdata) == 1
        group_name = next(iter(rawdata.keys()))
        link = rawdata.get(group_name, getlink=True)
        assert isinstance(link, h5py.ExternalLink)
        assert link.filename == cache_file.name
        assert link.path == f"/rawdata/{cache_group_name}"


