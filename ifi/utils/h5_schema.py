#!/usr/bin/env python3
"""
HDF5 Schema Validation and Loading
==================================

Define and validate the canonical HDF5 schema used for IFI analysis results.
Assumptions: Result files are written by `ifi.utils.file_io.save_results_to_hdf5`.
Key I/O: Validation of HDF5 files under `ifi/results/<shot_num>/*.h5`.

Classes:
    SchemaValidationError: Exception raised for schema validation errors.
    
Functions:
    validate_h5_schema: Validate an HDF5 file against the expected schema.
    load_h5_data: Load and validate an HDF5 file, returning its data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py

try:
    from .common import LogManager, log_tag
except ImportError as e:  # pragma: no cover - import fallback
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag


LogManager()
logger = LogManager().get_logger(__name__)


class SchemaValidationError(Exception):
    """
    Raised when an HDF5 results file does not conform to the expected IFI schema.

    This is intentionally separate from `ValidationError` in `ifi.utils.validation`
    to distinguish file/schema issues from normal input-parameter validation.
    """

    pass


def _require_group(hf: h5py.File, path: str) -> h5py.Group:
    """
    Ensure that a group exists at the given path.

    Args:
        hf: Open HDF5 file handle.
        path: Group path (e.g. '/metadata').

    Returns:
        The requested group.

    Raises:
        SchemaValidationError: If the group is missing or of incorrect type.
    """
    if path not in hf:
        msg = f"Required group '{path}' is missing."
        logger.error(f"{log_tag('H5SCHEMA','GROUP')} {msg}")
        raise SchemaValidationError(msg)

    obj = hf[path]
    if not isinstance(obj, h5py.Group):
        msg = f"Path '{path}' must be a group, found {type(obj).__name__}."
        logger.error(f"{log_tag('H5SCHEMA','GROUP')} {msg}")
        raise SchemaValidationError(msg)

    return obj


def _require_attr(group: h5py.Group, name: str) -> None:
    """
    Ensure that an attribute exists on the given group.

    Args:
        group: HDF5 group.
        name: Attribute name.

    Raises:
        SchemaValidationError: If the attribute is missing.
    """
    if name not in group.attrs:
        path = group.name
        msg = f"Attribute '{name}' is missing on group '{path}'."
        logger.error(f"{log_tag('H5SCHEMA','ATTR')} {msg}")
        raise SchemaValidationError(msg)


def validate_h5_schema(file_path: str | Path) -> None:
    """
    Validate that a results HDF5 file conforms to the canonical IFI schema.

    This checks for the presence of required groups and attributes used by
    `save_results_to_hdf5` / `load_results_from_hdf5`. It is intentionally
    conservative: optional groups (e.g. `stft_results`, `cwt_results`,
    `density_data`, `vest_data`) are only validated if present.

    Args:
        file_path: Path to the HDF5 file to validate.

    Raises:
        SchemaValidationError: If the file does not match the expected schema.
        FileNotFoundError: If the file does not exist.
    """
    h5_path = Path(file_path)
    if not h5_path.exists():
        msg = f"HDF5 file not found: {h5_path}"
        logger.error(f"{log_tag('H5SCHEMA','PATH')} {msg}")
        raise FileNotFoundError(msg)

    logger.info(f"{log_tag('H5SCHEMA','VALD')} Validating HDF5 schema: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        # --- /metadata ---
        metadata = _require_group(hf, "/metadata")
        for attr_name in ("shot_number", "created_at", "ifi_version"):
            _require_attr(metadata, attr_name)

        # --- /signals ---
        # signals group must exist but may be empty (marked via attrs['empty']).
        signals = _require_group(hf, "/signals")
        if not signals.attrs.get("empty", False):
            # When not explicitly empty, at least one child group is expected.
            if not list(signals.keys()):
                msg = (
                    "Group '/signals' is not marked empty but contains no child groups."
                )
                logger.error(f"{log_tag('H5SCHEMA','SIGNL')} {msg}")
                raise SchemaValidationError(msg)

        # --- Optional groups: only validate basic type/structure if present ---
        for opt_group in ("/stft_results", "/cwt_results", "/density_data", "/vest_data"):
            if opt_group in hf:
                obj = hf[opt_group]
                if not isinstance(obj, h5py.Group):
                    msg = f"Optional path '{opt_group}' must be a group, found {type(obj).__name__}."
                    logger.error(f"{log_tag('H5SCHEMA','GROUP')} {msg}")
                    raise SchemaValidationError(msg)

    logger.info(f"{log_tag('H5SCHEMA','VALD')} Schema validation passed: {h5_path}")


def load_h5_data(file_path: str | Path) -> Optional[dict]:
    """
    Convenience wrapper that validates an HDF5 result file before loading it.

    This function is intended as a thin abstraction over `load_results_from_hdf5`
    when a **single** file path is available rather than the usual
    `base_dir/shot_num` directory layout.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        A results dictionary compatible with `load_results_from_hdf5`, or None
        if validation fails.
    """
    from .file_io import load_results_from_hdf5  # Lazy import to avoid cycles

    validate_h5_schema(file_path)

    # The existing loader works with a `base_dir/shot_num` layout, so here we
    # adapt by calling it on the parent directory and inferring the shot number
    # from the filename when possible.
    h5_path = Path(file_path)
    shot_str = h5_path.stem
    try:
        shot_num = int(shot_str)
    except ValueError:
        # Fallback for unknown/derived filenames (e.g. for shot_num == 0)
        logger.warning(
            f"{log_tag('H5SCHEMA','LOAD')} Could not infer numeric shot number from '{shot_str}'. "
            "Using 0 and base_dir = parent directory."
        )
        shot_num = 0

    base_dir = str(h5_path.parent.parent) if shot_num != 0 else str(h5_path.parent)
    return load_results_from_hdf5(shot_num=shot_num, base_dir=base_dir)


__all__ = [
    "SchemaValidationError",
    "validate_h5_schema",
    "load_h5_data",
]


