#!/usr/bin/env python3
"""
Inspect HDF5 Structure
=======================

Inspect HDF5 file structure and print a tree.

Functions:
    validate_h5_schema: Validate the HDF5 schema.
    load_h5_data: Load the HDF5 data.
    inspect_h5: Inspect the HDF5 structure and print a tree.
    main: Main function to inspect the HDF5 structure and print a tree.

Examples:
    ```bash
    python ifi/utils/io_h5_inspect.py <path0.h5> [path1.h5 ...]
    ```

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

from .io_h5 import (
    H5_GROUP_CWT,
    H5_GROUP_DENSITY,
    H5_GROUP_RAWDATA,
    H5_GROUP_STFT,
    H5_GROUP_VEST,
    NATURAL_NAME_RE,
)
from .log_manager import LogManager

logger = LogManager().get_logger(__name__, "WARNING")


def _is_valid_natural_name(name: str) -> bool:
    return bool(NATURAL_NAME_RE.match(str(name)))


def _has_required_root_metadata(hf: h5py.File) -> bool:
    """Check if the root metadata group has the required attributes."""
    required = ("shot_number", "created_at", "ifi_version")
    return all(name in hf.attrs for name in required)


def _validate_natural_names(group: h5py.Group, group_label: str) -> None:
    """Validate the natural names of the child groups."""
    for child_name in group.keys():
        if not _is_valid_natural_name(str(child_name)):
            raise SchemaValidationError(
                f"Invalid child name '{child_name}' under '{group_label}'. "
                "Expected ^[A-Za-z_][A-Za-z0-9_]*$."
            )


class SchemaValidationError(Exception):
    """Raised when an HDF5 file does not conform to the expected IFI schema."""


def validate_h5_schema(file_path: str | Path) -> None:
    """
    Validate that a results HDF5 file conforms to IFI canonical schema.
    """
    h5_path = Path(file_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        # Metadata: canonical root attrs
        has_root_meta = _has_required_root_metadata(hf)
        if not has_root_meta:
            raise SchemaValidationError(
                "Missing required metadata. Expected either root attrs "
                "shot_number/created_at/ifi_version."
            )

        # Raw signal group: canonical
        raw_group = hf.get(f"/{H5_GROUP_RAWDATA}")
        if raw_group is None:
            raise SchemaValidationError(
                f"Required group '/{H5_GROUP_RAWDATA}' not found."
            )

        if not raw_group.attrs.get("empty", False):
            if not list(raw_group.keys()):
                raise SchemaValidationError(
                    "Signal group is not marked empty but contains no child groups."
                )
            _validate_natural_names(raw_group, raw_group.name)

        optional_groups = (
            f"/{H5_GROUP_STFT}",
            f"/{H5_GROUP_CWT}",
            f"/{H5_GROUP_DENSITY}",
        )
        for opt_grp in optional_groups:
            if opt_grp in hf:
                obj = hf[opt_grp]
                if not isinstance(obj, h5py.Group):
                    raise SchemaValidationError(
                        f"Optional group '{opt_grp}' must be a group, found {type(obj).__name__}."
                    )
                _validate_natural_names(obj, opt_grp)

        vest_group = hf.get(f"/{H5_GROUP_VEST}")
        if vest_group is not None and not isinstance(vest_group, h5py.Group):
            raise SchemaValidationError(
                f"Optional group '/{H5_GROUP_VEST}' must be a group, "
                f"found {type(vest_group).__name__}."
            )


def load_h5_data(file_path: str | Path) -> dict | None:
    """Validate a single HDF5 file and load it via the existing results loader."""
    if "load_results_from_hdf5" not in sys.modules:
        from .io_utils import load_results_from_hdf5  # Lazy import to avoid cycles

    validate_h5_schema(file_path)

    h5_path = Path(file_path)
    shot_str = h5_path.stem
    try:
        shot_num = int(shot_str)
    except ValueError:
        logger.warning(
            "Could not infer numeric shot number from '%s'. Using shot_num=0.",
            shot_str,
        )
        shot_num = 0

    base_dir = str(h5_path.parent.parent) if shot_num != 0 else str(h5_path.parent)
    return load_results_from_hdf5(shot_num=shot_num, base_dir=base_dir)


def _describe_h5_obj(
    name: str, obj: h5py.Group | h5py.Dataset, prefix: str = "", 
    max_attrs_group: int = 5, max_attrs_dataset: int = 5
    ) -> list[str]:
    lines: list[str] = []
    if isinstance(obj, h5py.Group):
        lines.append(f"{prefix}{name}/  (Group)")
        for k, v in list(obj.attrs.items())[:max_attrs_group]:
            vstr = str(v)
            if len(vstr) > 60:
                vstr = vstr[:57] + "..."
            lines.append(f"{prefix}  @{k}: {vstr}")
        if len(obj.attrs) > max_attrs_group:
            lines.append(f"{prefix}  ... and {len(obj.attrs) - max_attrs_group} more attrs")
        for key in obj.keys():
            child = obj[key]
            if isinstance(child, h5py.Dataset):
                lines.append(
                    f"{prefix}  {key}  (Dataset shape={child.shape} dtype={child.dtype})"
                )
                for ak, av in list(child.attrs.items())[:max_attrs_dataset]:
                    lines.append(f"{prefix}    @{ak}: {av}")
                if len(child.attrs) > max_attrs_dataset:
                    lines.append(f"{prefix}    ... and {len(child.attrs) - max_attrs_dataset} more attrs")
            else:
                lines.extend(_describe_h5_obj(key, child, prefix=prefix + "  "))
    return lines


def inspect_h5(path: Path, max_attrs: int = 5) -> list[str]:
    """Inspect HDF5 file structure (groups/datasets/attrs) and return printable lines."""
    lines = [f"File: {path}", "=" * 60]
    try:
        with h5py.File(path, "r") as hf:
            root_meta_keys = [k for k in hf.attrs.keys()]
            if root_meta_keys:
                lines.append(f"Root metadata attrs ({len(root_meta_keys)}):")
                for mk in sorted(root_meta_keys):
                    lines.append(f"  @{mk}: {hf.attrs[mk]}")
            else:
                lines.append("Root metadata attrs: none")

            for key in sorted(hf.keys()):
                if not _is_valid_natural_name(str(key)):
                    lines.append(
                        f"WARNING: Invalid top-level name '{key}' (natural naming rule)"
                    )
                obj = hf[key]
                if isinstance(obj, h5py.Dataset):
                    lines.append(f"/{key}  (Dataset shape={obj.shape} dtype={obj.dtype})")
                    for ak, av in list(obj.attrs.items())[:max_attrs]:
                        lines.append(f"  @{ak}: {av}")
                    if len(obj.attrs) > max_attrs:
                        lines.append(f"  ... and {len(obj.attrs) - max_attrs} more attrs")
                else:
                    lines.extend(_describe_h5_obj(key, obj, prefix="/"))
    except Exception as e:
        lines.append(f"Error: {e}")
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python ifi/utils/io_h5_inspect.py <path0.h5> [path1.h5 ...]")
        return 1

    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"Not found: {path}")
            continue
        for line in inspect_h5(path):
            print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
