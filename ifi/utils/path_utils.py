from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def is_windows() -> bool:
    return os.name == "nt"


def to_path(path_like: PathLike) -> Path:
    if isinstance(path_like, Path):
        return path_like
    return Path(str(path_like))


def normalize_to_forward_slash(path_like: PathLike) -> str:
    """Return a string path using forward slashes. Safe for Windows and pandas."""
    if isinstance(path_like, Path):
        return path_like.as_posix()
    return str(path_like).replace("\\", "/")


def ensure_str_path(path_like: PathLike) -> str:
    """Coerce to string and normalize separators to forward slashes."""
    return normalize_to_forward_slash(path_like)


def resolve_repo_root(start: PathLike | None = None) -> Path:
    """Best-effort: find project root by ascending until directory containing 'ifi' package.

    This avoids relying on IDE working directory and supports interactive runs.
    """
    start_path = to_path(start) if start else Path(__file__).resolve()
    for parent in [start_path] + list(start_path.parents):
        if (parent / "ifi").is_dir():
            return parent if parent.is_dir() else parent.parent
    return Path(__file__).resolve().parent.parent


def add_repo_root_to_sys_path() -> Path:
    root = resolve_repo_root()
    root_str = normalize_to_forward_slash(root)
    if root_str not in [normalize_to_forward_slash(p) for p in sys.path]:
        sys.path.insert(0, root_str)
    return root


def chdir_repo_root() -> Path:
    root = resolve_repo_root()
    os.chdir(root)
    return root


