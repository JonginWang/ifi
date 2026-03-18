#!/usr/bin/env python3
"""
NasDB
=====

Composable NAS controller built from focused mixins.
This module is a composite class that combines multiple mixins to provide a comprehensive NAS data access interface.

    The module is composed of the following mixins:
    - NasDBMixinSetup: Setup and connection mixin
    - NasDBMixinQueryFiles: File discovery and size-estimation mixin
    - NasDBMixinCache: Results/cache helper mixin
    - NasDBMixinQueryShots: Shot-data query mixin
    - NasDBMixinReadDispatch: Read dispatch and CSV type detection mixin
    - NasDBMixinParseCsv: CSV parser mixin
    - NasDBMixinParseData: Data parser mixin
    - NasDBMixinReadTop: Top-lines read mixin

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from .nas_db_mixin_cache import NasDBMixinCache
from .nas_db_mixin_parse_csv import NasDBMixinParseCsv
from .nas_db_mixin_parse_data import NasDBMixinParseData
from .nas_db_mixin_query_files import NasDBMixinQueryFiles
from .nas_db_mixin_query_shots import NasDBMixinQueryShots
from .nas_db_mixin_read_dispatch import NasDBMixinReadDispatch
from .nas_db_mixin_read_top import NasDBMixinReadTop
from .nas_db_mixin_setup import NasDBMixinSetup
from .nas_db_utils import (
    ALLOWED_EXTENSIONS,
    EXTENSION_PRIORITY,
    _extract_filename_from_path,
    _is_drive_or_unc_path,
    _looks_like_path,
)


class NasDB(
    NasDBMixinSetup,
    NasDBMixinQueryFiles,
    NasDBMixinCache,
    NasDBMixinQueryShots,
    NasDBMixinReadDispatch,
    NasDBMixinParseCsv,
    NasDBMixinParseData,
    NasDBMixinReadTop,
):
    """NAS data access class composed via mixins."""


__all__ = [
    "NasDB",
    "ALLOWED_EXTENSIONS",
    "EXTENSION_PRIORITY",
    "_extract_filename_from_path",
    "_is_drive_or_unc_path",
    "_looks_like_path",
]
