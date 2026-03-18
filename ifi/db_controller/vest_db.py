#!/usr/bin/env python3
"""
VestDB
======

Facade class for VEST MySQL access.
Implementation is split across mixins to keep this entry module compact.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from .vest_db_mixin_query import VestDBMixinQuery
from .vest_db_mixin_setup import VestDBMixinConfig, VestDBMixinConnection


class VestDB(VestDBMixinConfig, VestDBMixinConnection, VestDBMixinQuery):
    """
    VEST database controller.

    Mixins:
        - VestDBMixinConfig: config + field label loading
        - VestDBMixinConnection: connect/disconnect/query/context-manager
        - VestDBMixinQuery: shot-specific query/transform routines
    """

    def __init__(self, config_path: str = "ifi/config.ini"):
        self._setup_logger(component=__name__)
        self._load_config(config_path)
