#!/usr/bin/env python3
"""
VestDB Interface/Base
=====================

VestDB base class with shared logger helpers.

Classes:
    VestDBIface: Interface for VestDB mixins
    VestDBBase: Base class for VestDB mixins

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import logging
from typing import Protocol

from ..utils.log_manager import LogManager


class VestDBIface(Protocol):
    """Minimum logger contract expected by VestDB mixins."""

    logger: logging.Logger | None

    def _setup_logger(self, component: str | None = None, level: str = "INFO") -> None:
        ...

    def _ensure_logger(self, component: str | None = None, level: str = "INFO") -> None:
        ...


class VestDBBase(VestDBIface):
    """Shared logger lifecycle helpers for VestDB mixins."""

    logger: logging.Logger | None = None

    def _setup_logger(self, component: str | None = None, level: str = "INFO") -> None:
        logger_name = component or f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = LogManager().get_logger(logger_name, level=level)

    def _ensure_logger(self, component: str | None = None, level: str = "INFO") -> None:
        if not hasattr(self, "logger") or self.logger is None:
            self._setup_logger(component=component, level=level)


__all__ = [
    "VestDBIface", 
    "VestDBBase",
]
