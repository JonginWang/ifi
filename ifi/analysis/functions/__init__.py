#!/usr/bin/env python3
"""
Functions
=========

This module contains the functions for the analysis.

Functions:
    - power_conversion: Power conversion functions.
    - remezord: Remezord function.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

from .interpolateNonFiniteValues import interpolateNonFinite
from .power_conversion import amp2db, db2amp, db2mag, db2pow, mag2db, pow2db
from .remezord import remezord
from .ridge_tracking import extract_fridges

__all__ = [
    "interpolateNonFinite",
    "amp2db",
    "db2amp",
    "db2mag",
    "db2pow",
    "mag2db",
    "pow2db",
    "remezord",
    "extract_fridges",
]
