#!/usr/bin/env python3
"""
Tektronix Scope Controller
==========================

This module contains the classes for controlling Tektronix scopes.

Classes:
    TekScopeController: A class to control Tektronix scopes using the "tm-devices".
"""

from .scope import TekScopeController

__all__ = ["TekScopeController"]
