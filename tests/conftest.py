#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for the ifi test suite.

This module contains pytest hooks that are automatically discovered and executed:
- pytest_addoption: Registers custom command-line options
- pytest_configure: Configures pytest behavior based on options

Functions defined here (pytest_* naming convention) are hook functions that
pytest automatically discovers and calls at appropriate times during test execution.
No explicit pytest import is needed for these hook functions.
"""

import os


def pytest_addoption(parser):
    """Add custom command-line options for pytest.
    
    This hook is called by pytest during command-line argument parsing.
    It registers the --run-heavy option that enables heavy performance tests.
    
    Args:
        parser: pytest's argument parser instance (automatically provided by pytest)
    """
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="run heavy performance tests"
    )


def pytest_configure(config):
    """Configure pytest with custom options.
    
    This hook is called by pytest after command-line arguments are parsed
    but before test collection begins. We use it to set environment variables
    based on command-line options.
    
    Args:
        config: pytest's Config object (automatically provided by pytest)
    """
    # Set environment flag if --run-heavy option is used
    # This allows test modules to check for the flag at import time
    if config.getoption("--run-heavy", default=False):
        os.environ["_PYTEST_RUN_HEAVY"] = "1"

