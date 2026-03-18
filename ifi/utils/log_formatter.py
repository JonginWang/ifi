#!/usr/bin/env python3
"""
Logging Formatter
================

This module contains the functions for logging formatter.
It includes the "CustomFormatter" class for formatting log messages.
And the "log_tag" function for standardizing the tags.

Classes:
    CustomFormatter: A class for formatting log messages.

Functions:
    log_tag: Return a standardized tag like "[SPECR-STFT]" with 5-char codes.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import logging


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that right-aligns the logger name to 15 characters
    and truncates with an ellipsis if it's longer.
    The fill_char (default: '#') is used to fill the name to the limit if it's shorter.
    The none_char (default: '.') is used to fill the name to the limit if it's not a string.

    Args:
        fmt(str): The format of the log message.
        datefmt(str): The format of the date and time.
        style(str): The style of the log message.
        limit(int): The limit of the log message.
        ellipsis(str): The ellipsis to use.
        fill_char(str): The fill character to use.
        none_char(str): The none character to use.

    Attributes:
        datefmt(str): The format of the date and time.
        limit(int): The limit of the log message.
        ellipsis(str): The ellipsis to use.
        fill_char(str): The fill character to use.
        none_char(str): The none character to use.
        level_limit(int): The limit of the log level.

    Methods:
        format(self, record): Format the log message.
    """

    def __init__(
        self,
        fmt=None,
        datefmt=None,
        style="%",
        limit=15,
        ellipsis="*",
        fill_char="#",
        none_char=".",
    ):
        super().__init__(fmt, datefmt, style)
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.limit = limit
        self.ellipsis = ellipsis
        self.fill_char = fill_char
        self.none_char = none_char
        self.level_limit = len("CRITICAL")
        # if len(ellipsis) ==3:
        #     self.ellipsis = ellipsis
        # elif len(ellipsis) > 3:
        #     self.ellipsis = ellipsis[:3]
        # else:
        #     self.ellipsis = ellipsis.ljust(3, '*')

    def format(self, record):
        # Store the original name
        name_original = record.name
        # Store the original level
        level_original = record.levelname
        if isinstance(record.name, str):
            if len(record.name) > self.limit:  # self.limit = 15
                # Truncate and add ellipsis
                record.name = (
                    record.name[: self.limit - len(self.ellipsis)] + self.ellipsis
                )
            elif len(record.name) < self.limit:
                # Use ljust for left alignment with fill char
                record.name = record.name.ljust(self.limit, self.fill_char)
        else:
            # BUG FIX: Assign a string, not a list
            record.name = self.none_char * self.limit

        if isinstance(record.levelname, str):
            record.levelname = record.levelname.ljust(self.level_limit, " ")
        # Format the record
        record_formatted = super().format(record)
        # Restore the original name
        record.name = name_original
        record.levelname = level_original

        return record_formatted


def log_tag(major: str, minor: str) -> str:
    """Return a standardized tag like "[SPECR-STFT]" with 5-char codes.

    The function pads/truncates both codes to exactly 5 uppercase characters
    to keep log prefixes visually consistent and grep-friendly. Use together
    with parameterized logging, e.g.:

        # Example (preferred modern usage):
        # logger.info(f"{log_tag('PLOTS','SAVE')} Message with {path}")

    Args:
        major: Major subsystem (e.g., 'PLOTS', 'NASDB').
        minor: Subcomponent/action (e.g., 'STFT', 'CONN').

    Returns:
        Bracketed tag string: "[MAJOR-MINOR]" with each 5 chars, uppercase.
    """
    try:
        def _fmt(code: str) -> str:
            s = (str(code) if code is not None else "") .upper()
            return (s[:5]).ljust(5, " ") if len(s) >= 5 else s.ljust(5, " ")

        return f"[{_fmt(major)}-{_fmt(minor)}]"
    except Exception:
        return "[UNKN -UNKN ]"
