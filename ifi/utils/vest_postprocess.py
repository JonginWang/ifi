#!/usr/bin/env python3
"""
Vest Processing
================

This module contains the utilities for VEST post-processing,
for flattening the shot list and converting the VEST data of 
voltages to physical quantities.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import re

_SHOT_QUERY_TEXT_RE = re.compile(r"^[\d\s,:]+$")


def is_shot_query_text(text: str) -> bool:
    """Return True when a string looks like a shot query rather than a file path."""
    normalized = str(text).strip()
    if not normalized:
        return False
    if "/" in normalized or "\\" in normalized:
        return False
    return bool(_SHOT_QUERY_TEXT_RE.fullmatch(normalized))


def normalize_shot_query_items(query_text: str) -> list[str]:
    """Split a shot query string into FlatShotList-ready tokens."""
    normalized = str(query_text).strip()
    if not normalized:
        return []
    if ":" in normalized and "," not in normalized and " " not in normalized:
        return [normalized]
    return [token for token in re.split(r"[\s,]+", normalized) if token]


class FlatShotList:
    """
    Parses and flattens a nested list of shot numbers (int) and file paths (str),
    separating them into distinct lists of unique integers and strings.

    This class handles various input formats including single integers, strings
    with ranges (e.g., "12345:12349:2"), file paths, and nested lists thereof.
    It is designed to prepare a clean query list for NasDB and VestDB.

    Attributes:
        nums (List[int]): A sorted list of unique shot numbers.
        paths (List[str]): A sorted list of unique file paths.
        all (List[Union[int, str]]): A combined list of unique paths and numbers.

    Args:
        raw_list(list[Union[int, str, list]]): The raw list to parse.

    Returns:
        None
    """

    def __init__(self, raw_list: list[int | str | list]):
        self.nums: list[int] = []
        self.paths: list[str] = []

        if raw_list:
            self._flatten_and_parse(raw_list)

        self.nums = sorted(list(set(self.nums)))
        self.paths = sorted(list(set(self.paths)))
        self.all = self.paths + self.nums

    def _flatten_and_parse(self, shot_list: list):
        """
        Recursively flattens and parses the input list.

        Args:
            shot_list(list): The list to flatten and parse.

        Returns:
            None
        """
        stack = list(shot_list)

        while stack:
            item = stack.pop(0)

            if isinstance(item, list):
                stack = item + stack
                continue

            if isinstance(item, int):
                self.nums.append(item)
                continue

            if isinstance(item, str):
                if is_shot_query_text(item) and ("," in item or " " in item):
                    stack = normalize_shot_query_items(item) + stack
                    continue

                if ":" in item:
                    try:
                        parts = list(map(int, item.split(":")))
                        if len(parts) == 2:
                            self.nums.extend(range(parts[0], parts[1] + 1))
                        elif len(parts) == 3:
                            self.nums.extend(range(parts[0], parts[1] + 1, parts[2]))
                        else:
                            self.paths.append(item)
                    except ValueError:
                        self.paths.append(item)

                elif item.isdigit():
                    self.nums.append(int(item))

                else:
                    self.paths.append(item)

            if isinstance(item, range):
                self.nums.extend(list(item))


__all__ = [
    "FlatShotList",
    "is_shot_query_text",
    "normalize_shot_query_items",
]
