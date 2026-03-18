#!/usr/bin/env python3
"""
Vest Processing
==============

This module contains the utilities for VEST post-processing,
for flattening the shot list and converting the VEST data of 
voltages to physical quantities.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations


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
]
