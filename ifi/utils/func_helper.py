#!/usr/bin/env python3
"""
Function Helper
================

This module contains the functions for function helper.
It includes the "assign_kwargs" decorator for injecting 
the declared keyword arguments into function calls from the caller.

Functions:
    assign_kwargs: A decorator for injecting declared keyword arguments 
    into a function call.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable  # noqa: UP035

import numpy as np


def assign_kwargs(keys: list[str]) -> Callable[[Callable], Callable]:
    """
    A decorator that injects keyword arguments into a method call.

    It retrieves values for the given `keys` from the method's `kwargs`
    or from a `self.kwargs_fallback` dictionary on the instance, and
    passes them as keyword arguments to the wrapped method.

    Args:
        keys(list[str]): The keys to inject into the method call.

    Returns:
        A decorator that injects keyword arguments into a method call.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # This dictionary will hold the arguments we prepare.
            injected_kwargs = {}

            for k in keys:
                # Prioritize kwargs passed directly to the call.
                if k in kwargs:
                    # Use the value from kwargs and remove it to avoid duplicate argument errors.
                    value = kwargs.pop(k)
                # Fallback to the 'kwargs_fallback' attribute on the instance.
                elif hasattr(self, "kwargs_fallback") and k in self.kwargs_fallback:
                    value = self.kwargs_fallback[k]
                # If the key is not found anywhere, it's a critical error.
                else:
                    raise ValueError(
                        f"Missing required parameter '{k}' for method '{func.__name__}'"
                    )

                # Only convert non-string values to numpy arrays.
                # This preserves text options and other scalar types.
                if isinstance(value, str):
                    injected_kwargs[k] = value
                else:
                    injected_kwargs[k] = np.asarray(value)

            # The original `kwargs` no longer contains the injected keys.
            # We can now safely call the function with the original *args,
            # the remaining **kwargs, and our new **injected_kwargs.
            return func(self, *args, **injected_kwargs, **kwargs)

        return wrapper

    return decorator


def normalize_kwargs(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize optional kwargs mapping into a plain mutable dict."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"Expected mapping for kwargs, got {type(options).__name__}")
    return dict(options)


def merge_kwargs(*options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge multiple kwargs mappings from left to right."""
    merged: dict[str, Any] = {}
    for option in options:
        merged.update(normalize_kwargs(option))
    return merged


def normalize_call_args(args: tuple[Any, ...] | list[Any] | None) -> tuple[Any, ...]:
    """Normalize optional positional call args into a tuple."""
    if args is None:
        return ()
    if isinstance(args, tuple):
        return args
    if isinstance(args, list):
        return tuple(args)
    raise TypeError(f"Expected tuple/list for call args, got {type(args).__name__}")
