import sys
import os
import functools
import numpy as np
from typing import Callable

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    When running as a bundled exe, the path is relative to the temp
    directory created by PyInstaller (_MEIPASS).
    """
    try:
        # PyInstaller creates a temp folder and stores its path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in a bundle, so the base path is the project root
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path) 


def assign_kwargs(keys: list[str]) -> Callable[[Callable], Callable]:
    """
    A decorator that injects keyword arguments into a method call.

    It retrieves values for the given `keys` from the method's `kwargs`
    or from a `self.kwargs_fallback` dictionary on the instance, and
    passes them as keyword arguments to the wrapped method.
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
                elif hasattr(self, 'kwargs_fallback') and k in self.kwargs_fallback:
                    value = self.kwargs_fallback[k]
                # If the key is not found anywhere, it's a critical error.
                else:
                    raise ValueError(f"Missing required parameter '{k}' for method '{func.__name__}'")
                
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
