import sys
import os
import functools
import numpy as np
from typing import Callable
import logging
from datetime import datetime

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


def ensure_dir_exists(path: str):
    """
    Ensures that a directory exists at the given path.
    If the directory does not exist, it is created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")


def setup_logging(level="INFO"):
    """
    Configures the root logger to save all log levels to a file and
    stream INFO and above to the console.
    
    Log files are saved in the 'logs' directory with the format:
    YYYY-MM-DD_executed_script_name.log
    """
    if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        log_level = getattr(logging, level.upper())
    else:
        log_level = logging.INFO

    try:
        # Determine the name of the script that is being executed.
        if hasattr(sys.modules['__main__'], '__file__'):
            main_script_path = sys.modules['__main__'].__file__
            script_name = os.path.splitext(os.path.basename(main_script_path))[0]
        else:
            # Fallback for certain interactive environments like some Jupyter setups
            script_name = 'interactive'
            
        # Create log directory if it doesn't exist
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Define log file name and path
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f"{date_str}_{script_name}.log")
        
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # Always set root to the lowest level to capture everything

        # Clear any existing handlers to avoid duplicate logs
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # --- File Handler (logs everything from DEBUG level) ---
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # --- Console Handler (logs INFO and above) ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level) # Set console level based on the function argument
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        logging.info(f"Logging configured. All logs will be saved to: {log_file}")

    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.error("Failed to configure advanced logging.", exc_info=True)
