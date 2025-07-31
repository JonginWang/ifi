import sys
import os
import functools
import numpy as np
from typing import Callable, List, Union
import logging
import atexit
from datetime import datetime


class LogManager:
    """
    A Singleton class to manage logging configuration for the application.
    It ensures that logging is configured only once per session.
    
    Usage:
        LogManager(level="INFO")  # Configure logging on first call
        LogManager()              # Subsequent calls do nothing but return the instance
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, level="INFO"):
        # The __init__ will be called every time, but the setup logic
        # is guarded by the '_configured' flag.
        if hasattr(self, '_configured') and self._configured:
            return

        self._setup_logging(level)
        self._configured = True

    def _log_shutdown(self):
        """Function registered with atexit to log a shutdown message."""
        logging.info(f"\n{'  Logging ended  '.center(60, '=')}")

    def _setup_logging(self, level):
        """
        Configures the root logger. This method is called only once.
        It logs to a unique file per execution and streams to the console.
        """
        if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = getattr(logging, level.upper())
        else:
            log_level = logging.INFO

        try:
            if hasattr(sys.modules['__main__'], '__file__'):
                main_script_path = sys.modules['__main__'].__file__
                script_name = os.path.splitext(os.path.basename(main_script_path))[0]
            else:
                script_name = 'interactive'
            
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            date_str = datetime.now().strftime('%y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"{date_str}_{script_name}.log")
            
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)

            if root_logger.hasHandlers():
                root_logger.handlers.clear()

            # File Handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = CustomFormatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
            logging.info(f"\n{'  Logging started  '.center(60, '=')}")
            logging.info(f" All logs for this execution will be saved to: {log_file}")
            
            atexit.register(self._log_shutdown)

        except Exception:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            logging.error("Failed to configure advanced logging.", exc_info=True)


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that right-aligns the logger name to 15 characters
    and truncates with an ellipsis if it's longer.
    The fill_char (default: '#') is used to fill the name to the limit if it's shorter.
    The none_char (default: '.') is used to fill the name to the limit if it's not a string.
    """
    def __init__(self, fmt=None, datefmt=None, style='%', limit=15, ellipsis="*", fill_char='#', none_char='.'):
        super().__init__(fmt, datefmt, style)
        self.datefmt = '%Y-%m-%d %H:%M:%S'
        self.limit = limit
        self.ellipsis = ellipsis
        self.fill_char = fill_char
        self.none_char = none_char
        # if len(ellipsis) ==3:
        #     self.ellipsis = ellipsis
        # elif len(ellipsis) > 3:
        #     self.ellipsis = ellipsis[:3]
        # else:
        #     self.ellipsis = ellipsis.ljust(3, '*')

    def format(self, record):
        # Store the original name
        name_original = record.name
        if isinstance(record.name, str):
            if len(record.name) > self.limit:  # self.limit = 15
                # Truncate and add ellipsis
                record.name = record.name[:self.limit - len(self.ellipsis)] + self.ellipsis
            elif len(record.name) < self.limit:
                # Use ljust for left alignment with fill char
                record.name = record.name.ljust(self.limit, self.fill_char)
        else:
            # BUG FIX: Assign a string, not a list
            record.name = self.none_char * self.limit
        # Format the record
        record_formatted = super().format(record)
        # Restore the original name
        record.name = name_original
        return record_formatted


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
        
class FlatShotList:
    """
    Parses and flattens a nested list of shot numbers (int) and file paths (str),
    separating them into distinct lists of unique integers and strings.

    This class handles various input formats including single integers, strings
    with ranges (e.g., "12345:12349:2"), file paths, and nested lists thereof.
    It is designed to prepare a clean query list for NAS_DB and VEST_DB.

    Attributes:
        nums (List[int]): A sorted list of unique shot numbers.
        paths (List[str]): A sorted list of unique file paths.
        all (List[Union[int, str]]): A combined list of unique paths and numbers.
    """
    ALLOWED_EXTENSIONS = ['.csv', '.dat', '.mat', '.isf', '.wfm']

    def __init__(self, raw_list: List[Union[int, str, list]]):
        self.nums: List[int] = []
        self.paths: List[str] = []
        
        if raw_list:
            self._flatten_and_parse(raw_list)

        self.nums = sorted(list(set(self.nums)))
        self.paths = sorted(list(set(self.paths)))
        self.all = self.paths + self.nums

    def _flatten_and_parse(self, shot_list: list):
        """Recursively flattens and parses the input list."""
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
                if ':' in item:
                    try:
                        parts = list(map(int, item.split(':')))
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
