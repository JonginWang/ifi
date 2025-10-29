#!/usr/bin/env python3
"""
    Common Utilities
    ================

    This module contains the functions for common utilities.
    It includes the LogManager class for managing logging configuration,
    the CustomFormatter class for formatting log messages,
    the assign_kwargs decorator for injecting keyword arguments into a method call,
    and the FlatShotList class for parsing and flattening a list of shot numbers and file paths.
"""

import sys
import logging
import atexit
from pathlib import Path
from typing import Callable, List, Union
from datetime import datetime
import functools
import numpy as np

class LogManager:
    """
    A Singleton class to manage logging configuration for the application.
    It ensures that logging is configured only once per session.
    
    Usage:
        LogManager(level="WARNING")  # Configure logging on first call
        LogManager()                 # Subsequent calls do nothing but return the instance
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, level="WARNING"):
        # The __init__ will be called every time, but the setup logic
        # is guarded by the '_configured' flag.
        if hasattr(self, '_configured') and self._configured:
            # If already configured, check if we need to update the level
            if hasattr(self, '_current_level') and self._current_level != level:
                self._update_log_level(level)
            return

        self._setup_logging(level)
        self._configured = True
        self._current_level = level

    def _log_shutdown(self):
        """Function registered with atexit to log a shutdown message."""
        logging.info(f"\n{'  Logging ended  '.center(80, '=')}")

    def _setup_logging(self, level):
        """
        Configures the root logger. This method is called only once.
        It logs to a unique file per execution and streams to the console.
        """
        if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = getattr(logging, level.upper())
        else:
            log_level = logging.WARNING

        try:
            # Get the calling script name for log file naming
            if hasattr(sys.modules['__main__'], '__file__'):
                main_script_path = sys.modules['__main__'].__file__
                script_name = Path(main_script_path).stem
            else:
                # IDE/iPython 환경 감지
                if 'ipython' in sys.modules or 'jupyter' in sys.modules:
                    script_name = 'ipython'
                elif hasattr(sys, 'ps1'):  # Interactive Python shell
                    script_name = 'interactive'
                else:
                    script_name = 'unknown'
            
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            date_str = datetime.now().strftime('%y%m%d_%H%M%S')
            log_file = log_dir / f"{date_str}_{script_name}.log"
            
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
            console_formatter = CustomFormatter('%(levelname)s | %(message)s')
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            
            logging.info(f"\n{'  Logging started  '.center(80, '=')}\n")
            logging.info(f"\nAll logs for this execution will be saved to: {log_file}")
            
            atexit.register(self._log_shutdown)

        except Exception:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            logging.error("  Failed to configure advanced logging  ".center(80, "="), exc_info=True)

    def _update_log_level(self, new_level):
        """
        Update the log level for both console and file handlers.
        This allows changing the level after initial setup.
        """
        if new_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = getattr(logging, new_level.upper())
        else:
            log_level = logging.WARNING

        root_logger = logging.getLogger()
        
        # Update console handler level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
                break
        
        self._current_level = new_level
        logging.debug("\n" + f"Log level updated to: {new_level}".center(80, "=") + "\n")

    def get_logger(self, name: str = None, level: str = None) -> logging.Logger:
        """
        Get a logger with a specific name and level.
        This allows different parts of the application to have different log levels.
        
        Args:
            name: Logger name (e.g., 'ifi.analysis', 'ifi.test')
            level: Log level for this specific logger
            
        Returns:
            Logger instance
        """
        if name is None:
            return logging.getLogger()
        
        logger = logging.getLogger(name)
        
        if level is not None:
            if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.setLevel(getattr(logging, level.upper()))
            else:
                logger.setLevel(logging.WARNING)
        
        return logger

    def create_specialized_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        """
        Create a specialized logger for a specific purpose.
        This logger will inherit the root logger's handlers but can have its own level.
        
        Args:
            name: Logger name (e.g., 'ifi.analysis', 'ifi.test')
            level: Log level for this logger
            
        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)
        
        if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger.setLevel(getattr(logging, level.upper()))
        else:
            logger.setLevel(logging.INFO)
        
        return logger

    def setup_ide_logging(self, level: str = "INFO") -> logging.Logger:
        """
        Setup logging specifically for IDE environments (Spyder, Jupyter, etc.).
        This creates a more user-friendly logging setup for interactive development.
        
        Args:
            level: Log level for IDE logging
            
        Returns:
            Logger instance configured for IDE use
        """
        # Create a specialized logger for IDE use
        ide_logger = self.create_specialized_logger('ifi.ide', level)
        
        # Add a custom handler for IDE that shows more context
        if not any(isinstance(h, logging.StreamHandler) for h in ide_logger.handlers):
            ide_handler = logging.StreamHandler()
            ide_handler.setLevel(getattr(logging, level.upper()))
            
            # IDE-friendly formatter (simpler, more readable)
            ide_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            ide_handler.setFormatter(ide_formatter)
            ide_logger.addHandler(ide_handler)
        
        return ide_logger


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
                record.name = record.name[:self.limit - len(self.ellipsis)] + self.ellipsis
            elif len(record.name) < self.limit:
                # Use ljust for left alignment with fill char
                record.name = record.name.ljust(self.limit, self.fill_char)
        else:
            # BUG FIX: Assign a string, not a list
            record.name = self.none_char * self.limit

        if isinstance(record.levelname, str):
            record.levelname = record.levelname.ljust(self.level_limit, ' ')
        # Format the record
        record_formatted = super().format(record)
        # Restore the original name
        record.name = name_original
        record.levelname = level_original

        return record_formatted


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

def ensure_dir_exists(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def ensure_str_path(path_like: Union[str, Path]) -> str:
    """Coerce to string and normalize separators to forward slashes."""
    return normalize_to_forward_slash(path_like)

def normalize_to_forward_slash(path_like: Union[str, Path]) -> str:
    """Return a string path using forward slashes. Safe for Windows and pandas."""
    if isinstance(path_like, Path):
        return path_like.as_posix()
    return str(path_like).replace("\\", "/")

def resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    
    Args:
        relative_path: Relative path to resource
        
    Returns:
        Absolute path to resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.parent
    
    return base_path / relative_path
