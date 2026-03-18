#!/usr/bin/env python3
"""
Logging Manager (Customized)
=============================

This module contains the "LogManager" class for managing logging configuration.

Classes:
    LogManager: A Singleton class to manage logging configuration for the application.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import atexit
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from .log_formatter import CustomFormatter, log_tag

_INVALID_LOG_BASENAME_RE = re.compile(r'[<>:"/\\|?*]+')


def _sanitize_log_basename(name: str) -> str:
    sanitized = _INVALID_LOG_BASENAME_RE.sub("_", str(name)).strip(" ._")
    return sanitized or "interactive"


class LogManager:
    """
    A Singleton class to manage logging configuration for the application.
    It ensures that logging is configured only once per session.

    Args:
        level(str): The log level to configure.

    Attributes:
        _instance(cls): The singleton instance of the LogManager class.
        _configured(bool): A flag to indicate if the logging has been configured.
        _current_level(str): The current log level.

    Methods:
        __new__: The singleton instance of the LogManager class.
        __init__: Initialize the LogManager class.
        _log_shutdown: Log a shutdown message.
        _setup_logging: Configure the root logger.
        _update_log_level: Update the log level for both console and file handlers.
        get_logger: Get a logger with a specific name and level.
        create_specialized_logger: Create a specialized logger for a specific purpose.
        setup_ide_logging: Setup logging specifically for IDE environments (Spyder, Jupyter, etc.).

    Returns:
        None

    Examples:
        ```python
        from .log_manager import LogManager

        # 1. Basic usage for call logger
        LogManager(level="WARNING")  # Configure logging on first call
        LogManager()                 # Subsequent calls do nothing but return the instance

        # 2. Get the logger for the analysis module with WARNING level
        LogManager().get_logger("ifi.analysis", "WARNING")

        # 3. Create a specialized logger for the analysis module with DEBUG level
        analysis_logger = LogManager().create_specialized_logger("ifi.analysis", "DEBUG")

        # 4. Create a class with a logger to specify the logger name
        class SomeThing:
            def __init__(self):
                self._setup_logger()

            def _setup_logger(self):
                log_manager = LogManager(level="INFO")
                self.logger = log_manager.get_logger(f"ifi.{self.__class__.__name__.lower()}")
        ```
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the LogManager class. This is called only once.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The singleton instance of the LogManager class.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, level="WARNING"):
        """
        Initialize the LogManager class. This method is called every time.

        Args:
            level(str): The log level to configure.

        Returns:
            None
        """
        # The __init__ will be called every time, but the setup logic
        # is guarded by the '_configured' flag.
        if hasattr(self, "_configured") and self._configured:
            # If already configured, check if we need to update the level
            if hasattr(self, "_current_level") and self._current_level != level:
                self._update_log_level(level)
            return

        self._setup_logging(level)
        self._configured = True
        self._current_level = level

    def _log_shutdown(self):
        """Function registered with atexit to log a shutdown message."""
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                root_logger.removeHandler(handler)

        if not root_logger.handlers:
            return

        logging.info(f"\n{log_tag('LOGS','SHUTD')} Logging ended")

    def _setup_logging(self, level):
        """
        Configures the root logger. This method is called only once.
        It logs to a unique file per execution and streams to the console.

        Args:
            level(str): The log level to configure.

        Returns:
            None
        """
        if level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = getattr(logging, level.upper())
        else:
            log_level = logging.WARNING

        try:
            # Get the calling script name for log file naming
            if hasattr(sys.modules["__main__"], "__file__"):
                main_script_path = sys.modules["__main__"].__file__
                script_name = _sanitize_log_basename(Path(main_script_path).stem)
            else:
                script_name = "interactive"

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            date_str = datetime.now().strftime("%y%m%d_%H%M%S")
            log_file = log_dir / f"{date_str}_{script_name}.log"

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)

            if root_logger.hasHandlers():
                root_logger.handlers.clear()

            # File Handler
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = CustomFormatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = CustomFormatter("%(levelname)s | %(message)s")
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

            logging.info(f"\n{log_tag('LOGS','START')} Logging started\n")
            logging.info(f"\n{log_tag('LOGS','START')} All logs for this execution will be saved to: {log_file}")

            atexit.register(self._log_shutdown)

        except Exception:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
            logging.error(
                f"{log_tag('LOGS','INIT')} Failed to configure advanced logging",
                exc_info=True,
            )

    def _update_log_level(self, new_level):
        """
        Update the log level for both console and file handlers.
        This allows changing the level after initial setup.

        Args:
            new_level(str): The new log level to set.

        Returns:
            None
        """
        if new_level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = getattr(logging, new_level.upper())
        else:
            log_level = logging.WARNING

        root_logger = logging.getLogger()

        # Update console handler level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(log_level)
                break

        self._current_level = new_level
        logging.debug(
            f"\n{log_tag('LOGS','UPDAT')} Log level updated to: {new_level}\n"
        )

    def get_logger(self, name: str = None, level: str = None) -> logging.Logger:
        """
        Get a logger with a specific name and level.
        This allows different parts of the application to have different log levels.

        Args:
            name(str): Logger name (e.g., 'ifi.analysis', 'ifi.test')
            level(str): Log level for this specific logger

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

    def create_specialized_logger(
        self, name: str, level: str = "INFO"
    ) -> logging.Logger:
        """
        Create a specialized logger for a specific purpose.
        This logger will inherit the root logger's handlers but can have its own level.

        Args:
            name(str): Logger name (e.g., 'ifi.analysis', 'ifi.test')
            level(str): Log level for this logger

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
            level(str): Log level for IDE logging

        Returns:
            Logger instance configured for IDE use
        """
        # Create a specialized logger for IDE use
        ide_logger = self.create_specialized_logger("ifi.ide", level)

        # Add a custom handler for IDE that shows more context
        if not any(isinstance(h, logging.StreamHandler) for h in ide_logger.handlers):
            ide_handler = logging.StreamHandler()
            ide_handler.setLevel(getattr(logging, level.upper()))

            # IDE-friendly formatter (simpler, more readable)
            ide_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
            ide_handler.setFormatter(ide_formatter)
            ide_logger.addHandler(ide_handler)

        return ide_logger
