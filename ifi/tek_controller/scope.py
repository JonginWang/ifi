#!/usr/bin/env python3
"""
TekScopeController
==================

This module contains the TekScopeController class for controlling Tektronix scopes.

Args:
    logger(logging.Logger): Logger instance for the TekScopeController class.
    scope(Union[MDO3, MSO5]): Type alias for the supported scope models.

Classes:
    TekScopeController: A controller class for Tektronix scopes using the tm-devices library.

Functions:
    list_devices: Function to list all discovered device strings.
    connect: Function to connect to a specific scope using its identifier string.
    disconnect: Function to disconnect from the currently connected scope.
    get_idn: Function to get the identification string of the connected scope.
    get_trigger_state: Function to get the current state of the trigger system.
    get_waveform_data: Function to acquire waveform data from the specified channel.
"""

import numpy as np
import pandas as pd
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Union
from tm_devices import DeviceManager
from tm_devices.drivers import MDO3, MSO5
from tm_devices.helpers import PYVISA_PY_BACKEND
try:
    from ..utils.common import LogManager, log_tag
    from ..utils.file_io import convert_to_hdf5
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag
    from ifi.utils.file_io import convert_to_hdf5

# Get logger instance
LogManager()
logger = LogManager().get_logger(__name__)


class ScopeState(Enum):
    """
    Enum class representing the internal state of the TekScopeController.

    Attributes:
        IDLE: No active operation is in progress.
        CONNECTING: A connection attempt to the scope is in progress.
        ACQUIRING: Waveform acquisition from the scope is in progress.
        SAVING: Data saving is in progress (reserved for higher-level workflows).
        ERROR: An error has occurred; controller should be inspected or reset.
    """

    IDLE = auto()
    CONNECTING = auto()
    ACQUIRING = auto()
    SAVING = auto()
    ERROR = auto()


# By creating a type alias for the supported scope models,
# it is easy to refer to the collection of Tektronix scope device classes.
scope = Union[MDO3, MSO5]


class TekScopeController:
    """
    A controller class for Tektronix scopes using the tm-devices library.

    This class simplifies interaction with the scope by wrapping the
    powerful tm-devices library. It handles device discovery, connection,
    and data acquisition.

    Args:
        None

    Attributes:
        dm(DeviceManager): The device manager for the TekScopeController.
        scope(Union[MDO3, MSO5]): The scope object for the TekScopeController.

    Methods:
        list_devices: Function to list all discovered device strings.
        connect: Function to connect to a specific scope using its identifier string.
        disconnect: Function to disconnect from the currently connected scope.
        get_idn: Function to get the identification string of the connected scope.
        get_trigger_state: Function to get the current state of the trigger system.
        get_waveform_data: Function to acquire waveform data from the specified channel.
    """

    def __init__(self) -> None:
        """
        Initialize the TekScopeController.

        The controller starts in the IDLE state with no connected scope but with
        a ready-to-use DeviceManager based on the tm-devices library.
        """
        with DeviceManager(verbose=True) as device_manager:
            self.dm: DeviceManager = device_manager

        # Enable resetting the devices when connecting and closing
        self.dm.setup_cleanup_enabled = True
        self.dm.teardown_cleanup_enabled = True
        # Use the PyVISA-py backend
        self.dm.visa_library = PYVISA_PY_BACKEND

        # Currently connected scope instance (if any)
        self.scope: Optional[scope] = None

        # Internal controller state for higher-level orchestration
        self.state: ScopeState = ScopeState.IDLE

    def is_idle(self) -> bool:
        """
        Check whether the controller is currently idle.

        Returns:
            bool: True if the controller state is IDLE, False otherwise.
        """
        return self.state == ScopeState.IDLE

    def is_busy(self) -> bool:
        """
        Check whether the controller is currently performing a blocking operation.

        Returns:
            bool: True if the controller is in CONNECTING, ACQUIRING or SAVING
            state, False otherwise.
        """
        return self.state in {
            ScopeState.CONNECTING,
            ScopeState.ACQUIRING,
            ScopeState.SAVING,
        }

    def set_error(self) -> None:
        """
        Set the controller state to ERROR.

        This helper can be used by higher-level workflows when they detect
        unrecoverable failures outside of the low-level controller methods.
        """
        self.state = ScopeState.ERROR

    def list_devices(self) -> list[str]:
        """
        Return a list of all discovered device strings.

        The string contains information like model and serial number,
        suitable for display in a GUI.

        Returns:
            list[str]: A list of all discovered device strings.

        Raises:
            Exception: If an error occurs while listing devices.
        """
        # The DeviceManager's __str__ representation is a good summary.
        # Here we create a list of identifiers for each device.
        return [f"{dev.model} - {dev.serial}" for dev in self.dm.devices]

    def connect(self, device_identifier: str) -> bool:
        """
        Connects to a specific scope using its identifier string.

        Args:
            device_identifier(str): The identifier string from the list_devices() method.

        Returns:
            True if connection is successful, False otherwise.

        Raises:
            Exception: If an error occurs while connecting to the scope.
        """
        # The identifier is in "MODEL - SERIAL" format. We need the serial.
        self.state = ScopeState.CONNECTING
        try:
            serial = device_identifier.split(" - ")[1]
            self.scope = self.dm.get_device(serial=serial)
            # Setting the scope object to none if it is not a scope.
            if not hasattr(self.scope, "idn_string") or self.scope.idn_string is None:
                logger.error(
                    f"{log_tag('TEKSC', 'CONN ')} Device is not found or does not have an identification string."
                )
                self.scope = None
                raise TypeError("Device is not a Scope")
            if not isinstance(self.scope, (MDO3, MSO5)):
                if self.scope.idn_string is not None:
                    logger.error(
                        f"{log_tag('TEKSC', 'CONN ')} Device is not a Scope: {self.scope.idn_string}"
                    )
                self.scope = None
                raise TypeError("Device is not a Scope")

            logger.info(
                f"{log_tag('TEKSC', 'CONN ')} Successfully connected to: {self.scope.idn_string}"
            )
            self.state = ScopeState.IDLE
            return True
        except (IndexError, KeyError, TypeError) as e:
            logger.error(
                f"{log_tag('TEKSC', 'CONN ')} Failed to connect to {device_identifier}: {e}"
            )
            self.scope = None
            self.state = ScopeState.ERROR
            return False

    def disconnect(self) -> None:
        """
        Disconnects from the currently connected scope.

        The controller state is reset to IDLE regardless of whether a scope was
        previously connected.
        """
        if self.scope:
            logger.info(
                f"{log_tag('TEKSC', 'DISC ')} Disconnecting from {self.scope.device_name}"
            )
            self.scope.close()
            self.scope = None

        # When no scope is connected, the controller is conceptually idle.
        self.state = ScopeState.IDLE
        # self.dm.close_all_devices() # Use this for a full cleanup

    def get_idn(self) -> str:
        """
        Returns the identification string of the connected scope.

        Returns:
            str: The identification string of the connected scope.
        """
        return self.scope.idn_string if self.scope else "Not connected"

    def get_trigger_state(self) -> str:
        """
        Gets the current state of the trigger system.

        Returns:
            str: The current state of the trigger system.
        """
        return self.scope.trigger.state if self.scope else "UNKNOWN"

    def get_waveform_data(
        self, channel: str = " "
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Acquires waveform data from the specified channel using high-level tm-devices methods.

        This method leverages the library's internal logic to handle data scaling and formatting.

        Args:
            channel(str): The channel to acquire data from (e.g., "CH1").

        Returns:
            A tuple containing two NumPy arrays (time, voltage), or None on failure.
        """
        if not self.scope:
            logger.error(
                f"{log_tag('TEKSC', 'GETWF')} Cannot get waveform, no scope connected."
            )
            self.state = ScopeState.ERROR
            return None

        self.state = ScopeState.ACQUIRING
        try:
            # Use high-level methods to configure the data source and retrieve the waveform.
            # This is a more realistic representation of how tm-devices works.
            self.scope.commands.data.source.write(channel)

            # The waveform object returned by .get() should contain scaled x and y data.
            waveform = self.scope.commands.curve.get()

            time_values = waveform.x
            voltage_values = waveform.y

            self.state = ScopeState.IDLE
            return time_values, voltage_values

        except Exception as e:
            logger.error(
                f"{log_tag('TEKSC', 'GETWF')} An unexpected error occurred while getting waveform for {channel}: {e}"
            )
            self.state = ScopeState.ERROR
            return None

    def acquire_data(
        self, channels: list[str]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Acquire waveform data for a list of channels.

        This method calls :meth:`get_waveform_data` for each requested channel
        and aggregates the results into a dictionary.

        Args:
            channels(list[str]): Channel names to acquire (e.g. ``['CH1', 'CH2']``).

        Returns:
            dict[str, tuple[np.ndarray, np.ndarray]]: Mapping from channel name
                to ``(time, voltage)`` arrays. Channels that fail to acquire
                are omitted from the dictionary.

        Notes:
            - The controller state is set to ``ACQUIRING`` at the beginning of
              the multi-channel acquisition and restored to ``IDLE`` once all
              requested channels have been processed, unless a fatal error
              occurs.
            - Individual channel acquisition errors are logged and skipped,
              allowing partial results to be returned.
        """
        if not self.scope:
            logger.error(
                f"{log_tag('TEKSC', 'ACQ  ')} Cannot acquire data, no scope connected."
            )
            self.state = ScopeState.ERROR
            return {}

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.state = ScopeState.ACQUIRING

        for ch in channels:
            try:
                data = self.get_waveform_data(channel=ch)
                if data is not None:
                    results[ch] = data
                else:
                    logger.warning(
                        f"{log_tag('TEKSC', 'ACQ  ')} Skipping channel {ch}: no data returned."
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    f"{log_tag('TEKSC', 'ACQ  ')} Error acquiring data for {ch}: {exc}"
                )

        # If at least one channel succeeded, return to IDLE, otherwise mark ERROR.
        if results:
            self.state = ScopeState.IDLE
        else:
            self.state = ScopeState.ERROR

        return results

    def save_data(
        self,
        directory: Path | str,
        shot_code: int,
        suffix: str,
        data: dict[str, np.ndarray],
        file_format: str = "CSV",
    ) -> Optional[Path]:
        """
        Save acquired waveform data to a CSV file using a standard naming scheme.

        The output filename is formatted as ``{shot_code}{suffix}.<ext>`` and
        written into the provided directory. The input ``data`` dictionary is
        converted into a pandas DataFrame before being written.

        Args:
            directory: Target directory where the CSV file will be saved.
            shot_code: Shot number used as the filename prefix.
            suffix: Suffix string appended to the filename (for example,
                ``\"_056\"`` or ``\"_ALL\"``).
            data: Mapping from column name to NumPy array. All arrays should
                have the same length (e.g. ``{\"TIME\": ..., \"CH1\": ...}``).
            file_format: Output format. Currently supported values are
                ``\"CSV\"`` and ``\"HDF5\"``. CSV is the default.

        Returns:
            pathlib.Path | None: The full path to the saved CSV file on
            success, or None if saving failed.

        Notes:
            - The controller state is set to ``SAVING`` at the beginning of
              this method and guaranteed to be restored to ``IDLE`` in a
              ``finally`` block, even if an exception is raised.
            - File-system errors are logged and result in ``None`` being
              returned; callers can inspect the return value to detect
              failures.
        """
        directory_path = Path(directory)
        file_format_upper = file_format.upper()
        if file_format_upper == "CSV":
            filename = f"{shot_code}{suffix}.csv"
        elif file_format_upper == "HDF5":
            filename = f"{shot_code}{suffix}.h5"
        else:
            logger.error(
                f"{log_tag('TEKSC', 'SAVE ')} Unsupported file format: {file_format}"
            )
            return None

        filepath = directory_path / filename

        logger.info(
            f"{log_tag('TEKSC', 'SAVE ')} Saving data to {filepath} "
            f"with columns: {list(data.keys())}"
        )

        self.state = ScopeState.SAVING
        try:
            directory_path.mkdir(parents=True, exist_ok=True)

            if file_format_upper == "CSV":
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
            elif file_format_upper == "HDF5":
                # First write a temporary CSV in memory via pandas, then
                # convert it into HDF5 using the shared utility.
                temp_csv = directory_path / f"{shot_code}{suffix}.csv"
                df = pd.DataFrame(data)
                df.to_csv(temp_csv, index=False)
                convert_to_hdf5(temp_csv, filepath)

            return filepath
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                f"{log_tag('TEKSC', 'SAVE ')} Failed to save data to {filepath}: {exc}"
            )
            return None
        finally:
            # Always return to IDLE regardless of success or failure.
            self.state = ScopeState.IDLE
