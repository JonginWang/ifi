#!/usr/bin/env python3
"""
    TekScopeController
    ==================

    This module contains the TekScopeController class for controlling Tektronix scopes.
"""


import logging
import numpy as np
from typing import Optional, Union
from tm_devices import DeviceManager
from tm_devices.drivers import MDO3, MSO5
from tm_devices.helpers import PYVISA_PY_BACKEND
from ifi.utils.common import LogManager

# Get logger instance
LogManager()
LOGGER = logging.getLogger(__name__)

# By creating a type alias for the supported scope models,
# it is easy to refer to the collection of Tektronix scope device classes.
Scope = Union[MDO3, MSO5]


class TekScopeController:
    """
    A controller class for Tektronix scopes using the tm-devices library.

    This class simplifies interaction with the scope by wrapping the
    powerful tm-devices library. It handles device discovery, connection,
    and data acquisition.
    """

    def __init__(self):
        """Initializes the TekScopeController."""
        with DeviceManager(verbose=True) as device_manager:
            self.dm = device_manager
         
        # Enable resetting the devices when connecting and closing
        self.dm.setup_cleanup_enabled = True
        self.dm.teardown_cleanup_enabled = True
        # Use the PyVISA-py backend
        self.dm.visa_library = PYVISA_PY_BACKEND

        # self.scope: Optional[ ] = None

    def list_devices(self) -> list[str]:
        """
        Return a list of all discovered device strings.
        
        The string contains information like model and serial number, 
        suitable for display in a GUI.
        """
        # The DeviceManager's __str__ representation is a good summary.
        # Here we create a list of identifiers for each device.
        return [f"{dev.model} - {dev.serial}" for dev in self.dm.devices]

    def connect(self, device_identifier: str) -> bool:
        """
        Connects to a specific scope using its identifier string.

        Args:
            device_identifier: The identifier string from the list_devices() method.

        Returns:
            True if connection is successful, False otherwise.
        """
        # The identifier is in "MODEL - SERIAL" format. We need the serial.
        try:
            serial = device_identifier.split(' - ')[1]
            self.scope = self.dm.get_device(serial=serial)
            # Setting the scope object to none if it is not a scope.
            if not hasattr(self.scope, "idn_string") or self.scope.idn_string is None:
                LOGGER.error("Device is not found or does not have an identification string.")
                self.scope = None
                raise TypeError("Device is not a Scope")
            if not isinstance(self.scope, (MDO3, MSO5)):
                if self.scope.idn_string is not None:
                    LOGGER.error(f"Device is not a Scope: {self.scope.idn_string}")
                self.scope = None
                raise TypeError("Device is not a Scope")
            
            LOGGER.info(f"Successfully connected to: {self.scope.idn_string}")
            return True
        except (IndexError, KeyError, TypeError) as e:
            LOGGER.error(f"Failed to connect to {device_identifier}: {e}")
            self.scope = None
            return False

    def disconnect(self):
        """Disconnects from the currently connected scope."""
        if self.scope:
            LOGGER.info(f"Disconnecting from {self.scope.device_name}")
            self.scope.close()
            self.scope = None
        # self.dm.close_all_devices() # Use this for a full cleanup

    def get_idn(self) -> str:
        """Returns the identification string of the connected scope."""
        return self.scope.idn_string if self.scope else "Not connected"

    def get_trigger_state(self) -> str:
        """Gets the current state of the trigger system."""
        return self.scope.trigger.state if self.scope else "UNKNOWN"

    def get_waveform_data(self, channel: str = " ") -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Acquires waveform data from the specified channel using high-level tm-devices methods.

        This method leverages the library's internal logic to handle data scaling and formatting.

        Args:
            channel: The channel to acquire data from (e.g., "CH1").

        Returns:
            A tuple containing two NumPy arrays (time, voltage), or None on failure.
        """
        if not self.scope:
            LOGGER.error("Cannot get waveform, no scope connected.")
            return None

        try:
            # Use high-level methods to configure the data source and retrieve the waveform.
            # This is a more realistic representation of how tm-devices works.
            self.scope.commands.data.source.write(channel)
            
            # The waveform object returned by .get() should contain scaled x and y data.
            waveform = self.scope.commands.curve.get()
            
            time_values = waveform.x
            voltage_values = waveform.y

            return time_values, voltage_values

        except Exception as e:
            LOGGER.error(f"An unexpected error occurred while getting waveform for {channel}: {e}")
            return None 