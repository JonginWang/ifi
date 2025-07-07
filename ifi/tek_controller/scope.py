import pyvisa as visa
import time
import numpy as np

class TektronixScope:
    """
    A class to control a Tektronix MDO3000 series oscilloscope via VISA.
    Enhanced with features inspired by community repositories.
    """
    def __init__(self, resource_string: str):
        """
        Initializes the VISA resource manager and sets the resource string.

        :param resource_string: The VISA resource string for the oscilloscope.
        """
        self.resource_string = resource_string
        self.rm = visa.ResourceManager()
        self.instrument = None

    @staticmethod
    def find_scopes() -> list[str]:
        """
        Finds all connected Tektronix instruments.
        :return: A list of VISA resource strings.
        """
        try:
            rm = visa.ResourceManager()
            resources = rm.list_resources()
            # Tektronix Vendor ID is 0x0699
            tek_scopes = [r for r in resources if '0x0699' in r]
            return tek_scopes
        except Exception:
            return []

    def connect(self):
        """
        Connects to the oscilloscope.
        Raises:
            visa.errors.VisaIOError: If connection fails.
        """
        self.instrument = self.rm.open_resource(self.resource_string)
        self.instrument.timeout = 20000  # Increased timeout for longer operations
        # Clear the event status register and queue
        self.instrument.write('*CLS')
        # Set response header to off for cleaner query results
        self.instrument.write('HEADER OFF')
        
    def disconnect(self):
        """
        Disconnects from the oscilloscope.
        """
        if self.instrument:
            self.instrument.close()
            self.instrument = None

    def get_id(self) -> str:
        """
        Queries the oscilloscope's identification string.
        :return: The identification string.
        """
        return self.instrument.query('*IDN?').strip()

    def get_trigger_state(self) -> str:
        """
        Gets the current state of the trigger system.
        Common states: READY, TRIGGER, SAVE, ARMED
        :return: The trigger state string.
        """
        return self.instrument.query('TRIGger:STATE?').strip()

    def set_usb_drive(self, drive_letter: str = "E:"):
        """
        Sets the current working directory to the specified USB drive.
        :param drive_letter: The drive letter of the USB port (e.g., "E:").
        """
        self.instrument.write(f'FILESYSTEM:CWD "{drive_letter}/"')

    def list_files(self, extension: str = ".wfm") -> list[str]:
        """
        Lists files with a specific extension in the current directory.
        :param extension: The file extension to filter by (e.g., ".wfm").
        :return: A list of filenames.
        """
        files_str = self.instrument.query('FILESYSTEM:LDIR?')
        # The output is a comma-separated string, sometimes with quotes.
        # This parsing needs to be robust based on the actual scope output.
        files = [f.strip('"') for f in files_str.strip().split(',') if f.lower().endswith(extension)]
        return files

    def recall_waveform(self, filename: str, ref_channel: str = "REF1"):
        """
        Recalls a waveform file from the USB drive to a reference channel.
        :param filename: The name of the file on the USB drive.
        :param ref_channel: The reference channel to load into (e.g., "REF1").
        """
        # Assuming the CWD is already set to the USB drive root
        self.instrument.write(f'RECALL:WAVEFORM "E:/{filename}", {ref_channel}')
        # Wait for the operation to complete
        self.instrument.query('*OPC?')

    def get_waveform_data(self, channel: str = "CH1", start=1, stop=10000) -> tuple | None:
        """
        Downloads waveform data and scaling preamble for a specific channel.
        """
        try:
            self.instrument.write(f'DATA:SOURCE {channel}')
            self.instrument.write('DATA:ENCDG SRIbinary') # Signed, Big-endian binary
            self.instrument.write(f'DATA:START {start}')
            record_length = int(self.instrument.query('HORizontal:RECOrdlength?'))
            self.instrument.write(f'DATA:STOP {record_length}')
            
            # Get waveform scaling factors
            ymult = float(self.instrument.query('WFMOutpre:YMULT?'))
            yzero = float(self.instrument.query('WFMOutpre:YZERO?'))
            yoff = float(self.instrument.query('WFMOutpre:YOFF?'))
            xincr = float(self.instrument.query('WFMOutpre:XINCR?'))

            # Get waveform data
            raw_data = self.instrument.query_binary_values('CURVE?', datatype='h', is_big_endian=True, container=np.array)
            
            # Convert raw data to voltage and time
            voltage_values = (raw_data - yoff) * ymult + yzero
            time_values = np.arange(0, xincr * len(voltage_values), xincr)
            
            return time_values, voltage_values

        except visa.errors.VisaIOError as e:
            print(f"Error getting waveform for {channel}: {e}")
            return None

    def save_all_channels_to_file(self, base_filename: str, save_path: str):
        """
        Saves waveform data for all available channels (CH1-4) to separate CSV files.
        """
        import os
        import pandas as pd

        for i in range(1, 5):
            ch = f"CH{i}"
            data = self.get_waveform_data(ch)
            if data:
                time_vals, voltage_vals = data
                df = pd.DataFrame({'Time (s)': time_vals, 'Voltage (V)': voltage_vals})
                
                # Create directory if it doesn't exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                filename = os.path.join(save_path, f"{base_filename}_{ch}.csv")
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")

    def delete_file(self, filename: str):
        """
        Deletes a file from the USB drive.
        :param filename: The name of the file to delete.
        """
        # Assuming the CWD is already set to the USB drive root
        self.instrument.write(f'FILESYSTEM:DELETE "E:/{filename}"')
        self.instrument.query('*OPC?')

    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect() 