<div align="center">
    <img src="./image/IFI_icon01.png" width="300">
</div>

# Tektronix Oscilloscope Data Automator

This project provides a graphical user interface (GUI) to automate the process of transferring waveform data from a Tektronix MDO3000 series oscilloscope's USB stick to a computer.

## Features

- **GUI Interface:** Easy-to-use interface to start and stop the automation process.
- **State Machine Logic:** Robustly handles the process of connecting, listing files, transferring data, and deleting files from the USB drive.
- **Modular Design:** The oscilloscope control logic is separated from the GUI, making it easy to maintain and extend.

## How to Run

1.  **Change to the parent directory of `ifi`**

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python -m ifi.main
    ```

## Project Structure

```
Package root/
├── cache/
├── docs/
├── ifi/
│   ├── __init__.py
│   ├── main.py
│   ├── gui/
│   │   ├── __init__.py
│   │   └── main_window.py
│   ├── tek_controller/
│   ├── db_controller/
│   ├── analysis/
│   ├── __init__.py
│   └── scope.py
│   ├── README.md
├── tests/
├── run.py
├── README.md
└── requirements.txt
```

## Building an Executable

You can package this application into a single executable file using PyInstaller. For best results, it is **highly recommended** to build inside a clean virtual environment to avoid bundling unnecessary large libraries.

1.  **Create and Activate a Virtual Environment:**
    From the project's root directory, run:
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    .\venv\Scripts\activate
    ```
    *(For macOS/Linux, use `source venv/bin/activate`)*

2.  **Install Dependencies in the Virtual Environment:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

4.  **Run the Build Command:**
    From the project's root directory (the parent directory of `ifi`), run the following command. This command also bundles the `config.ini` file.

    *   **Windows (cmd or PowerShell):**
        ```bash
        pyinstaller --onefile --windowed --icon=ifi/images/IFI_icon01.ico --add-data "ifi/config.ini;ifi" run.py --name IFI_Automator
        ```
    *   **Linux / macOS:**
        ```bash
        pyinstaller --onefile --windowed --icon=ifi/images/IFI_icon01.ico --add-data "ifi/config.ini:ifi" run.py --name IFI_Automator
        ```

    The executable will be created in the `dist` folder. 
