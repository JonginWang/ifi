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
    pip install -r ifi/requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python -m ifi.main
    ```

## Project Structure

```
ifi/
├── __init__.py
├── main.py
├── gui/
│   ├── __init__.py
│   └── main_window.py
├── tek_controller/
│   ├── __init__.py
│   └── scope.py
├── requirements.txt
└── README.md
```

## Building an Executable

You can package this application into a single executable file using PyInstaller.

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```

2.  **Run the Build Command:**
    From the project's root directory (the parent directory of `ifi`), run the following command. For best results on Windows, convert `IFI_icon01.png` to `IFI_icon01.ico` first.

    ```bash
    pyinstaller --onefile --windowed --icon=ifi/image/IFI_icon01.png ifi/main.py --name IFI_Automator
    ```
    The executable will be created in the `dist` folder. 