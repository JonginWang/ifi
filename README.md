# ifi
InterFerometer Instrumentation 

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
    pip install -r ifi/requirements.txt
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
