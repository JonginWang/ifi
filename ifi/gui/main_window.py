#!/usr/bin/env python3
"""
Main Window
===========

This module contains the GUI for the IFI application.

Classes:
    AppState: The enum class for the application state.
    Application: The main application class for the IFI application.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from enum import Enum, auto
import queue
import threading
import time
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pynput import keyboard
try:
    from .. import get_project_root
    from ..tek_controller.scope import TekScopeController
    from ..db_controller.vest_db import VEST_DB
    from ..gui.suffix_config import (
        MachineSuffixConfig,
        build_data_dict_from_channels,
        load_suffix_config,
    )
    from ..gui.workers.vest_db_worker import VestDbPollingWorker
    from ..utils.file_io import read_waveform_file, save_waveform_to_csv
    from ..analysis.params.params_plot import FontStyle
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import get_project_root
    from ifi.tek_controller.scope import TekScopeController
    from ifi.db_controller.vest_db import VEST_DB
    from ifi.gui.suffix_config import (
        MachineSuffixConfig,
        build_data_dict_from_channels,
        load_suffix_config,
    )
    from ifi.gui.workers.vest_db_worker import VestDbPollingWorker
    from ifi.utils.file_io import read_waveform_file, save_waveform_to_csv
    from ifi.analysis.params.params_plot import FontStyle


class AppState(Enum):
    """
    The enum class for the application state.

    Attributes:
        IDLE(auto()): The idle state.
        RUNNING(auto()): The running state.
        STOPPING(auto()): The stopping state.
        ERROR(auto()): The error state.
    """

    IDLE = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


class Application(tk.Frame):
    """
    The main application class for the IFI application.

    Args:
        master(tk.Tk): The master window.

    Attributes:
        master(tk.Tk): The master window.
        state(AppState): The current state of the application.
        task_queue(queue.Queue): The queue for tasks.
        gui_queue(queue.Queue): The queue for GUI updates.
        scope_controller(TekScopeController): The controller for the scope.
        selected_scope1_id(tk.StringVar): The selected scope 1 identifier.
        selected_scope2_id(tk.StringVar): The selected scope 2 identifier.
        vest_db(VEST_DB): The database for the VEST data.
        log_area(scrolledtext.ScrolledText): The area for the log.
        start_button(ttk.Button): The button to start the process.
        stop_button(ttk.Button): The button to stop the process.
        db_conn_button(ttk.Button): The button to connect to the VEST database.
        db_status_light(tk.Label): The light to indicate the status of the VEST database connection.
        shot_num_var(tk.StringVar): The variable for the current shot number.
        save_opt_var(tk.StringVar): The variable for the save option.
        save_opt_dropdown(ttk.Combobox): The dropdown for the save option.
        save_status_light(tk.Label): The light to indicate the status of the save.
        manual_save_name(tk.StringVar): The variable for the manual save name.
        manual_save_button(ttk.Button): The button to manually save the data.
        log_area(scrolledtext.ScrolledText): The area for the log.
        start_button(ttk.Button): The button to start the process.
        stop_button(ttk.Button): The button to stop the process.

    Methods:
        create_widgets(): Create the widgets for the application.
        create_left_pane_widgets(): Create the widgets for the left pane.
        create_right_pane_widgets(): Create the widgets for the right pane.
        create_tab_content(): Create the content for the tabs.
        start_process(): Start the process.
        stop_process(): Stop the process.
        save_log(): Save the log to a file.
        on_key_press(): The callback for the key press event.
        worker_loop(): The loop for the worker.
        process_gui_queue(): The queue for the GUI updates.
        connect_db_action(): The action to connect to the VEST database.
        manual_save_action(): The action to manually save the data.
        __del__(): The destructor for the application.
        log_message(): The message to the log area.
        plot_data(): The data to the plot.
        load_waveform_action(): The action to load the waveform from a file.
        connect_scope1_action(): The action to connect to the scope 1.
        connect_scope2_action(): The action to connect to the scope 2.
        connect_scope3_action(): The action to connect to the scope 3.
        connect_scope4_action(): The action to connect to the scope 4.
        connect_scope5_action(): The action to connect to the scope 5.
        connect_scope6_action(): The action to connect to the scope 6.

    Examples:
        ```python
        from .main_window import Application

        app = Application()
        app.mainloop()
        ```
    """

    def __init__(self, master=None):
        """
        Initialize the application.

        Args:
            master(tk.Tk): The master window.
        """
        super().__init__(master)
        self.master = master
        self.master.title("Tektronix Data Automator")
        self.master.geometry("1000x600")
        self.pack(fill="both", expand=True)

        self.state = AppState.IDLE

        # --- Thread-safe Queues ---
        self.task_queue = queue.Queue()
        self.gui_queue = queue.Queue()

        # --- Backend Objects ---
        # Now we have controller objects that manage the devices
        self.scope_controller = TekScopeController()
        # These will hold the identifier string ("MODEL - SERIAL")
        self.selected_scope1_id = tk.StringVar()
        self.selected_scope2_id = tk.StringVar()

        config_file = Path(get_project_root()) / "ifi" / "config.ini"
        self.vest_db = VEST_DB(config_path=config_file)

        # --- Suffix / Channel Configuration ---
        suffix_ini = Path(get_project_root()) / "ifi" / "gui" / "suffix.ini"
        # For now, assume the first tab ("IF 94G") corresponds to machine key "94G".
        self.suffix_config: MachineSuffixConfig | None = load_suffix_config(
            suffix_ini, machine_key="94G"
        )

        self.vest_db_poller: VestDbPollingWorker | None = None

        self.create_widgets()

        # --- Worker Thread ---
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # --- Initial Tasks ---
        self.task_queue.put(("find_scopes", {}))

        # --- Keyboard Listener ---
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.process_gui_queue()

    def create_widgets(self):
        """Create the widgets for the application."""
        # --- Main Layout ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane, width=350)
        right_frame = ttk.Frame(main_pane, width=650)
        main_pane.add(left_frame, weight=1)
        main_pane.add(right_frame, weight=3)

        # --- Left Pane ---
        self.create_left_pane_widgets(left_frame)

        # --- Right Pane ---
        self.create_right_pane_widgets(right_frame)

    def create_left_pane_widgets(self, parent):
        """
        Create the widgets for the left pane.

        Args:
            parent(ttk.Frame): The parent frame.
        """
        parent.pack_propagate(False)
        # Control Section
        control_frame = ttk.LabelFrame(parent, text="Control")
        control_frame.pack(padx=10, pady=10, fill="x")
        self.start_button = ttk.Button(
            control_frame, text="Start", command=self.start_process
        )
        self.start_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ttk.Button(
            control_frame, text="Stop", command=self.stop_process, state="disabled"
        )
        self.stop_button.pack(side="left", padx=5, pady=5)

        # VEST DB Section
        db_frame = ttk.LabelFrame(parent, text="VEST DB")
        db_frame.pack(padx=10, pady=10, fill="x")
        self.db_conn_button = ttk.Button(
            db_frame, text="Conn. VEST DB", command=self.connect_db_action
        )
        self.db_conn_button.pack(side="left", padx=5, pady=5)
        self.db_status_light = tk.Label(
            db_frame, text="●", fg="gray", font=("Helvetica", 16)
        )
        self.db_status_light.pack(side="left", padx=5, pady=5)
        ttk.Label(db_frame, text="Curr. Shot. Num:").pack(side="left", padx=5, pady=5)
        self.shot_num_var = tk.StringVar(value="N/A")
        ttk.Entry(db_frame, textvariable=self.shot_num_var, state="readonly").pack(
            side="left", padx=5, pady=5, expand=True, fill="x"
        )

        # Save Section
        save_frame = ttk.LabelFrame(parent, text="Save")
        save_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(save_frame, text="Save Opt.:").pack(side="left", padx=5, pady=5)
        self.save_opt_var = tk.StringVar()
        save_options = ["MySQL", "Trig", "Manual"]
        self.save_opt_dropdown = ttk.Combobox(
            save_frame,
            textvariable=self.save_opt_var,
            values=save_options,
            state="readonly",
        )
        self.save_opt_dropdown.pack(side="left", padx=5, pady=5)
        self.save_opt_dropdown.set("Manual")  # Default value
        self.save_status_light = tk.Label(
            save_frame, text="●", fg="gray", font=("Helvetica", 16)
        )
        self.save_status_light.pack(side="left", padx=5, pady=5)
        ttk.Label(save_frame, text="Save Name:").pack(side="left", padx=5, pady=5)
        self.manual_save_name = tk.StringVar(value="manual_save")
        ttk.Entry(save_frame, textvariable=self.manual_save_name).pack(
            side="left", padx=5, pady=5, expand=True, fill="x"
        )
        self.manual_save_button = ttk.Button(
            save_frame, text="Save", command=self.manual_save_action
        )
        self.manual_save_button.pack(side="left", padx=5, pady=5)

        # Log Section
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_area.pack(padx=5, pady=5, fill="both", expand=True)
        self.log_area.configure(state="disabled")
        save_log_button = ttk.Button(log_frame, text="Save Log", command=self.save_log)
        save_log_button.pack(padx=5, pady=5, anchor="e")

    def create_right_pane_widgets(self, parent):
        """
        Create the widgets for the right pane.

        Args:
            parent(ttk.Frame): The parent frame.
        """
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)

        notebook.add(tab1, text="IF 94G")
        notebook.add(tab2, text="IF 280G")
        notebook.add(tab3, text="Meas.")

        self.create_tab_content(tab1, num_scopes=2)
        self.create_tab_content(tab2, num_scopes=1)
        self.create_tab_content(tab3, num_scopes=1)

    def create_tab_content(self, parent_tab, num_scopes):
        """
        Create the content for the tabs.

        Args:
            parent_tab(ttk.Frame): The parent tab.
            num_scopes(int): The number of scopes.
        """
        # DAQ Option Section
        daq_frame = ttk.LabelFrame(parent_tab, text="DAQ Option")
        daq_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(daq_frame, text="Scope 1:").pack(side="left", padx=5, pady=5)
        self.scope1_dropdown = ttk.Combobox(
            daq_frame, textvariable=self.selected_scope1_id, state="readonly"
        )
        self.scope1_dropdown.pack(side="left", padx=5, pady=5)
        self.scope1_dropdown.bind("<<ComboboxSelected>>", self.connect_scope1_action)

        if num_scopes == 2:
            ttk.Label(daq_frame, text="Scope 2:").pack(side="left", padx=5, pady=5)
            self.scope2_dropdown = ttk.Combobox(
                daq_frame, textvariable=self.selected_scope2_id, state="readonly"
            )
            self.scope2_dropdown.pack(side="left", padx=5, pady=5)
            # self.scope2_dropdown.bind("<<ComboboxSelected>>", self.connect_scope2_action)

        # Waveform Section
        waveform_frame = ttk.LabelFrame(parent_tab, text="Waveform")
        waveform_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # --- Top controls for waveform ---
        wf_control_frame = ttk.Frame(waveform_frame)
        wf_control_frame.pack(fill="x", padx=5, pady=5)

        load_file_button = ttk.Button(
            wf_control_frame, text="Load from File", command=self.load_waveform_action
        )
        load_file_button.pack(side="left")

        # Matplotlib Graph
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.grid()
        self.ax.set_xlabel("Time [s]", **FontStyle.label)
        self.ax.set_ylabel("Voltage [V]", **FontStyle.label)

        canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, waveform_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_process(self):
        """Start the process."""
        # Will be implemented
        pass

    def stop_process(self):
        """Stop the process."""
        # Will be implemented
        pass

    def save_log(self):
        """Save the current log to a file."""
        try:
            save_dir = filedialog.askdirectory(title="Select directory to save log")
            if save_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"tek_automator_log_{timestamp}.txt"
                filepath = Path(save_dir) / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.log_area.get("1.0", tk.END))

                self.log_message(f"Log saved to {filepath}")
        except Exception as e:
            self.log_message(f"Failed to save log: {e}", "ERROR")

    def on_key_press(self, key):
        """
        Pynput keyboard listener callback.

        Args:
            key(pynput.keyboard.Key): The key that was pressed.
        """
        # TODO: For example, if the F12 key is pressed, trigger the manual save.
        if key == keyboard.Key.f12:
            print("F12 pressed, triggering manual save.")
            # Instead of directly processing in the GUI, request the task to the worker thread.
            # Get the save name from the GUI thread and pass it to the worker
            save_name = self.manual_save_name.get()
            self.task_queue.put(("manual_save", {"save_name": save_name}))

    def worker_loop(self):
        """This loop runs in a background thread to handle long-running tasks."""
        while True:
            try:
                task_name, kwargs = self.task_queue.get()

                if task_name == "find_scopes":
                    scopes_found = self.scope_controller.list_devices()
                    self.gui_queue.put(("update_scope_list", {"scopes": scopes_found}))

                elif task_name == "connect_scope":
                    identifier = kwargs.get("identifier")
                    is_connected = self.scope_controller.connect(identifier)
                    # We can add more GUI feedback here
                    self.gui_queue.put(
                        (
                            "log",
                            {
                                "level": "INFO" if is_connected else "ERROR",
                                "msg": f"Connection to {identifier}: {'OK' if is_connected else 'Failed'}",
                            },
                        )
                    )

                elif task_name == "connect_db":
                    is_connected = self.vest_db.connect()
                    self.gui_queue.put(
                        ("db_status_update", {"connected": is_connected})
                    )

                elif task_name == "get_shot_num":
                    shot_num = self.vest_db.get_next_shot_code()
                    if shot_num:
                        self.gui_queue.put(("shot_num_update", {"shot_num": shot_num}))

                elif task_name == "manual_save":
                    save_name = kwargs.get("save_name")
                    if not save_name:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "WARN",
                                    "msg": "Manual save failed: Save name cannot be empty.",
                                },
                            )
                        )
                        continue

                    if not self.scope_controller.is_idle():
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "WARN",
                                    "msg": "Manual save request ignored: scope controller is busy.",
                                },
                            )
                        )
                        continue

                    self.gui_queue.put(
                        (
                            "log",
                            {
                                "level": "INFO",
                                "msg": f"Starting manual save for '{save_name}'...",
                            },
                        )
                    )

                    # For simplicity, we save CH1. This could be configurable in the GUI.
                    channel_to_save = "CH1"
                    waveform_data = self.scope_controller.get_waveform_data(
                        channel=channel_to_save
                    )

                    if waveform_data:
                        time_data, voltage_data = waveform_data
                        # Define a save directory
                        save_dir = "data"  # Or get this from a config file/GUI
                        filename = f"{save_name}_{channel_to_save}.csv"
                        filepath = Path(save_dir) / filename

                        # Use the new file_io function
                        save_waveform_to_csv(
                            filepath,
                            time_data,
                            voltage_data,
                            channel_name=channel_to_save,
                        )

                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "INFO",
                                    "msg": f"Successfully saved waveform to {filepath}",
                                },
                            )
                        )
                        # Also plot the newly saved data
                        self.gui_queue.put(
                            (
                                "plot_update",
                                {"time": time_data, "voltage": voltage_data},
                            )
                        )
                    else:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "ERROR",
                                    "msg": "Failed to get waveform data for manual save.",
                                },
                            )
                        )

                elif task_name == "auto_trigger":
                    shot_code = kwargs.get("shot_code")
                    if not isinstance(shot_code, int):
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "WARN",
                                    "msg": "Auto trigger ignored: invalid shot code.",
                                },
                            )
                        )
                        continue

                    if not self.scope_controller.is_idle():
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "WARN",
                                    "msg": "Auto trigger ignored: scope controller is busy.",
                                },
                            )
                        )
                        continue

                    self.gui_queue.put(
                        (
                            "log",
                            {
                                "level": "INFO",
                                "msg": f"Auto trigger for shot {shot_code}: starting acquisition and save.",
                            },
                        )
                    )

                    # Determine channels based on suffix configuration. If no
                    # configuration is available, fall back to a single CH1
                    # acquisition.
                    if self.suffix_config is not None:
                        configured_channels = [
                            name
                            for name in self.suffix_config.profile.channels
                            if name.upper() != "TIME"
                        ]
                        channels = configured_channels or ["CH1"]
                        suffix = self.suffix_config.profile.suffix
                    else:
                        channels = ["CH1"]
                        suffix = "_AUTO"

                    acquired = self.scope_controller.acquire_data(channels)
                    if not acquired:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "ERROR",
                                    "msg": f"Auto trigger for shot {shot_code} failed: no data acquired.",
                                },
                            )
                        )
                        continue

                    # Use the first successfully acquired channel as the time
                    # reference for building the data dictionary.
                    first_channel = channels[0]
                    time_data, voltage_data = acquired[first_channel]
                    # Build data dict according to configured channel ordering.
                    save_data = build_data_dict_from_channels(
                        self.suffix_config.profile.channels
                        if self.suffix_config is not None
                        else ["TIME", first_channel],
                        time_array=time_data,
                        channel_arrays={ch: acquired[ch][1] for ch in acquired},
                    )

                    save_dir = Path("data")
                    filepath = self.scope_controller.save_data(
                        directory=save_dir,
                        shot_code=shot_code,
                        suffix=suffix,
                        data=save_data,
                    )

                    if filepath is not None:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "INFO",
                                    "msg": f"Auto trigger: saved shot {shot_code} to {filepath}",
                                },
                            )
                        )
                        self.gui_queue.put(
                            (
                                "plot_update",
                                {"time": time_data, "voltage": voltage_data},
                            )
                        )
                    else:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "ERROR",
                                    "msg": f"Auto trigger for shot {shot_code} failed: save_data returned None.",
                                },
                            )
                        )

                elif task_name == "read_and_plot_file":
                    filepath = kwargs.get("filepath")
                    self.gui_queue.put(
                        ("log", {"level": "INFO", "msg": f"Reading file: {filepath}"})
                    )
                    time_data, voltage_data = read_waveform_file(filepath)
                    if time_data is not None:
                        # Send data to GUI thread for plotting
                        self.gui_queue.put(
                            (
                                "plot_update",
                                {"time": time_data, "voltage": voltage_data},
                            )
                        )
                    else:
                        self.gui_queue.put(
                            (
                                "log",
                                {
                                    "level": "ERROR",
                                    "msg": f"Failed to read waveform from {Path(filepath).name}",
                                },
                            )
                        )

                # TODO: Add other tasks here (e.g., 'start_monitoring', 'stop_monitoring').

                self.task_queue.task_done()
            except Exception as e:
                # Send the error to the GUI queue to display on the screen.
                self.gui_queue.put(
                    ("log", {"level": "ERROR", "msg": f"Worker error: {e}"})
                )

    def process_gui_queue(self):
        """Process items from the gui_queue to update the GUI safely."""
        try:
            while True:
                task_name, kwargs = self.gui_queue.get_nowait()

                if task_name == "update_scope_list":
                    scopes = kwargs.get("scopes", [])
                    # Assuming you have dropdowns named self.scope1_dropdown etc.
                    self.scope1_dropdown["values"] = scopes
                    # self.scope2_dropdown['values'] = scopes # If you have a second one

                elif task_name == "db_status_update":
                    is_connected = kwargs.get("connected")
                    color = "green" if is_connected else "red"
                    self.db_status_light.config(fg=color)
                    msg = "DB Connected." if is_connected else "DB Connection Failed."
                    self.log_message(msg, "INFO" if is_connected else "ERROR")

                elif task_name == "shot_num_update":
                    self.shot_num_var.set(kwargs.get("shot_num", "Error"))

                elif task_name == "log":
                    self.log_message(kwargs.get("msg"), kwargs.get("level"))

                elif task_name == "plot_update":
                    t = kwargs.get("time")
                    v = kwargs.get("voltage")
                    self.plot_data(t, v)

        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_gui_queue)

    def connect_db_action(self):
        """Action for the 'Conn. VEST DB' button."""
        self.task_queue.put(("connect_db", {}))
        # Try to get the DB shot number immediately when the button is pressed.
        self.task_queue.put(("get_shot_num", {}))

        # Start VEST DB polling worker after a successful manual connection attempt.
        if self.vest_db_poller is None:
            # The callback enqueues a GUI-safe update
            def _on_new_shot_code(shot_code: int) -> None:
                self.gui_queue.put(
                    ("shot_num_update", {"shot_num": str(shot_code)})
                )

            self.vest_db_poller = VestDbPollingWorker(
                vest_db=self.vest_db,
                on_new_shot_code=_on_new_shot_code,
                poll_interval=30.0,
            )

        self.vest_db_poller.start_polling()

    def manual_save_action(self):
        """Action for the manual 'Save' button."""
        save_name = self.manual_save_name.get()
        self.task_queue.put(("manual_save", {"save_name": save_name}))

    def __del__(self):
        """Ensure listener thread is stopped when the app closes."""
        self.keyboard_listener.stop()
        if self.vest_db_poller is not None:
            self.vest_db_poller.stop_polling()

    def log_message(self, msg, level="INFO"):
        """
        Directly updates the log area. To be called from the main GUI thread only.

        Args:
            msg(str): The message to log.
            level(str): The level of the message.
        """
        log_entry = f"[{level}] {time.strftime('%H:%M:%S')} {msg}\n"
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, log_entry)
        self.log_area.configure(state="disabled")
        self.log_area.see(tk.END)

    def plot_data(self, time_data, voltage_data):
        """Helper function to plot data on the matplotlib canvas.

        Args:
            time_data(np.ndarray): The time data.
            voltage_data(np.ndarray): The voltage data.
        """
        if time_data is not None and voltage_data is not None:
            self.ax.clear()
            self.ax.plot(time_data, voltage_data)
            self.ax.grid(True)
            self.ax.set_xlabel("Time [s]", **FontStyle.label)
            self.ax.set_ylabel("Voltage [V]", **FontStyle.label)
            self.ax.figure.canvas.draw()
            self.log_message(f"Plotted waveform with {len(time_data)} points.")
        else:
            self.log_message("Failed to plot data.", "WARN")

    def load_waveform_action(self):
        """Load a waveform file and display it."""
        try:
            filepath = filedialog.askopenfilename(
                title="Select waveform file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if filepath:
                result = read_waveform_file(filepath)
                if result:
                    time_data, voltage_data = result
                    self.plot_data(time_data, voltage_data)
                    self.log_message(f"Loaded waveform from {Path(filepath).name}")
                else:
                    self.gui_queue.put(
                        (
                            "log",
                            {
                                "level": "ERROR",
                                "msg": f"Failed to read waveform from {Path(filepath).name}",
                            },
                        )
                    )
        except Exception as e:
            self.log_message(f"Error loading waveform: {e}", "ERROR")

    def connect_scope1_action(self, event):
        """
        Action when a scope is selected from the first dropdown.

        Args:
            event(tk.Event): The event that triggered the action.
        """
        selected_identifier = self.selected_scope1_id.get()
        if selected_identifier:
            self.task_queue.put(("connect_scope", {"identifier": selected_identifier}))
