import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from enum import Enum, auto
import queue
import os
import threading
import time

# 상대 경로 임포트를 위해 경로 설정이 필요할 수 있습니다.
from ..tek_controller.scope import TektronixScope
from ..db_controller.vest_db import VEST_DB

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from pynput import keyboard

class AppState(Enum):
    IDLE = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()

class Application(tk.Frame):
    def __init__(self, master=None):
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
        self.scope1 = None
        self.scope2 = None
        self.vest_db = VEST_DB()

        self.create_widgets()
        
        # --- Worker Thread ---
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        
        # --- Keyboard Listener ---
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.process_gui_queue()

    def create_widgets(self):
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
        parent.pack_propagate(False)
        # Control Section
        control_frame = ttk.LabelFrame(parent, text="Control")
        control_frame.pack(padx=10, pady=10, fill="x")
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_process)
        self.start_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.pack(side="left", padx=5, pady=5)

        # VEST DB Section
        db_frame = ttk.LabelFrame(parent, text="VEST DB")
        db_frame.pack(padx=10, pady=10, fill="x")
        self.db_conn_button = ttk.Button(db_frame, text="Conn. VEST DB", command=self.connect_db_action)
        self.db_conn_button.pack(side="left", padx=5, pady=5)
        self.db_status_light = tk.Label(db_frame, text="●", fg="gray", font=("Helvetica", 16))
        self.db_status_light.pack(side="left", padx=5, pady=5)
        ttk.Label(db_frame, text="Curr. Shot. Num:").pack(side="left", padx=5, pady=5)
        self.shot_num_var = tk.StringVar(value="N/A")
        ttk.Entry(db_frame, textvariable=self.shot_num_var, state="readonly").pack(side="left", padx=5, pady=5, expand=True, fill="x")

        # Save Section
        save_frame = ttk.LabelFrame(parent, text="Save")
        save_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(save_frame, text="Save Opt.:").pack(side="left", padx=5, pady=5)
        self.save_opt_var = tk.StringVar()
        save_options = ["MySQL", "Trig", "Manual"]
        self.save_opt_dropdown = ttk.Combobox(save_frame, textvariable=self.save_opt_var, values=save_options, state="readonly")
        self.save_opt_dropdown.pack(side="left", padx=5, pady=5)
        self.save_opt_dropdown.set("Manual") # Default value
        self.save_status_light = tk.Label(save_frame, text="●", fg="gray", font=("Helvetica", 16))
        self.save_status_light.pack(side="left", padx=5, pady=5)
        ttk.Label(save_frame, text="Save Name:").pack(side="left", padx=5, pady=5)
        self.manual_save_name = tk.StringVar(value="manual_save")
        ttk.Entry(save_frame, textvariable=self.manual_save_name).pack(side="left", padx=5, pady=5, expand=True, fill="x")
        self.manual_save_button = ttk.Button(save_frame, text="Save", command=self.manual_save_action)
        self.manual_save_button.pack(side="left", padx=5, pady=5)

        # Log Section
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_area.pack(padx=5, pady=5, fill="both", expand=True)
        self.log_area.configure(state='disabled')
        save_log_button = ttk.Button(log_frame, text="Save Log", command=self.save_log)
        save_log_button.pack(padx=5, pady=5, anchor="e")

    def create_right_pane_widgets(self, parent):
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
        # DAQ Option Section
        daq_frame = ttk.LabelFrame(parent_tab, text="DAQ Option")
        daq_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Label(daq_frame, text="Scope 1:").pack(side="left", padx=5, pady=5)
        self.scope1_var = tk.StringVar()
        self.scope1_dropdown = ttk.Combobox(daq_frame, textvariable=self.scope1_var, state="readonly")
        self.scope1_dropdown.pack(side="left", padx=5, pady=5)
        
        if num_scopes == 2:
            ttk.Label(daq_frame, text="Scope 2:").pack(side="left", padx=5, pady=5)
            self.scope2_var = tk.StringVar()
            self.scope2_dropdown = ttk.Combobox(daq_frame, textvariable=self.scope2_var, state="readonly")
            self.scope2_dropdown.pack(side="left", padx=5, pady=5)

        # Waveform Section
        waveform_frame = ttk.LabelFrame(parent_tab, text="Waveform")
        waveform_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Matplotlib Graph
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.grid()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        
        canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, waveform_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def start_process(self):
        # Will be implemented
        pass

    def stop_process(self):
        # Will be implemented
        pass

    def save_log(self):
        log_content = self.log_area.get("1.0", tk.END)
        if not log_content.strip():
            self.log_message("Log is empty, nothing to save.", "WARN")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Log File"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            self.log_message(f"Log saved to {file_path}")

    def on_key_press(self, key):
        """ Pynput keyboard listener callback. """
        # 예시: F12 키를 누르면 수동 저장 트리거
        if key == keyboard.Key.f12:
            print("F12 pressed, triggering manual save.")
            # GUI에서 직접 처리하는 대신 워커 스레드에 작업을 요청합니다.
            self.task_queue.put(('manual_save', {}))
        
    def worker_loop(self):
        """ This loop runs in a background thread to handle long-running tasks. """
        while True:
            try:
                task_name, kwargs = self.task_queue.get()
                
                if task_name == 'connect_db':
                    is_connected = self.vest_db.connect()
                    self.gui_queue.put(('db_status_update', {'connected': is_connected}))
                
                elif task_name == 'get_shot_num':
                    shot_num = self.vest_db.get_next_shot_code()
                    if shot_num:
                        self.gui_queue.put(('shot_num_update', {'shot_num': shot_num}))

                elif task_name == 'manual_save':
                    # GUI에서 직접 가져오면 스레드 충돌이 날 수 있으므로, kwargs로 전달받아야 함
                    save_name = kwargs.get('save_name')
                    # scope 객체도 안전하게 가져와야 함
                    # self.scope1.save_all_channels_to_file(save_name, 'path')
                    print(f"Worker: Pretending to save with name {save_name}")
                    self.gui_queue.put(('log', {'level': 'INFO', 'msg': f"Manual save '{save_name}' triggered."}))

                # 여기에 다른 task들 (e.g., 'start_monitoring', 'stop_monitoring')을 추가
                
                self.task_queue.task_done()
            except Exception as e:
                # 에러를 GUI 큐로 보내 화면에 표시
                self.gui_queue.put(('log', {'level': 'ERROR', 'msg': f"Worker error: {e}"}))

    def process_gui_queue(self):
        """ Process items from the gui_queue to update the GUI safely. """
        try:
            while True:
                task_name, kwargs = self.gui_queue.get_nowait()
                
                if task_name == 'db_status_update':
                    is_connected = kwargs.get('connected')
                    color = "green" if is_connected else "red"
                    self.db_status_light.config(fg=color)
                    msg = "DB Connected." if is_connected else "DB Connection Failed."
                    self.log_message(msg, "INFO" if is_connected else "ERROR")
                    
                elif task_name == 'shot_num_update':
                    self.shot_num_var.set(kwargs.get('shot_num', 'Error'))
                    
                elif task_name == 'log':
                    self.log_message(kwargs.get('msg'), kwargs.get('level'))
                
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_gui_queue)

    def connect_db_action(self):
        """ Action for the 'Conn. VEST DB' button. """
        self.task_queue.put(('connect_db', {}))
        # 버튼을 누르면 바로 DB shot number 가져오기 시도
        self.task_queue.put(('get_shot_num', {}))
    
    def manual_save_action(self):
        """ Action for the manual 'Save' button. """
        save_name = self.manual_save_name.get()
        self.task_queue.put(('manual_save', {'save_name': save_name}))
        
    def __del__(self):
        """ Ensure listener thread is stopped when the app closes. """
        self.keyboard_listener.stop()

    def log_message(self, msg, level="INFO"):
        """ Directly updates the log area. To be called from the main GUI thread only. """
        log_entry = f"[{level}] {time.strftime('%H:%M:%S')} {msg}\n"
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, log_entry)
        self.log_area.configure(state='disabled')
        self.log_area.see(tk.END) 