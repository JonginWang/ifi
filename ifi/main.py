#!/usr/bin/env python3
"""
    Main entry point for the IFI application.
    ==============================

    This module is the entry point for the IFI application.
    It is used to run the application.
"""

import tkinter as tk

from ifi.utils.common import LogManager
from ifi.gui.main_window import Application

# Initialize logging
LogManager(level="INFO")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == '__main__':
    main() 