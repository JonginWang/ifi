"""
    Main entry point for the IFI application.
    ==============================

    This module is the entry point for the IFI application.
    It is used to run the application.
"""

import tkinter as tk
import sys
from pathlib import Path


# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    pass

# Path-based imports for IDE compatibility
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