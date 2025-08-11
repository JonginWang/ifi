import tkinter as tk
import sys
from pathlib import Path

# Path-based imports for IDE compatibility
from ifi.gui.main_window import Application

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == '__main__':
    main() 