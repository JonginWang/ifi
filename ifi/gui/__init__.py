"""
    GUI
    ====

    This module contains the GUI for the IFI application.
"""

import sys
import logging
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception:
    # print(f"!! Could not find ifi package root: {e}")
    pass

from ifi.gui.main_window import Application