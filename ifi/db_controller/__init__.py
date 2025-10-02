"""
    DB Controller
    ============

    This module contains the DB Controller for the IFI application.
"""

import sys
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

from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB

__all__ = ['NAS_DB', 'VEST_DB']