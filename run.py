"""
    This Module is the entry point for PyInstaller.
    =============================================

    It is used to run the main function from the ifi package's main module.
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
except Exception as e:
    print(f"!! Could not find ifi package root: {e}")
    pass

from ifi import main

if __name__ == '__main__':
    # Run the main function from the ifi package's main module.
    main.main() 