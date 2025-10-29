#!/usr/bin/env python3
"""
    ifi package initialization.
    ==============================
    
    This file initializes the 'ifi' package.
    It makes the 'ifi' directory a Python package.

    Functions:
        - get_project_root: Get the project root directory.
        - add_project_root: Add the project root to the Python path.

    Variables:
        - IFI_ROOT: The project root directory.
        - __all__: List of public functions and variables.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

def get_project_root(package_name: str = 'ifi',
                   markers: tuple[str, ...] = ('.git', 'pyproject.toml', 'setup.cfg')) -> Path | None:
    # Priority 1: Environment variable
    envname = os.getenv(f'{package_name.upper()}_ROOT')
    if envname:
        p = Path(envname).expanduser().resolve()
        if p.exists(): 
            return p

    # Priority 2: Searching from package directory
    current_path = Path(__file__).resolve()
    current_dir = current_path if current_path.is_dir() else current_path.parent
    dir_chain = [current_dir, *current_dir.parents]
    root_dirs = [p for p in dir_chain if p.name.lower() == package_name.lower()]

    if root_dirs:
        return root_dirs[-1]   # The outermost ifi

    # BELOW IS THE OLD CODE FOR FINDING THE IFIPACKAGE ROOT, KEEPING IT FOR REFERENCE
    # ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
    #                 + list(current_dir.parents) if p.name == 'ifi']
    # IFI_ROOT = ifi_parents[-1] if ifi_parents else None


    # Priority 3: Using markers
    for p in dir_chain:
        if any((p / m).exists() for m in markers):
            return p

    return None

def add_project_root(package_name: str = 'ifi', markers: tuple[str, ...] = ('.git', 'pyproject.toml', 'setup.cfg')):
    """
    Add the project root to the Python path.
    """
    sys.path.append(str(get_project_root(package_name, markers)))

# Calculate the projectroot once at module import time
IFI_ROOT: Path = get_project_root('ifi') or Path(__file__).expanduser().resolve().parent

# Add the project root to the Python path
add_project_root('ifi')

__all__ = ['IFI_ROOT', 'get_project_root', 'add_project_root']