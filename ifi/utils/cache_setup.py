"""
Numba Cache Setup Utilities
============================

Utilities for configuring numba cache directories to avoid permission issues
with system-wide Anaconda installations.
"""

import os
import tempfile
import logging

def setup_numba_cache(project_root=None, verbose=True):
    """
    Configure numba cache to use a user-writable directory.
    
    Args:
        project_root (str, optional): Path to project root. If None, auto-detect.
        verbose (bool): Whether to print status messages.
    
    Returns:
        str: Path to the configured cache directory.
    """
    if project_root is None:
        # Auto-detect project root (go up from utils directory)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Option 1: Use project-local cache directory
    cache_dir = os.path.join(project_root, 'cache', 'numba_cache')
    
    # Option 2: Use user's temp directory if project cache fails
    if not os.path.exists(os.path.dirname(cache_dir)):
        try:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        except PermissionError:
            # Fallback to user temp directory
            cache_dir = os.path.join(tempfile.gettempdir(), 'ifi_numba_cache')
    
    # Option 3: Use user's home directory as last resort
    if not os.access(os.path.dirname(cache_dir), os.W_OK):
        cache_dir = os.path.join(os.path.expanduser('~'), '.ifi', 'numba_cache')
    
    # Create the directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables
    os.environ['NUMBA_CACHE_DIR'] = cache_dir
    os.environ['NUMBA_ENABLE_CUDASIM'] = '0'  # Disable CUDA simulation
    os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'  # Disable Intel SVML for compatibility
    os.environ['NUMBA_THREADING_LAYER'] = 'safe'  # Use thread-safe layer
    
    if verbose:
        print(f"Numba cache directory set to: {cache_dir}")
    
    return cache_dir

def setup_project_cache():
    """
    Set up comprehensive cache configuration for the entire IFI project.
    This should be called at the very beginning of main scripts.
    
    Returns:
        dict: Configuration information including cache paths.
    """
    # Configure numba cache
    numba_cache = setup_numba_cache(verbose=False)
    
    # Additional project-specific cache settings
    config = {
        'numba_cache_dir': numba_cache,
        'cache_configured': True
    }
    
    # Log the configuration
    logging.info(f"Project cache configured: {numba_cache}")
    
    return config

# Legacy function for backward compatibility
def setup_ssqueezepy_cache():
    """
    Legacy function - use setup_project_cache() instead.
    """
    return setup_numba_cache()

# Quick setup function for import-time configuration
def quick_setup():
    """
    Quick setup function that can be imported and called immediately.
    Suppresses output for cleaner imports.
    """
    try:
        return setup_numba_cache(verbose=False)
    except Exception:
        # Silently fall back if setup fails
        return None

# Auto-setup when this module is imported (optional)
# Uncomment the line below if you want automatic setup on import
# quick_setup()