#!/usr/bin/env python3
"""
Cache Setup Utilities
=====================

This module is used to set up the cache for the IFI package.

Args:
    _cache_initialized(bool): Flag to track if cache has been initialized.
    _cache_config(Dict[str, Any]): Dictionary containing the cache configuration.
    _project_root(Path): Path to the project root.

Functions:
    setup_project_cache: Set up the cache for the IFI package.
    get_cache_config: Get the current cache configuration without setting up.
    is_cache_initialized: Check if cache has been initialized.
    force_disable_jit: Force disable JIT compilation as a last resort for permission issues.
    enable_torch: Enable PyTorch module, handling dummy torch modules.
"""

import os
import sys
import importlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Global flag to track if cache has been initialized
_cache_initialized = False
_cache_config = None
_project_root: Optional[Path] = None


def _get_project_root() -> Optional[Path]:
    """Lazy import of get_project_root to avoid circular imports."""
    from ifi import get_project_root

    return get_project_root()


def setup_project_cache() -> Dict[str, Any]:
    """
    Set up project cache for numba and other components.

    This function uses lazy initialization - it only sets up the cache once
    per Python process, even if called multiple times.

    Args:
        None

    Returns:
        Dict containing cache configuration

    Raises:
        ValueError: If an error occurs while setting up the project cache.
        PermissionError: If the cache directory is not writable.
        OSError: If the cache directory cannot be created.
        Exception: If an error occurs while setting up the project cache.
    """
    global _cache_initialized, _cache_config

    # If already initialized, return existing config
    if _cache_initialized:
        return _cache_config

    # Get project root lazily to avoid circular imports
    global _project_root
    if _project_root is None:
        _project_root = _get_project_root() or Path(__file__).parent.parent.parent

    # Try multiple cache directory options in order of preference
    cache_options = [
        _project_root / "cache" / "numba_cache",  # Project cache
        Path.home() / ".ifi_cache" / "numba_cache",  # User home cache
        Path(tempfile.gettempdir()) / "ifi_numba_cache",  # System temp cache
    ]

    cache_dir = None
    for option in cache_options:
        try:
            option.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = option / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            cache_dir = option
            print(f"Using cache directory: {cache_dir}")
            break
        except (PermissionError, OSError) as e:
            print(f"Failed to use cache directory {option}: {e}")
            continue

    if cache_dir is None:
        # Last resort: disable JIT compilation
        print(
            "Warning: Could not create any cache directory. Disabling JIT compilation."
        )
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        cache_dir = Path(tempfile.gettempdir()) / "ifi_disabled_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Set up Numba environment variables
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        
        # Configure threading layer - respect user's pre-set value if exists
        # Check if user has already set NUMBA_THREADING_LAYER
        user_threading_layer = os.environ.get("NUMBA_THREADING_LAYER")
        actual_threading_layer = "safe"  # Default fallback
        
        try:
            import numba
            
            def _verify_threading_layer(layer_name: str) -> bool:
                """
                Verify if a threading layer is actually available by compiling test code.
                
                This function attempts to actually compile and run a simple parallel
                function to verify that the threading layer can be loaded at runtime.
                """
                try:
                    # Set environment variable first
                    os.environ['NUMBA_THREADING_LAYER'] = layer_name
                    # Try to set config
                    numba.config.THREADING_LAYER = layer_name
                    
                    # For 'safe', assume it works (always available)
                    if layer_name == "safe":
                        return True
                    
                    # For tbb/omp, try to actually compile and run parallel code
                    # This is the only way to verify TBB DLL can be loaded
                    try:
                        from numba import njit, prange
                        import numpy as np
                        
                        # Create a simple parallel function to test threading layer
                        @njit(parallel=True)
                        def _test_threading_layer(arr):
                            """Test function to verify threading layer works."""
                            for i in prange(len(arr)):
                                arr[i] = arr[i] * 2.0
                            return arr
                        
                        # Try to compile and run (this will attempt to load TBB/OpenMP)
                        test_arr = np.ones(100, dtype=np.float64)
                        _ = _test_threading_layer(test_arr)  # Compile and run to verify threading layer
                        
                        # If we get here without exception, threading layer loaded successfully
                        return True
                    except (ValueError, RuntimeError) as e:
                        # Check if error is specifically about threading layer
                        error_msg = str(e).lower()
                        if "no threading layer" in error_msg or "threading layer" in error_msg:
                            # Threading layer failed to load
                            return False
                        # Other errors (e.g., compilation errors) - assume it might work
                        # Actual verification will happen when user code compiles
                        return True
                    except Exception:
                        # Unexpected errors - assume it might work
                        return True
                except (AttributeError, ValueError, RuntimeError, ImportError):
                    return False
                return False
            
            if user_threading_layer is None:
                # User hasn't set it, try to find best available option
                # Priority: tbb > omp > safe
                threading_layer = None
                for preferred in ["tbb", "omp", "safe"]:
                    if _verify_threading_layer(preferred):
                        threading_layer = preferred
                        actual_threading_layer = preferred
                        break
                
                if threading_layer is None:
                    # All attempts failed, use safe
                    threading_layer = "safe"
                    os.environ['NUMBA_THREADING_LAYER'] = "safe"
                    actual_threading_layer = "safe"
            else:
                # User has explicitly set it, respect their choice but verify it works
                threading_layer = user_threading_layer
                if _verify_threading_layer(threading_layer):
                    actual_threading_layer = threading_layer
                else:
                    print(f"Warning: User requested '{threading_layer}' but it's not available")
                    # Try fallback options
                    for fallback in ["omp", "safe"]:
                        if _verify_threading_layer(fallback):
                            actual_threading_layer = fallback
                            os.environ["NUMBA_THREADING_LAYER"] = fallback
                            print(f"  Using fallback threading layer: {fallback}")
                            break
                    else:
                        actual_threading_layer = "safe"
                        os.environ["NUMBA_THREADING_LAYER"] = "safe"
            
            print(f"Numba threading layer: {actual_threading_layer}")
        except ImportError:
            print("Warning: Numba not available, using default threading layer: safe")
            actual_threading_layer = "safe"
            os.environ["NUMBA_THREADING_LAYER"] = "safe"
        except Exception as e:
            print(f"Warning: Could not configure Numba threading layer: {e}")
            print("Falling back to default threading layer: safe")
            actual_threading_layer = "safe"
            os.environ["NUMBA_THREADING_LAYER"] = "safe"

        # Prevent torch import to avoid DLL initialization errors on Windows
        # This prevents ssqueezepy from attempting to import torch.fft
        # by monkey-patching sys.modules before ssqueezepy import
        if "torch" not in sys.modules:
            # Create a dummy torch module to prevent actual import
            class DummyTorchModule:
                """Dummy torch module to prevent DLL initialization errors."""

                pass

            sys.modules["torch"] = DummyTorchModule()
            sys.modules["torch.fft"] = DummyTorchModule()

        # Additional safety settings for Windows
        if os.name == "nt":  # Windows
            os.environ["NUMBA_DISABLE_JIT"] = (
                "0"  # Keep JIT enabled but with safe settings
            )
            os.environ["NUMBA_DEBUG"] = (
                "0"  # Disable debug mode to reduce file operations
            )

    # Store configuration
    _cache_config = {
        "cache_dir": cache_dir,
        "numba_cache_dir": str(cache_dir),
        "threading_layer": os.environ.get("NUMBA_THREADING_LAYER", "safe"),
        "disable_intel_svml": "1",
        "jit_disabled": os.environ.get("NUMBA_DISABLE_JIT", "0") == "1",
    }

    # Mark as initialized
    _cache_initialized = True

    print("Project cache configured successfully.")
    return _cache_config


def get_cache_config() -> Dict[str, Any]:
    """
    Get the current cache configuration without setting up.

    Returns:
        Dict containing cache configuration, or None if not initialized

    Raises:
        Exception: If an error occurs while getting the cache configuration.
    """
    return _cache_config if _cache_initialized else None


def is_cache_initialized() -> bool:
    """
    Check if cache has been initialized.

    Returns:
        True if cache is initialized, False otherwise

    Raises:
        Exception: If an error occurs while checking if the cache is initialized.
    """
    return _cache_initialized


def force_disable_jit():
    """
    Force disable JIT compilation as a last resort for permission issues.

    Raises:
        Exception: If an error occurs while forcing disable JIT compilation.
    """
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    print("JIT compilation disabled due to permission issues.")


def enable_torch():
    """Enable PyTorch module, handling dummy torch modules."""
    
    # Check if torch is already loaded and is real (not dummy)
    if "torch" in sys.modules:
        existing_torch = sys.modules["torch"]
        # Check if it's a real torch module (has __version__ attribute)
        if hasattr(existing_torch, "__version__"):
            try:
                # Try to access version to confirm it's real
                _ = existing_torch.__version__
                print(f"Real torch already loaded: version {existing_torch.__version__}")
                return existing_torch
            except (AttributeError, RuntimeError):
                pass  # Fall through to reload logic
    
    # Torch is either not loaded or is a dummy - remove it
    print("Removing dummy/stub torch modules...")
    modules_to_remove = []
    for name in list(sys.modules.keys()):
        if name == "torch" or name.startswith("torch."):
            modules_to_remove.append(name)
    
    for name in modules_to_remove:
        sys.modules.pop(name, None)
    
    # Invalidate import caches
    importlib.invalidate_caches()
    
    # Try to import real torch
    try:
        torch = importlib.import_module("torch")
        if hasattr(torch, "__version__"):
            print(f"Successfully loaded real torch: version {torch.__version__}")
            return torch
        else:
            raise ImportError("Loaded torch module does not have __version__ attribute")
    except (ImportError, RuntimeError, AttributeError) as e:
        print(f"Error loading torch: {e}")
        print("Torch may already be partially loaded. Try restarting the kernel.")
        raise
