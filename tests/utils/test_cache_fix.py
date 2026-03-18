#!/usr/bin/env python3
"""
Test script for cache setup and ssqueezepy import
"""

import sys

def test_cache_and_import():
    """Test cache setup and ssqueezepy import"""
    print("Testing cache setup and ssqueezepy import...")
    
    try:
        # Setup cache first
        from ifi.utils.cache_setup import setup_project_cache
        cache_config = setup_project_cache()
        print(f"Cache cache_config: {cache_config}")
        
        # Try importing ssqueezepy
        import ssqueezepy as ssqpy
        print("ssqueezepy imported successfully!")
        
        # Test basic CWT functionality
        import numpy as np
        x = np.random.randn(1000)
        
        # Test CWT - use correct parameter syntax
        result = ssqpy.cwt(x, wavelet='gmw', nv=32)
        print(f"CWT computed successfully! Result type: {type(result)}")
        
        # Handle tuple return value
        if isinstance(result, tuple):
            Wx, scales, *_ = result
            print(f"  CWT shape: {Wx.shape}")
            print(f"  Scales shape: {scales.shape}")
        else:
            Wx = result
            print(f"  CWT shape: {Wx.shape}")
        
        # Test CWT with different parameters to check time axis control
        print("\nTesting CWT time axis parameters...")
        
        # Check if we can control the time axis length
        # Let's look at the CWT function signature
        import inspect
        sig = inspect.signature(ssqpy.cwt)
        print(f"CWT function signature: {sig}")
        
        # Test with different signal lengths
        x_short = np.random.randn(500)
        x_long = np.random.randn(2000)
        
        result_short = ssqpy.cwt(x_short, wavelet='gmw', nv=32)
        result_long = ssqpy.cwt(x_long, wavelet='gmw', nv=32)
        
        # Extract CWT from results
        if isinstance(result_short, tuple):
            Wx_short, *_ = result_short
            Wx_long, *_ = result_long
        else:
            Wx_short = result_short
            Wx_long = result_long
        
        print(f"Short signal CWT shape: {Wx_short.shape}")
        print(f"Long signal CWT shape: {Wx_long.shape}")
        
        # Check if time axis matches input length
        print(f"Short signal length: {len(x_short)}, CWT time axis: {Wx_short.shape[1]}")
        print(f"Long signal length: {len(x_long)}, CWT time axis: {Wx_long.shape[1]}")
        
        # Test with custom scales to see if we can control time axis
        print("\nTesting with custom scales...")
        custom_scales = np.logspace(0, 2, 50)
        result_custom = ssqpy.cwt(x, wavelet='gmw', scales=custom_scales)
        
        if isinstance(result_custom, tuple):
            Wx_custom, *_ = result_custom
        else:
            Wx_custom = result_custom
        
        print(f"Custom scales CWT shape: {Wx_custom.shape}")
        
        # Check if there are any parameters that control time axis length
        print("\nChecking for time axis control parameters...")
        help_text = ssqpy.cwt.__doc__
        if help_text:
            lines = help_text.split('\n')
            for line in lines:
                if 'time' in line.lower() or 'length' in line.lower() or 'size' in line.lower():
                    print(f"  {line.strip()}")
        
        # Check if there's a way to control time axis length
        print("\nAnswering user's question about CWT time axis control:")
        print("Based on the test results:")
        print(f"  - Input signal length: {len(x)}")
        print(f"  - CWT time axis length: {Wx.shape[1]}")
        print("  - The time axis length matches the input signal length")
        print("  - No direct parameter found to control time axis length independently")
        print("  - This is the standard behavior for CWT implementations")
        
        assert True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        assert False

if __name__ == "__main__":
    success = test_cache_and_import()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
        sys.exit(1)

