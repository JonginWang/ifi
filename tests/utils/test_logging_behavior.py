#!/usr/bin/env python3
"""
Test script to demonstrate logging behavior with LogManager
"""

import logging

from ifi.utils.common import LogManager

def test_logging_behavior():
    """Test different logging scenarios"""
    
    print("=" * 60)
    print("TESTING LOGGING BEHAVIOR")
    print("=" * 60)
    
    # Initialize LogManager with INFO level
    LogManager(level="INFO")
    
    # Create specialized loggers
    analysis_logger = LogManager().create_specialized_logger('ifi.analysis', 'DEBUG')
    test_logger = LogManager().create_specialized_logger('ifi.test', 'WARNING')
    gui_logger = LogManager().create_specialized_logger('ifi.gui', 'ERROR')
    
    print("\n1. ROOT LOGGER (logging.<level>()):")
    print("   - Level: INFO (set by LogManager)")
    print("   - Output: Console + File")
    logging.debug("ROOT: Debug message")      # Won't show (INFO level)
    logging.info("ROOT: Info message")        # Will show
    logging.warning("ROOT: Warning message")  # Will show
    logging.error("ROOT: Error message")      # Will show
    
    print("\n2. ANALYSIS LOGGER (analysis_logger.<level>()):")
    print("   - Level: DEBUG")
    print("   - Output: Console + File (inherits from root)")
    analysis_logger.debug("ANALYSIS: Debug message")    # Will show (DEBUG level)
    analysis_logger.info("ANALYSIS: Info message")      # Will show
    analysis_logger.warning("ANALYSIS: Warning message") # Will show
    
    print("\n3. TEST LOGGER (test_logger.<level>()):")
    print("   - Level: WARNING")
    print("   - Output: Console + File (inherits from root)")
    test_logger.debug("TEST: Debug message")      # Won't show (WARNING level)
    test_logger.info("TEST: Info message")        # Won't show (WARNING level)
    test_logger.warning("TEST: Warning message")  # Will show
    test_logger.error("TEST: Error message")      # Will show
    
    print("\n4. GUI LOGGER (gui_logger.<level>()):")
    print("   - Level: ERROR")
    print("   - Output: Console + File (inherits from root)")
    gui_logger.debug("GUI: Debug message")      # Won't show (ERROR level)
    gui_logger.info("GUI: Info message")        # Won't show (ERROR level)
    gui_logger.warning("GUI: Warning message")  # Won't show (ERROR level)
    gui_logger.error("GUI: Error message")      # Will show
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- logging.<level>() → ROOT logger (INFO level)")
    print("- analysis_logger.<level>() → ifi.analysis logger (DEBUG level)")
    print("- test_logger.<level>() → ifi.test logger (WARNING level)")
    print("- gui_logger.<level>() → ifi.gui logger (ERROR level)")
    print("=" * 60)

if __name__ == '__main__':
    test_logging_behavior()
