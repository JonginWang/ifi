"""Test script for get_interferometry_params function."""

from ifi.file_io import get_interferometry_params
from ifi.utils import LogManager

LogManager()

def test_params():
    print("Testing get_interferometry_params function:")
    print("=" * 50)
    
    # Test cases based on the rules defined
    test_cases = [
        (45821, "45821_ALL.csv"),    # Rule 3: >= 41542, _ALL -> 280GHz CDM
        (45821, "45821_056.csv"),    # Rule 3: >= 41542, _0XX -> 94GHz CDM
        (45821, "45821_789.csv"),    # Rule 3: >= 41542, other -> 94GHz CDM
        (40000, "40000.dat"),        # Rule 2: 39302-41398 -> 94GHz FPGA
        (30000, "30000.csv"),        # Rule 1: 0-39265 -> 94GHz IQ
        (50000, "50000.csv"),        # Outside range -> unknown
    ]
    
    for shot_num, filename in test_cases:
        params = get_interferometry_params(shot_num, filename)
        print(f"Shot {shot_num}, file '{filename}':")
        print(f"  Method: {params['method']}")
        print(f"  Frequency: {params['freq']} GHz")
        print(f"  Reference column: {params['ref_col']}")
        print(f"  Probe columns: {params['probe_cols']}")
        if 'amp_ref_col' in params:
            print(f"  Amplitude ref column: {params['amp_ref_col']}")
        if 'amp_probe_cols' in params:
            print(f"  Amplitude probe columns: {params['amp_probe_cols']}")
        print()

if __name__ == "__main__":
    test_params()