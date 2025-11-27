#!/usr/bin/env python3
"""
Smart Analysis Runner
=====================
Automatically determines execution strategy based on file counts per shot.

This script:
1. Takes a shot range (e.g., 45000:45010) as input
2. Queries NAS DB to find existing shot numbers and their file counts
3. Executes analysis with smart grouping:
   - 3 files per shot → run analysis for single shot
   - 1-2 files per shot → combine 2 shots into one analysis run

Usage:
    python scripts/run_analysis_smart.py 45000:45010 [additional args...]
    python scripts/run_analysis_smart.py 45000:45010 --freq 280 --density --stft --save_data

Example:
    python scripts/run_analysis_smart.py 45000:45010 --freq 280 --density --stft --stft_cols 0 1 --save_plots --save_data --color_density_by_amplitude --vest_fields 102 109 101 144 214 171
"""

import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

try:
    from ifi.db_controller.nas_db import NAS_DB
    from ifi.utils.common import FlatShotList, get_project_root
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    sys.exit(1)


def parse_shot_range(range_str: str) -> List[int]:
    """
    Parse a shot range string into a list of shot numbers.
    
    Args:
        range_str: Range string like "45000:45010"
        
    Returns:
        List of shot numbers
    """
    if ":" not in range_str:
        try:
            return [int(range_str)]
        except ValueError:
            raise ValueError(f"Invalid shot range format: {range_str}")
    
    parts = range_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid shot range format: {range_str}. Expected 'start:end'")
    
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"Invalid range: start ({start}) > end ({end})")
    
    return list(range(start, end + 1))


def get_file_counts_for_shots(nas_db: NAS_DB, shot_numbers: List[int]) -> dict[int, int]:
    """
    Query NAS DB to get file counts for each shot number.
    
    Args:
        nas_db: NAS_DB instance
        shot_numbers: List of shot numbers to check
        
    Returns:
        Dictionary mapping shot number to file count
    """
    file_counts = {}
    
    print(f"Querying NAS DB for {len(shot_numbers)} shot numbers...")
    for shot_num in shot_numbers:
        try:
            files = nas_db.find_files(query=shot_num, force_remote=False)
            file_count = len(files) if files else 0
            file_counts[shot_num] = file_count
            
            if file_count > 0:
                print(f"  Shot {shot_num}: {file_count} file(s)")
            else:
                print(f"  Shot {shot_num}: No files found")
        except Exception as e:
            print(f"  Shot {shot_num}: Error querying - {e}")
            file_counts[shot_num] = 0
    
    return file_counts


def group_shots_for_execution(file_counts: dict[int, int]) -> List[Tuple[List[int], str]]:
    """
    Group shots based on file counts for optimal execution.
    
    Strategy:
    - 3 files per shot → run single shot
    - 1-2 files per shot → combine 2 shots
    
    Args:
        file_counts: Dictionary mapping shot number to file count
        
    Returns:
        List of tuples: (list of shot numbers, reason string)
    """
    execution_groups = []
    
    # Separate shots by file count
    shots_with_3_files = [shot for shot, count in file_counts.items() if count == 3]
    shots_with_1_or_2_files = [shot for shot, count in file_counts.items() if count in [1, 2]]
    shots_with_0_files = [shot for shot, count in file_counts.items() if count == 0]
    shots_with_more_than_3 = [shot for shot, count in file_counts.items() if count > 3]
    
    # Group shots with 3 files: run individually
    for shot in sorted(shots_with_3_files):
        execution_groups.append(([shot], f"3 files"))
    
    # Group shots with 1-2 files: combine 2 shots
    shots_with_1_or_2_sorted = sorted(shots_with_1_or_2_files)
    i = 0
    while i < len(shots_with_1_or_2_sorted):
        if i + 1 < len(shots_with_1_or_2_sorted):
            # Combine 2 shots
            shot1, shot2 = shots_with_1_or_2_sorted[i], shots_with_1_or_2_sorted[i + 1]
            count1, count2 = file_counts[shot1], file_counts[shot2]
            execution_groups.append(
                ([shot1, shot2], f"{count1}+{count2} files (combined)")
            )
            i += 2
        else:
            # Single shot left with 1-2 files
            shot = shots_with_1_or_2_sorted[i]
            count = file_counts[shot]
            execution_groups.append(([shot], f"{count} file(s) (single)"))
            i += 1
    
    # Handle shots with more than 3 files: run individually
    for shot in sorted(shots_with_more_than_3):
        count = file_counts[shot]
        execution_groups.append(([shot], f"{count} files"))
    
    # Report shots with 0 files
    if shots_with_0_files:
        print(f"\nWarning: {len(shots_with_0_files)} shot(s) have no files and will be skipped:")
        for shot in sorted(shots_with_0_files):
            print(f"  Shot {shot}")
    
    return execution_groups


def format_shot_query(shot_numbers: List[int]) -> str:
    """
    Format a list of shot numbers into a query string for main_analysis.
    
    Args:
        shot_numbers: List of shot numbers
        
    Returns:
        Query string (e.g., "45000:45001" or "45000")
    """
    if len(shot_numbers) == 1:
        return str(shot_numbers[0])
    elif len(shot_numbers) == 2:
        return f"{shot_numbers[0]}:{shot_numbers[1]}"
    else:
        # For more than 2 shots, use range format if consecutive, otherwise comma-separated
        sorted_shots = sorted(shot_numbers)
        if sorted_shots == list(range(sorted_shots[0], sorted_shots[-1] + 1)):
            return f"{sorted_shots[0]}:{sorted_shots[-1]}"
        else:
            return ",".join(map(str, sorted_shots))


def run_analysis_for_group(shot_numbers: List[int], additional_args: List[str], project_root: Path) -> int:
    """
    Run analysis for a group of shot numbers.
    
    Args:
        shot_numbers: List of shot numbers to analyze
        additional_args: Additional command-line arguments
        project_root: Project root directory
        
    Returns:
        Exit code from the analysis process
    """
    query_str = format_shot_query(shot_numbers)
    
    cmd = [
        sys.executable,
        "-m", "ifi.analysis.main_analysis",
        query_str
    ] + additional_args
    
    print(f"\n{'='*60}")
    print(f"Running analysis for: {query_str}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        cmd,
        cwd=project_root,
        check=False
    )
    
    return result.returncode


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_analysis_smart.py SHOT_RANGE [additional args...]")
        print("Example: python scripts/run_analysis_smart.py 45000:45010 --freq 280 --density --stft --save_data")
        sys.exit(1)
    
    shot_range_str = sys.argv[1]
    additional_args = sys.argv[2:]
    
    # Parse shot range
    try:
        shot_numbers = parse_shot_range(shot_range_str)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Parsed shot range: {shot_range_str}")
    print(f"Total shots to check: {len(shot_numbers)}")
    print()
    
    # Get project root
    project_root = get_project_root()
    config_path = project_root / "ifi" / "config.ini"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Initialize NAS DB
    print("Connecting to NAS DB...")
    nas_db = NAS_DB(config_path=str(config_path))
    try:
        if not nas_db.connect():
            print("Error: Failed to connect to NAS DB")
            sys.exit(1)
        
        # Get file counts for all shots
        file_counts = get_file_counts_for_shots(nas_db, shot_numbers)
        
        # Count existing shots
        existing_shots = {shot: count for shot, count in file_counts.items() if count > 0}
        print(f"\nSummary:")
        print(f"  Total shots checked: {len(shot_numbers)}")
        print(f"  Shots with files: {len(existing_shots)}")
        print(f"  Shots without files: {len(shot_numbers) - len(existing_shots)}")
        
        if not existing_shots:
            print("\nNo shots with files found. Exiting.")
            sys.exit(0)
        
        # Group shots for execution
        execution_groups = group_shots_for_execution(file_counts)
        
        print(f"\nExecution plan: {len(execution_groups)} group(s)")
        for i, (shots, reason) in enumerate(execution_groups, 1):
            print(f"  Group {i}: Shots {shots} ({reason})")
        
        # Execute analysis for each group
        print(f"\n{'='*60}")
        print("Starting analysis execution...")
        print(f"{'='*60}\n")
        
        failed_groups = []
        for i, (shots, reason) in enumerate(execution_groups, 1):
            print(f"\n[Group {i}/{len(execution_groups)}] Processing shots: {shots} ({reason})")
            exit_code = run_analysis_for_group(shots, additional_args, project_root)
            
            if exit_code != 0:
                print(f"\n[ERROR] Analysis failed for group {i} (shots: {shots}) with exit code {exit_code}")
                failed_groups.append((i, shots, exit_code))
            else:
                print(f"\n[SUCCESS] Analysis completed for group {i} (shots: {shots})")
        
        # Final summary
        print(f"\n{'='*60}")
        print("Execution Summary")
        print(f"{'='*60}")
        print(f"Total groups: {len(execution_groups)}")
        print(f"Successful: {len(execution_groups) - len(failed_groups)}")
        print(f"Failed: {len(failed_groups)}")
        
        if failed_groups:
            print("\nFailed groups:")
            for group_num, shots, exit_code in failed_groups:
                print(f"  Group {group_num}: Shots {shots} (exit code: {exit_code})")
            sys.exit(1)
        else:
            print("\nAll analyses completed successfully!")
            sys.exit(0)
    
    finally:
        nas_db.disconnect()


if __name__ == "__main__":
    main()

