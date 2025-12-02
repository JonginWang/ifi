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
from typing import List, Tuple

# Add project root to path (must be done before importing IFI modules)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# IFI module imports (after path setup - linter warning is intentional)

try:
    from ifi import get_project_root
    from ifi.db_controller.nas_db import NAS_DB
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
        execution_groups.append(([shot], "3 files"))
    
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


def process_shot_range(shot_range_str: str, additional_args: List[str], project_root: Path, nas_db: NAS_DB) -> Tuple[bool, List[Tuple[int, List[int], int]]]:
    """
    Process a single shot range and return execution results.
    
    Args:
        shot_range_str: Shot range string (e.g., "45000:45010")
        additional_args: Additional command-line arguments
        project_root: Project root directory
        nas_db: NAS_DB instance
        
    Returns:
        Tuple of (success bool, list of failed groups)
    """
    # Parse shot range
    try:
        shot_numbers = parse_shot_range(shot_range_str)
    except ValueError as e:
        print(f"Error parsing shot range '{shot_range_str}': {e}")
        return False, []
    
    print(f"\n{'='*60}")
    print(f"Processing shot range: {shot_range_str}")
    print(f"Total shots to check: {len(shot_numbers)}")
    print(f"{'='*60}")
    
    # Get file counts for all shots
    file_counts = get_file_counts_for_shots(nas_db, shot_numbers)
    
    # Count existing shots
    existing_shots = {shot: count for shot, count in file_counts.items() if count > 0}
    print("\nSummary:")
    print(f"  Total shots checked: {len(shot_numbers)}")
    print(f"  Shots with files: {len(existing_shots)}")
    print(f"  Shots without files: {len(shot_numbers) - len(existing_shots)}")
    
    if not existing_shots:
        print("\nNo shots with files found. Skipping this range.")
        return True, []
    
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
    
    return len(failed_groups) == 0, failed_groups


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Analysis Runner - Automatically determines execution strategy based on file counts per shot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single shot range
  python scripts/run_analysis_smart.py 45000:45010 --freq 280 --density --stft --save_data
  
  # Multiple shot ranges
  python scripts/run_analysis_smart.py --shot-list "45000:45010" "45020:45030" --freq 280 --density --stft --save_data
  
  # Using example shot list from script (if --shot-list not provided, uses defaults)
  python scripts/run_analysis_smart.py --freq 280 --density --stft --save_data
        """
    )
    
    parser.add_argument(
        "--shot-list",
        nargs="+",
        help="One or more shot ranges (e.g., '45000:45010' '45020:45030'). "
             "If not provided, uses example list from script."
    )
    
    # Parse known args to separate shot ranges from additional args
    args, additional_args = parser.parse_known_args()
    
    # Example shot list (used if --shot-list is not provided)
    example_shot_ranges = [
        "46687:46691", "46692:46695", "46696:46699", "46700:46703",
        "46595:46599", "46600:46603", "46604:46607", "46608:46611", "46612:46615"
    ]
    
    # Determine shot ranges to process
    if args.shot_list:
        shot_ranges = args.shot_list
    else:
        # Check if first positional argument is a shot range (backward compatibility)
        if additional_args and not any(arg.startswith("--") for arg in additional_args[:1]):
            shot_ranges = [additional_args[0]]
            additional_args = additional_args[1:]
        else:
            shot_ranges = example_shot_ranges
            print("Using example shot list (use --shot-list to override)")
            print(f"Example ranges: {', '.join(shot_ranges[:3])}...")
            print()
    
    if not shot_ranges:
        parser.print_help()
        sys.exit(1)
    
    print(f"{'='*60}")
    print("Smart Analysis Runner - Batch Mode")
    print(f"{'='*60}")
    print(f"Total shot ranges to process: {len(shot_ranges)}")
    print(f"Shot ranges: {', '.join(shot_ranges)}")
    print(f"Additional arguments: {' '.join(additional_args) if additional_args else '(none)'}")
    print(f"{'='*60}\n")
    
    # Get project root
    project_root = get_project_root()
    config_path = project_root / "ifi" / "config.ini"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Initialize NAS DB (reuse connection for all ranges)
    print("Connecting to NAS DB...")
    nas_db = NAS_DB(config_path=str(config_path))
    try:
        if not nas_db.connect():
            print("Error: Failed to connect to NAS DB")
            sys.exit(1)
        
        # Process each shot range
        all_failed_groups = []
        successful_ranges = []
        failed_ranges = []
        
        for range_idx, shot_range_str in enumerate(shot_ranges, 1):
            print(f"\n{'#'*60}")
            print(f"Range {range_idx}/{len(shot_ranges)}: {shot_range_str}")
            print(f"{'#'*60}")
            
            success, failed_groups = process_shot_range(
                shot_range_str, additional_args, project_root, nas_db
            )
            
            if success:
                successful_ranges.append(shot_range_str)
            else:
                failed_ranges.append(shot_range_str)
                all_failed_groups.extend([(range_idx, group_num, shots, exit_code) 
                                         for group_num, shots, exit_code in failed_groups])
        
        # Final summary
        print(f"\n{'='*60}")
        print("Final Execution Summary")
        print(f"{'='*60}")
        print(f"Total ranges processed: {len(shot_ranges)}")
        print(f"Successful ranges: {len(successful_ranges)}")
        print(f"Failed ranges: {len(failed_ranges)}")
        
        if successful_ranges:
            print("\nSuccessful ranges:")
            for range_str in successful_ranges:
                print(f"  ✓ {range_str}")
        
        if failed_ranges:
            print("\nFailed ranges:")
            for range_str in failed_ranges:
                print(f"  ✗ {range_str}")
        
        if all_failed_groups:
            print("\nFailed groups:")
            for range_idx, group_num, shots, exit_code in all_failed_groups:
                print(f"  Range {range_idx}, Group {group_num}: Shots {shots} (exit code: {exit_code})")
            sys.exit(1)
        else:
            print("\nAll analyses completed successfully!")
            sys.exit(0)
    
    finally:
        nas_db.disconnect()


if __name__ == "__main__":
    main()

