#!/usr/bin/env python3
"""
Example script for running IFI analysis using run_analysis function.

This script demonstrates how to use run_analysis programmatically
instead of using command-line interface. Analysis configurations are defined
at the top of the file and can be executed sequentially or selectively.

Usage:
    python scripts/run_analysis_example.py
    python scripts/run_analysis_example.py --config 0  # Run only first config
    python scripts/run_analysis_example.py --config 0 2  # Run configs 0 and 2
"""

import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional

# Add project root to path (must be done before importing IFI modules)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# IFI module imports (after path setup - linter warning is intentional)
from ifi.db_controller.nas_db import NAS_DB  # noqa: E402
from ifi.db_controller.vest_db import VEST_DB  # noqa: E402
from ifi.analysis.main_analysis import run_analysis  # noqa: E402
from ifi.utils.common import LogManager  # noqa: E402

# Initialize logging
logger = LogManager().get_logger(__name__, level="INFO")

#%%
# ============================================================================
# ANALYSIS CONFIGURATIONS
# ============================================================================
# Define analysis configurations here. Each dictionary represents a complete
# set of arguments that would be passed to the command-line interface.
#
# Available parameters (all optional except 'query'):
#   - query: List of shot numbers or file paths (required)
#   - data_folders: Comma-separated list of data folders
#   - add_path: Add data_folders to default paths (bool)
#   - force_remote: Force remote fetch, ignore cache (bool)
#   - results_dir: Directory for results (str, default: "ifi/results")
#   - no_offset_removal: Disable offset removal (bool)
#   - offset_window: Window size for offset removal (int, default: 2001)
#   - stft: Enable STFT analysis (bool)
#   - stft_cols: Column indices for STFT (list of int)
#   - cwt: Enable CWT analysis (bool)
#   - cwt_cols: Column indices for CWT (list of int)
#   - plot: Show interactive plots (bool)
#   - no_plot_block: Non-blocking plot mode (bool)
#   - no_plot_raw: Don't plot raw data (bool)
#   - no_plot_ft: Don't plot time-frequency transforms (bool)
#   - downsample: Downsample factor for plotting (int, default: 10)
#   - trigger_time: Trigger time in seconds (float, default: 0.290)
#   - density: Enable density calculation (bool)
#   - freq: Filter to specific frequency groups (list of float: [94.0] or [280.0] or [94.0, 280.0])
#   - vest_fields: VEST DB field IDs to load (list of int)
#   - baseline: Baseline correction mode ("ip" or "trig")
#   - color_density_by_amplitude: Color-code density plots by amplitude (bool)
#   - amplitude_colormap: Colormap for amplitude coloring (str, default: "coolwarm")
#   - amplitude_impedance: System impedance in ohms [Ω] (float, default: 50.0)
#   - save_plots: Save plots to files (bool)
#   - save_data: Save data to HDF5 (bool)
#   - scheduler: Dask scheduler ("threads", "processes", "single-threaded")
# ============================================================================
#%%
ANALYSIS_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "basic_analysis",
        "description": "Basic analysis for a single shot (STFT + Density)",
        "enabled": True,  # Set to False to skip this configuration
        "query": ["45821"],
        "stft": True,
        "cwt": False,
        "density": True,
        "plot": False,
        "save_data": True,
        "save_plots": False,
        "scheduler": "threads",
    },
    {
        "name": "multiple_shots_cwt",
        "description": "Multiple shots with CWT analysis",
        "enabled": False,  # Disabled by default (CWT is memory-intensive)
        "query": ["45821", "45822", "45823"],
        "stft": True,
        "cwt": True,
        "density": True,
        "plot": False,
        "save_data": True,
        "scheduler": "processes",  # Use processes for better CPU utilization
    },
    {
        "name": "shot_range",
        "description": "Shot range analysis",
        "enabled": False,
        "query": ["45821:45825"],  # Range notation
        "stft": True,
        "cwt": False,
        "density": True,
        "save_data": True,
        "scheduler": "threads",
    },
    {
        "name": "with_vest_baseline",
        "description": "Analysis with VEST data and baseline correction",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "density": True,
        "save_data": True,
        "vest_fields": ["Ip", "Bt"],  # VEST field names (will be converted to IDs)
        "baseline": "ip",  # Baseline correction using Ip
    },
    {
        "name": "custom_columns",
        "description": "Analysis with specific column selection",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "stft_cols": [0, 1],  # Only analyze first two columns
        "cwt": True,
        "cwt_cols": [0],  # Only analyze first column for CWT
        "density": True,
        "save_data": True,
    },
    {
        "name": "plotting_mode",
        "description": "Analysis with interactive plotting enabled",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "density": True,
        "plot": True,
        "no_plot_block": True,  # Non-blocking plots
        "save_plots": True,  # Also save plots
        "save_data": True,
    },
    {
        "name": "amplitude_colored_density",
        "description": "Analysis with amplitude-based color-coding for density plots",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "density": True,
        "color_density_by_amplitude": True,  # Enable amplitude-based coloring
        "amplitude_colormap": "coolwarm",  # Colormap for amplitude
        "amplitude_impedance": 50.0,  # System impedance [Ω]
        "plot": False,  # Disable interactive plots for testing
        "save_data": True,
        "save_plots": True,  # Save plots to see the color-coding
    },
    {
        "name": "frequency_filtered_94ghz",
        "description": "Analysis filtered to 94GHz frequency only",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "density": True,
        "freq": [94.0],  # Process only 94GHz frequency
        "save_data": True,
    },
    {
        "name": "frequency_filtered_280ghz",
        "description": "Analysis filtered to 280GHz frequency only",
        "enabled": False,
        "query": ["45821"],
        "stft": True,
        "density": True,
        "freq": [280.0],  # Process only 280GHz frequency
        "save_data": True,
    },
]

#%%
def create_analysis_args(config: Dict[str, Any]) -> argparse.Namespace:
    """
    Create argparse.Namespace object from configuration dictionary.

    This function converts a configuration dictionary (as defined in ANALYSIS_CONFIGS)
    into an argparse.Namespace object compatible with run_analysis.

    Args:
        config: Configuration dictionary containing analysis parameters.
            Must contain 'query' key. All other parameters are optional.

    Returns:
        argparse.Namespace: Namespace object with all analysis parameters.

    Raises:
        ValueError: If 'query' is missing from config.
    """
    if "query" not in config:
        raise ValueError("Configuration must contain 'query' key")

    args = argparse.Namespace()

    # Query (required)
    query = config["query"]
    # Convert to list if single string
    if isinstance(query, str):
        args.query = [query]
    else:
        args.query = query

    # Data loading
    args.data_folders = config.get("data_folders", None)
    args.add_path = config.get("add_path", False)
    args.force_remote = config.get("force_remote", False)
    args.results_dir = config.get("results_dir", "ifi/results")

    # Data processing
    args.no_offset_removal = config.get("no_offset_removal", False)
    args.offset_window = config.get("offset_window", 2001)

    # Analysis flags
    args.stft = config.get("stft", False)
    args.cwt = config.get("cwt", False)
    args.density = config.get("density", False)
    args.plot = config.get("plot", False)

    # STFT/CWT parameters
    args.stft_cols = config.get("stft_cols", [])
    args.cwt_cols = config.get("cwt_cols", [])
    args.no_plot_raw = config.get("no_plot_raw", False)
    args.no_plot_ft = config.get("no_plot_ft", False)
    args.no_plot_block = config.get("no_plot_block", False)

    # Plotting parameters
    args.downsample = config.get("downsample", 10)
    args.trigger_time = config.get("trigger_time", 0.290)

    # VEST data
    vest_fields = config.get("vest_fields", [])
    # Convert field names to IDs if needed (simplified - may need actual mapping)
    args.vest_fields = vest_fields if isinstance(vest_fields, list) else []

    # Baseline correction
    args.baseline = config.get("baseline", None)

    # Frequency filtering
    freq = config.get("freq", None)
    if freq is not None:
        # Convert to list if single value
        if isinstance(freq, (int, float)):
            args.freq = [float(freq)]
        elif isinstance(freq, list):
            args.freq = [float(f) for f in freq]
        else:
            args.freq = None
    else:
        args.freq = None

    # Density plotting options
    args.color_density_by_amplitude = config.get("color_density_by_amplitude", False)
    args.amplitude_colormap = config.get("amplitude_colormap", "coolwarm")
    args.amplitude_impedance = config.get("amplitude_impedance", 50.0)

    # Output
    args.save_plots = config.get("save_plots", False)
    args.save_data = config.get("save_data", False)

    # Dask scheduler
    args.scheduler = config.get("scheduler", "threads")

    return args

#%%
def run_configuration(
    config: Dict[str, Any],
    nas_db: NAS_DB,
    vest_db: VEST_DB,
    config_index: Optional[int] = None,
) -> bool:
    """
    Run a single analysis configuration.

    Args:
        config: Configuration dictionary from ANALYSIS_CONFIGS.
        nas_db: Initialized NAS_DB instance.
        vest_db: Initialized VEST_DB instance.
        config_index: Optional index of the configuration (for logging).

    Returns:
        bool: True if analysis completed successfully, False otherwise.
    """
    config_name = config.get("name", f"config_{config_index}")
    description = config.get("description", "No description")
    enabled = config.get("enabled", True)

    if not enabled:
        logger.info(f"Skipping disabled configuration: {config_name}")
        return False

    logger.info("=" * 80)
    logger.info(f"Running configuration: {config_name}")
    if config_index is not None:
        logger.info(f"Configuration index: {config_index}")
    logger.info(f"Description: {description}")
    logger.info("=" * 80)

    try:
        # Create arguments from configuration
        args = create_analysis_args(config)

        # Run analysis
        results = run_analysis(
            query=args.query, args=args, nas_db=nas_db, vest_db=vest_db
        )

        if results:
            logger.info(
                f"Configuration '{config_name}' completed successfully. "
                f"Processed {len(results)} shot(s)."
            )
            return True
        else:
            logger.warning(f"Configuration '{config_name}' returned no results.")
            return False

    except Exception as e:
        logger.error(f"Configuration '{config_name}' failed: {e}", exc_info=True)
        return False

#%%
def list_configurations():
    """List all available analysis configurations."""
    logger.info("Available analysis configurations:")
    logger.info("=" * 80)
    for idx, config in enumerate(ANALYSIS_CONFIGS):
        name = config.get("name", f"config_{idx}")
        description = config.get("description", "No description")
        enabled = "✓" if config.get("enabled", True) else "✗"
        query = config.get("query", [])
        logger.info(
            f"  [{idx}] {enabled} {name}"
            f"\n      Description: {description}"
            f"\n      Query: {query}"
        )
    logger.info("=" * 80)

#%%
def main():
    """
    Main function to run analysis configurations.

    This function:
    1. Parses command-line arguments to select which configurations to run
    2. Initializes database controllers
    3. Runs selected configurations sequentially
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run IFI analysis using predefined configurations",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=int,
        nargs="+",
        help="Run specific configuration(s) by index. "
        "If not specified, runs all enabled configurations.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configurations regardless of 'enabled' flag.",
    )

    cli_args = parser.parse_args()

    # List configurations and exit if requested
    if cli_args.list:
        list_configurations()
        return

    # Initialize database controllers
    try:
        nas_db = NAS_DB(config_path="ifi/config.ini")
        vest_db = VEST_DB(config_path="ifi/config.ini")
        logger.info("Database controllers initialized successfully.")
    except FileNotFoundError:
        logger.error("Configuration file 'ifi/config.ini' not found. Exiting.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize database controllers: {e}")
        return

    # Determine which configurations to run
    if cli_args.config is not None:
        # Run specific configurations by index
        configs_to_run = []
        for idx in cli_args.config:
            if 0 <= idx < len(ANALYSIS_CONFIGS):
                configs_to_run.append((idx, ANALYSIS_CONFIGS[idx]))
            else:
                logger.warning(f"Configuration index {idx} is out of range. Skipping.")

        if not configs_to_run:
            logger.error("No valid configurations to run.")
            return
    else:
        # Run all enabled configurations (or all if --all is specified)
        if cli_args.all:
            configs_to_run = [
                (idx, config) for idx, config in enumerate(ANALYSIS_CONFIGS)
            ]
        else:
            configs_to_run = [
                (idx, config)
                for idx, config in enumerate(ANALYSIS_CONFIGS)
                if config.get("enabled", True)
            ]

        if not configs_to_run:
            logger.warning(
                "No enabled configurations found. Use --all to run all configurations."
            )
            return

    # Run configurations
    logger.info(f"Running {len(configs_to_run)} configuration(s)...")
    successful = 0
    failed = 0

    for idx, config in configs_to_run:
        success = run_configuration(config, nas_db, vest_db, config_index=idx)
        if success:
            successful += 1
        else:
            failed += 1
        logger.info("")  # Empty line between configurations

    # Summary
    logger.info("=" * 80)
    logger.info("Analysis Summary:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(configs_to_run)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
