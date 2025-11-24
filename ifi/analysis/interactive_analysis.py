#!/usr/bin/env python3
"""
Interactive Analysis
====================

This Script is an Interactive analysis pipeline for IFI package.
It is used to test the analysis pipeline and the results.

Functions:
    create_mock_args: Creates a mock argparse.Namespace object for interactive analysis.
        Creates a mock argparse.Namespace object for interactive analysis.
        Runs the analysis pipeline.

        Options:
            - query: Shot number or pattern
            - density: Whether to run the density analysis.
            - plot: Whether to plot the results.
            - overview_plot: Whether to plot the overview.
            - data_folders: Data folders to use.
            - add_path: Whether to add the path to the data folders.
            - force_remote: Whether to force the remote data.
            - vest_fields: VEST fields to use.
            - no_offset_removal: Whether to remove the offset.
            - offset_window: Window size for the offset removal.
            - baseline: Baseline to use.
            - plot_raw: Whether to plot the raw data.
            - trigger_time: Trigger time.
            - downsample: Downsample factor.
            - stft: Whether to run the STFT analysis.
            - stft_cols: STFT columns to use.
            - cwt: Whether to run the CWT analysis.
            - cwt_cols: CWT columns to use.
            - results_dir: Results directory.
            - save_plots: Whether to save the plots.
            - save_data: Whether to save the data.
            - scheduler: Scheduler to use.

        Returns:
            argparse.Namespace object for interactive analysis.
"""

import logging
from argparse import Namespace
import matplotlib.pyplot as plt
try:
    from .main_analysis import run_analysis
    from ..utils.common import LogManager
    from ..db_controller.nas_db import NAS_DB
    from ..db_controller.vest_db import VEST_DB
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.analysis.main_analysis import run_analysis
    from ifi.utils.common import LogManager
    from ifi.db_controller.nas_db import NAS_DB
    from ifi.db_controller.vest_db import VEST_DB

LogManager(level="DEBUG")


def create_mock_args():
    """
    Creates a mock argparse.Namespace object for interactive analysis.
    This simulates the command-line arguments for the analysis pipeline.

    Returns:
        Namespace (object): argparse.Namespace object for interactive analysis.
    """
    args = Namespace(
        # --- Essential Arguments ---
        query=["45821"],  # Shot number or pattern
        freq=[94.0, 280.0],
        density=True,
        plot=False,  # Disable interactive plotting for testing (can enable later)
        overview_plot=False,
        # --- Data Source Arguments ---
        data_folders=None,
        add_path=False,
        force_remote=False,
        vest_fields=[109, 101],
        # --- Processing Arguments ---
        no_offset_removal=False,
        offset_window=2001,
        baseline="ip",  # 'ip', 'trig', or None
        # --- Plotting Arguments ---
        plot_raw=False,
        no_plot_raw=False,
        no_plot_ft=False,
        trigger_time=0.290,
        downsample=10,
        # --- Density Plotting Options ---
        color_density_by_amplitude=False,  # Enable amplitude-based color-coding for density plots
        amplitude_colormap="coolwarm",  # Colormap for amplitude coloring
        amplitude_impedance=50.0,  # System impedance in ohms [Ω]
        # --- STFT/CWT Arguments ---
        stft=True,
        stft_cols=[0, 1],  # Column indices for STFT analysis
        cwt=False,  # Disable CWT for this test (STFT, density, plot, save only)
        cwt_cols=[0, 1],  # Column indices for CWT analysis (not used when cwt=False)
        # --- Saving Arguments ---
        results_dir="ifi/results",
        save_plots=False,
        save_data=True,  # Enable data saving to test HDF5 output
        # --- Performance Arguments ---
        scheduler="single-threaded",  # Use 'single-threaded' for easier debugging
    )
    return args


if __name__ == "__main__":
    # 1. Set up analysis parameters
    # You can modify these arguments directly for your specific test case.
    analysis_args = create_mock_args()

    # To test a different shot, for example:
    # analysis_args.query = ['44149']
    # analysis_args.stft = True
    # analysis_args.density = True
    # analysis_args.vest_fields = []
    # analysis_args.overview_plot = False

    # 2. Initialize database controllers
    try:
        nas_db = NAS_DB(config_path="ifi/config.ini")
        vest_db = VEST_DB(config_path="ifi/config.ini")
        logging.info("Database controllers initialized successfully.")
    except FileNotFoundError:
        logging.error("Configuration file 'ifi/config.ini' not found. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize database controllers: {e}")
        exit(1)

    # 3. Run the analysis pipeline
    # The results are returned in a dictionary.
    logging.info("Starting interactive analysis...")
    results = run_analysis(
        query=analysis_args.query,
        args=analysis_args,
        nas_db=nas_db,
        vest_db=vest_db
    )
    logging.info("Analysis finished.")

    # 4. Access and explore the results
    # The 'results' dictionary contains all the major data artifacts.
    # You can now inspect these variables in Spyder's Variable Explorer.
    if results:
        logging.info("--- Available Data ---")
        logging.info(f"Number of shots analyzed: {len(results)}")
        
        for shot_num, bundle in results.items():
            logging.info(f"\n--- Shot #{shot_num} ---")
            
            # Access processed data
            processed_data = bundle.get("processed_data", {})
            if processed_data:
                signals = processed_data.get("signals", {})
                density = processed_data.get("density", {})
                
                if signals:
                    logging.info(f"Signals available for frequencies: {list(signals.keys())} GHz")
                    for freq, df in signals.items():
                        logging.info(f"  {freq} GHz: Shape {df.shape}, Columns: {list(df.columns)[:5]}...")
                
                if density:
                    logging.info(f"Density data available for frequencies: {list(density.keys())} GHz")
                    for freq, df in density.items():
                        logging.info(f"  {freq} GHz: Shape {df.shape}, Columns: {list(df.columns)[:5]}...")
            
            # Access analysis results
            analysis_results = bundle.get("analysis_results", {})
            stft_results = analysis_results.get("stft", {})
            cwt_results = analysis_results.get("cwt", {})
            
            if stft_results:
                logging.info(f"STFT Results available for: {list(stft_results.keys())}")
            if cwt_results:
                logging.info(f"CWT Results available for: {list(cwt_results.keys())}")
            
            # Access VEST data
            raw_data = bundle.get("raw_data", {})
            vest_data = raw_data.get("vest", {})
            if vest_data:
                logging.info(f"VEST Data available: {list(vest_data.keys())}")

        # Keep plots open for inspection
        if analysis_args.plot or analysis_args.overview_plot:
            logging.info("Displaying plots. Close plot windows to end script.")
            plt.show()

    logging.info("Script finished. You can now explore the variables in your IDE.")
