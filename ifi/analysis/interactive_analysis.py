#!/usr/bin/env python3
"""
    Interactive Analysis
    ===================

    This Script is a Interactive analysis pipeline for IFI package.
    It is used to test the analysis pipeline and the results.

    Functions:
        create_mock_args: Creates a mock argparse.Namespace object for interactive analysis.
        main: Runs the analysis pipeline.
            - Creates a mock argparse.Namespace object for interactive analysis.
            - Runs the analysis pipeline.
            - Options:
                - query: Shot number or pattern
                - stft: Whether to run the STFT analysis.
                - density: Whether to run the density analysis.
                - plot: Whether to plot the results.
                - overview_plot: Whether to plot the overview.
                - data_folders: Data folders to use.
                - add_path: Whether to add the path to the data folders.
                - force_remote: Whether to force the remote data.

"""

import logging
from argparse import Namespace
import matplotlib.pyplot as plt

from ifi.analysis.main_analysis import run_analysis
from ifi.utils.common import LogManager

LogManager(level="DEBUG")


def create_mock_args():
    """
    Creates a mock argparse.Namespace object for interactive analysis.
    This simulates the command-line arguments.
    """
    args = Namespace(
        # --- Essential Arguments ---
        query=['45821'],  # Shot number or pattern
        stft=True,
        density=True,
        plot=True,
        overview_plot=True,

        # --- Data Source Arguments ---
        data_folders=None,
        add_path=False,
        force_remote=False,
        vest_fields=[109, 101],

        # --- Processing Arguments ---
        no_offset_removal=False,
        offset_window=2001,
        baseline='ip',  # 'ip', 'trig', or None
        
        # --- Plotting Arguments ---
        plot_raw=False,
        trigger_time=0.290,
        downsample=10,
        
        # --- STFT/CWT Arguments ---
        stft_cols=None, # e.g., [0, 1] or None for all
        cwt=False,
        cwt_cols=None,

        # --- Saving Arguments ---
        results_dir='ifi/results',
        save_plots=False,
        save_data=False,
        
        # --- Performance Arguments ---
        scheduler='single-threaded', # Use 'single-threaded' for easier debugging
    )
    return args


if __name__ == '__main__':
    # 1. Set up analysis parameters
    # You can modify these arguments directly for your specific test case.
    analysis_args = create_mock_args()

    # To test a different shot, for example:
    # analysis_args.query = ['44149']
    # analysis_args.stft = True
    # analysis_args.density = True
    # analysis_args.vest_fields = []
    # analysis_args.overview_plot = False
    
    # 2. Run the analysis pipeline
    # The results are returned in a dictionary.
    logging.info("Starting interactive analysis...")
    results = run_analysis(analysis_args)
    logging.info("Analysis finished.")

    # 3. Access and explore the results
    # The 'results' dictionary contains all the major data artifacts.
    # You can now inspect these variables in Spyder's Variable Explorer.
    if results:
        processed_data = results.get("processed_data")
        stft_results = results.get("stft_results")
        cwt_results = results.get("cwt_results")
        density_data = results.get("density_data")
        vest_data = results.get("vest_data")

        logging.info("--- Available Data ---")
        if processed_data is not None:
            logging.info(f"Processed Data Shape: {processed_data.shape}")
        if stft_results:
            logging.info(f"STFT Results available for: {list(stft_results.keys())}")
        if density_data is not None and not density_data.empty:
            logging.info(f"Density Data Shape: {density_data.shape}")
        if vest_data is not None and not vest_data.empty:
            logging.info(f"VEST Data Shape: {vest_data.shape}")

        # Keep plots open for inspection
        if analysis_args.plot or analysis_args.overview_plot:
            logging.info("Displaying plots. Close plot windows to end script.")
            plt.show()

    logging.info("Script finished. You can now explore the variables in your IDE.") 