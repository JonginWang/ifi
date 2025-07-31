"""
IFI Analysis Main Script
========================

This script orchestrates the data analysis workflow for the IFI project.
It handles data loading, processing, analysis, and visualization.
"""

import argparse
import logging
import os
import re
from typing import List, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask
from dask import delayed


# Set a project-local cache directory for numba to avoid permission errors.
# This directory should be added to .gitignore
numba_cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'cache', 'numba_cache')
os.makedirs(numba_cache_dir, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = numba_cache_dir


from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB
from ifi.analysis import processing, plots, spectrum, phi2ne
from ifi.analysis.plots import plot_cwt
from ifi.utils import LogManager, FlatShotList
from ifi import file_io

# The LogManager class ensures this setup runs only once per session.
# It's better to initialize it inside the main() function or at the start of the script logic.
# LogManager(level="INFO") # This call is redundant if called in main()

@dask.delayed
def load_and_process_file(nas_instance, file_path, args):
    """
    Loads a single file using the NAS_DB instance, then processes it
    through refining, offset removal, and STFT analysis.
    This function is designed to be run in parallel by Dask.
    """
    logging.info(f"Starting processing for: {os.path.basename(file_path)}")
    
    # 1. Read single file data
    # We call the internal _read_shot_file method directly.
    df_raw = nas_instance._read_shot_file(file_path)
    if df_raw is None:
        logging.warning(f"Failed to read {file_path}. Skipping.")
        return None, None, None, None # Return None for all results

    # 2. Refine data
    df_refined = processing.refine_data(df_raw)

    # 3. Remove offset
    if not args.no_offset_removal:
        df_processed = processing.remove_offset(df_refined, window_size=args.offset_window)
        logging.info(f"Offset removed from {file_path}")
    else:
        df_processed = df_refined
    
    # 4. Perform STFT analysis if requested
    stft_result = None
    if args.stft:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != 'TIME']
        
        cols_to_analyze_indices = range(len(all_data_cols))
        if args.ft_cols is not None:
            cols_to_analyze_indices = args.ft_cols

        fs = 1 / df_processed['TIME'].diff().mean()
        
        stft_result_for_file = {}
        for col_idx in cols_to_analyze_indices:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            f, t, Zxx = analyzer.compute_stft(signal, fs)
            stft_result_for_file[col_name] = {'f': f, 't': t, 'Zxx': Zxx}
         
        stft_result = {file_path: stft_result_for_file}
        logging.info(f"STFT analysis complete for {file_path}")

    # 5. Perform CWT analysis if requested
    cwt_result = None
    if args.cwt:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != 'TIME']
        
        cols_to_analyze_indices = range(len(all_data_cols))
        if args.ft_cols is not None:
            cols_to_analyze_indices = args.ft_cols

        fs = 1 / df_processed['TIME'].diff().mean()
        
        cwt_result_for_file = {}
        for col_idx in cols_to_analyze_indices:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            freqs, cwt_matrix= analyzer.compute_cwt(signal, fs)
            cwt_result_for_file[col_name] = {"t": df_processed.index.to_numpy(), "freqs": freqs, "cwt_matrix": cwt_matrix}
         
        cwt_result = {file_path: cwt_result_for_file}
        logging.info(f"CWT analysis complete for {file_path}")

    # Return a tuple of the processed data and any analysis results
    return file_path, df_processed, stft_result, cwt_result


def run_analysis(
    query: Union[int, str, List[Union[int, str]]],
    args: argparse.Namespace,
    nas_db: NAS_DB,
    vest_db: VEST_DB,
):
    """
    Performs a complete analysis workflow for a given query.
    """
    logging.info("\n========== Parsing Analysis Query ==========")
    flat_list = FlatShotList(query)
    logging.info(f"Found {len(flat_list.nums)} unique shot numbers: {flat_list.nums}")
    logging.info(f"Found {len(flat_list.paths)} unique file paths: {flat_list.paths}")

    if not flat_list.all:
        logging.warning("Query resulted in an empty list of targets. Nothing to do.")
        return

    # --- 1. Find and Load Data ---
    target_files = nas_db.find_files(
        query=flat_list.all,
        data_folders=args.data_folders,
        add_path=args.add_path,
        force_remote=args.force_remote
    )
    if not target_files:
        logging.warning(f"No files found for the given query. Skipping.")
        return

    # Group files by shot number
    files_by_shot = defaultdict(list)
    for f in target_files:
        match = re.search(r'(\d{5,})', os.path.basename(f))
        if match:
            files_by_shot[int(match.group(1))].append(f)
        else:
            files_by_shot['unknown'].append(f)
            
    logging.info(f"Grouped files into {len(files_by_shot)} shot(s).")

    # Load VEST data for all relevant shots
    vest_data_by_shot = defaultdict(dict)
    if flat_list.nums:
        logging.info(f"Loading VEST data for shots: {flat_list.nums}")
        for shot_num in flat_list.nums:
            vest_data_by_shot[shot_num] = vest_db.load_shot(shot=shot_num, fields=args.vest_fields) if shot_num > 0 else {}

    # --- Dask Task Creation ---
    tasks = [dask.delayed(load_and_process_file)(nas_db, f, args) for f in target_files]
    
    logging.info(f"Starting Dask computation for {len(tasks)} tasks...")
    results = dask.compute(*tasks, scheduler=args.scheduler)
    logging.info("Dask computation finished.")

    # --- Process Dask Results ---
    analysis_data = defaultdict(dict)
    stft_results = defaultdict(dict)
    cwt_results = defaultdict(dict)
    for file_path, df, stft_res, cwt_res in results:
        if df is None:
            continue
        
        match = re.search(r'(\d{5,})', os.path.basename(file_path))
        shot_num = int(match.group(1)) if match else 0 # Group under 0 if no shot number
        
        analysis_data[shot_num][os.path.basename(file_path)] = df
        if stft_res:
            stft_results[shot_num].update(stft_res)
        if cwt_res:
            cwt_results[shot_num].update(cwt_res)

    # --- 2. Process Each Shot ---
    all_analysis_bundles = {}
    shots_to_process = sorted(analysis_data.keys())

    for shot_num in shots_to_process:
        shot_nas_data = analysis_data[shot_num]
        shot_stft_data = stft_results.get(shot_num, {})
        shot_cwt_data = cwt_results.get(shot_num, {})
        
        logging.info(f"\n========== Post-processing for Shot #{shot_num} with {len(shot_nas_data)} files ==========")
        
        # --- Combine and Prepare Data ---
        ref_time_axis = list(shot_nas_data.values())[0].index
        combined_signals = pd.concat(
            [df.reindex(ref_time_axis, method="nearest", limit=1) for df in shot_nas_data.values()],
            axis=1
        )
        current_vest_data = vest_data_by_shot.get(shot_num, {})
        vest_lf_data = current_vest_data.get('25k') or current_vest_data.get('250k')
        
        # --- 5. Density Calculation & Baseline Correction ---
        phase_converter = phi2ne.PhaseConverter()
        density_data = pd.DataFrame(index=combined_signals.index)
        if args.density:
            params = file_io.get_density_analysis_params(f"{shot_num}_ALL.csv")
            if params:
                ref_signal = combined_signals[params['ref_col']].dropna().to_numpy()
                fs = 1 / combined_signals.index.to_series().diff().mean()
                f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
                for probe_col in params['probe_cols']:
                    if probe_col in combined_signals.columns:
                        phase = phase_converter.calc_phase_cdm(ref_signal, combined_signals[probe_col].to_numpy(), fs, f_center)
                        density_data[f"ne_{probe_col}"] = phase_converter.phase_to_density(phase, params['freq'])
                if args.baseline and vest_lf_data is not None:
                    density_data = phase_converter.correct_baseline(density_data, vest_lf_data, mode=args.baseline, trigger_time=args.trigger_time)
        
        # --- 6. Final Output Bundle ---
        analysis_bundle = {
            'shot_info': {'shot_num': shot_num},
            'raw_data': {'nas': shot_nas_data, 'vest': current_vest_data},
            'processed_data': {'signals': combined_signals, 'density': density_data},
            'analysis_results': {'stft': shot_stft_data, 'cwt': shot_cwt_data}
        }
        all_analysis_bundles[shot_num] = analysis_bundle

        # --- 7. Plotting and Saving ---
        if args.plot or args.save_plots:
            logging.info("Generating plots...")
            title_prefix = f"Shot #{shot_num} - " if shot_num else ""
            
            # Use a context manager to handle plot creation and showing/saving
            with plots.ifi_plotting(interactive=args.plot, save_dir=os.path.join(args.results_dir, str(shot_num)) if args.save_plots else None, save_prefix=title_prefix):
                if not args.no_plot_ft:
                    if shot_stft_data:
                        plots.plot_spectrograms(shot_stft_data, title_prefix=title_prefix, trigger_time=args.trigger_time, downsample=args.downsample)
                    if shot_cwt_data:
                        plots.plot_cwt(shot_cwt_data, trigger_time=args.trigger_time, title_prefix=title_prefix)
                
                if not args.no_plot_raw:
                    plots.plot_analysis_overview(
                        shot_num,
                        {"Processed Signals": combined_signals},
                        {"Density": density_data},
                        vest_lf_data,
                        trigger_time=args.trigger_time,
                        title_prefix=title_prefix,
                        downsample=args.downsample
                    )

        # --- Saving Logic Update ---
        if args.save_data:
            # The logic inside save_results_to_hdf5 now handles shot_num=0 correctly
            output_dir = os.path.join(args.results_dir, str(shot_num) if shot_num != 0 else 'unknown_shots')
            file_io.save_results_to_hdf5(
                output_dir, shot_num, combined_signals, shot_stft_data,
                shot_cwt_data, density_data, vest_lf_data
            )

    logging.info("\n========== Full Analysis Finished ==========")
    return all_analysis_bundles


def main():
    """
    Main entry point for the IFI analysis script.
    Parses command-line arguments and orchestrates the analysis process.
    """
    LogManager(level="INFO")
    
    parser = argparse.ArgumentParser(
        description='IFI Analysis Program',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Argument Parsing ---
    parser.add_argument(
        'query',
        nargs='+',
        help='One or more analysis targets, which can be:\n'
             '- Shot numbers (e.g., 45821)\n'
             '- A range of shots (e.g., "45821:45823")\n'
             '- A glob pattern for shots (e.g., "45*")\n'
             '- Full paths to specific data files.'
    )
    parser.add_argument(
        '--data_folders',
        type=str,
        default=None,
        help='Comma-separated list of data folders to search. Overrides config.ini defaults.'
    )
    parser.add_argument(
        '--add_path',
        action='store_true',
        help='''If specified, adds the --data_folders paths to the default paths 
            from config.ini instead of overriding them.'''
    )
    parser.add_argument(
        '--force_remote',
        action='store_true',
        help='Force fetching data from the remote NAS, ignoring any local cache.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='ifi/results',
        help='The directory to store analysis results and cached data.'
    )

    # Flags for data processing
    parser.add_argument(
        '--no_offset_removal',
        action='store_true',
        help='Disable the moving average offset removal step.'
    )
    parser.add_argument(
        '--offset_window',
        type=int,
        default=2001,
        help='The window size for the moving average offset removal.'
    )

    # Flags for data processing with freq.-time transform
    parser.add_argument(
        '--stft',
        action='store_true',
        help='Perform STFT analysis when retrieving data.'
    )
    parser.add_argument(
        '--cwt',
        action='store_true',
        help='Perform CWT analysis when retrieving data.'
    )
    parser.add_argument(
        '--ft_cols',
        nargs='*',
        type=int,
        default=[],
        help='''The indices of the columns to perform freq.-time transform analysis on.
            If empty, all columns will be analyzed.'''
    )

    # Flags for plotting
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show plots of the analysis results.'
    )
    parser.add_argument(
        '--no_plot_raw',
        action='store_true',
        help='''Don't show the plots of the raw data.'''
    )
    parser.add_argument(
        '--no_plot_ft',
        action='store_true',
        help='''Don't show the plots of the time-frequency trasnforms.'''
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=10,
        help='Downsample factor for plotting to improve performance. Default: 10.'
    )
    parser.add_argument(                        
        '--trigger_time',
        type=float,
        default=0.290,
        help='Trigger time in seconds to be added to the time axis for plotting.'
    )

    # Analysis-specific flags
    parser.add_argument(
        '--density',
        action='store_true',
        help='Perform phase and density calculation.'
    )
    parser.add_argument(
        '--vest_fields',
        nargs='*',
        type=int,
        default=[],
        help='Space-separated list of VEST DB field IDs to load and process.'
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=['ip', 'trig'],
        default=None,
        help="Perform baseline correction on density data."
    )
    
    # Output flags
    parser.add_argument(
        "--save_plots", 
        action="store_true", 
        help="Save generated plots to files."
    )
    parser.add_argument(
        "--save_data", 
        action="store_true", 
        help="Save processed data to HDF5 files."
    )

    # Dask scheduler

    parser.add_argument(
        "--scheduler",
        type=str,
        default='threads',
        choices=['threads', 'processes', 'single-threaded'],
        help="Dask scheduler to use for parallel processing."
    )
    
    args = parser.parse_args()

    # --- Initialize DB Controllers ---
    try:
        nas_db = NAS_DB(config_path='ifi/config.ini')
        vest_db = VEST_DB(config_path='ifi/config.ini')
    except FileNotFoundError:
        logging.error("Configuration file 'ifi/config.ini' not found. Exiting.")
        return
    except Exception as e:
        logging.error(f"Failed to initialize database controllers: {e}")
        return

    # The main change is here: call run_analysis with the raw query
    run_analysis(args.query, args, nas_db, vest_db)


if __name__ == '__main__':
    main() 