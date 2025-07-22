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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        return None

    # 2. Refine data
    df_refined = processing.refine_data(df_raw)

    # 3. Remove offset
    if not args.no_offset_removal:
        df_processed = processing.remove_offset(df_refined, window_size=args.offset_window)
    else:
        df_processed = df_refined
    
    # 4. Perform STFT analysis if requested
    stft_result = None
    if args.stft:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != 'TIME']
        
        cols_to_analyze_indices = range(len(all_data_cols))
        if args.stft_cols is not None:
            cols_to_analyze_indices = args.stft_cols

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

    # Return a tuple of the processed data and any analysis results
    return file_path, df_processed, stft_result


def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(description="IFI Analysis Pipeline")
    
    # --- Argument Parsing ---
    parser.add_argument(
        'query',
        nargs='+',
        help='The analysis target. Can be one or more shot numbers (e.g., 45821), patterns ("45*"), or full file paths.'
    )
    parser.add_argument(
        '--data_folders',
        type=str,
        default=None,
        help='Comma-separated list of data folders to search. Overrides config.ini defaults unless --add_path is used.'
    )
    parser.add_argument(
        '--add_path',
        action='store_true',
        help='If specified, adds the --data_folders paths to the default paths from config.ini instead of overriding them.'
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
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show plots of the analysis results.'
    )
    parser.add_argument(
        '--plot_raw',
        action='store_true',
        help='In addition to the main analysis plot, also show a plot of the data before offset removal.'
    )
    parser.add_argument(
        '--trigger_time',
        type=float,
        default=0.290,
        help='Trigger time in seconds to be added to the time axis for plotting.'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Downsample factor for plotting to improve performance.'
    )
    parser.add_argument(
        '--stft',
        action='store_true',
        help='Perform Short-Time Fourier Transform (STFT) analysis.'
    )
    parser.add_argument(
        '--stft_cols',
        nargs='*',
        type=int,
        default=None,
        help='Space-separated list of column indices to perform STFT on. If not provided, all columns are used.'
    )
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
        "--overview_plot",
        action="store_true",
        help="Generate a comprehensive overview plot.",
    )
    parser.add_argument(
        "--save_plots", action="store_true", help="Save generated plots to files."
    )
    # TODO: Add --cwt flag and logic when CWT analysis is implemented
    parser.add_argument(
        "--save_data", action="store_true", help="Save processed data to HDF5 files."
    )
    parser.add_argument(
        "--cwt", action="store_true", help="Perform Continuous Wavelet Transform (CWT) analysis."
    )
    parser.add_argument(
        "--cwt_cols",
        nargs="+",
        default=None,
        help="Specific columns to use for CWT. Defaults to all processed columns.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=['ip', 'trig'],
        default=None,
        help="Perform baseline correction on density data. Modes: 'ip' (uses plasma current ramp-up), 'trig' (uses fixed time window)."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default='threads',
        choices=['threads', 'processes', 'single-threaded'],
        help="Dask scheduler to use for parallel processing."
    )

    args = parser.parse_args()

    # --- Process Arguments ---
    # Convert single query item to correct type if possible
    processed_query: Union[str, int, List[Union[str, int]]]
    if len(args.query) == 1:
        item = args.query[0]
        processed_query = int(item) if item.isdigit() else item
    else:
        processed_query = [int(q) if q.isdigit() else q for q in args.query]
    
    data_folders_list = None
    if args.data_folders:
        data_folders_list = [folder.strip() for folder in args.data_folders.split(',')]

    logging.info("Starting IFI Analysis...")
    logging.info(f"Query: {processed_query}, Folders: {data_folders_list or 'Default'}, Results Dir: {args.results_dir}")

    # --- Main Analysis Steps ---
    try:
        with NAS_DB(config_path='ifi/config.ini') as nas:
            # Override the default dumping_folder with the one from arguments
            nas.dumping_folder = args.results_dir
            os.makedirs(nas.dumping_folder, exist_ok=True)

            # Determine the final list of folders to search
            search_folders = None
            if args.data_folders:
                user_folders = [folder.strip() for folder in args.data_folders.split(',')]
                if args.add_path:
                    # Combine user-specified folders with defaults from the config
                    search_folders = nas.default_data_folders + user_folders
                    logging.info(f"Adding to default search paths. Searching {len(search_folders)} folders.")
                else:
                    # Override the defaults with user-specified folders
                    search_folders = user_folders
            # If search_folders is still None, nas.find_files will use its defaults

            logging.info("Finding files...")
            # Instead of loading all data, just find the file paths first.
            target_files = nas.find_files(query=processed_query, data_folders=search_folders)

            if not target_files:
                logging.warning("No files found for the given query. Exiting.")
                return

            logging.info(f"Found {len(target_files)} files to process. Building Dask graph...")
            
            # --- Dask Parallel Processing ---
            # 1. Create a list of delayed tasks
            tasks = [load_and_process_file(nas, f, args) for f in target_files]
            
            # 2. Execute tasks in parallel
            logging.info(f"Executing {len(tasks)} tasks in parallel using '{args.scheduler}' scheduler...")
            # For debugging, force single-threaded execution to see logs properly
            results = dask.compute(*tasks, scheduler='single-threaded')
            
            # 3. Process results
            # Filter out None results from failed tasks
            results = [res for res in results if res is not None]
            if not results:
                logging.error("All processing tasks failed. Exiting.")
                return
                
            # Unpack results into separate dictionaries
            analysis_data = {file_path: df for file_path, df, stft in results}
            stft_results = {k: v for stft in (res[2] for res in results if res[2] is not None) for k, v in stft.items()}

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading and processing: {e}", exc_info=True)
        return

    # --- Combine data from multiple files into single dataframes for easier processing ---
    # We will use the time axis from the first file as the reference
    all_filenames = list(analysis_data.keys())
    ref_time_axis = analysis_data[all_filenames[0]].index

    def combine_dataframes(data_dict):
        """Combines a dictionary of dataframes into a single dataframe."""
        df_list = []
        for filename, df in data_dict.items():
            # Rename columns to include filename prefix to avoid collision
            prefix = os.path.basename(filename).split(".")[0]
            df_renamed = df.rename(
                columns={c: f"{prefix}_{c}" for c in df.columns}
            )
            # Reindex to the common time axis
            df_reindexed = df_renamed.reindex(ref_time_axis, method="nearest", limit=1)
            df_list.append(df_reindexed)
        return pd.concat(df_list, axis=1)

    # Combined dataframes for plotting and further analysis
    # NOTE: The concept of "raw_data" and "refined_data" is simplified in the Dask workflow
    # as processing is done in one go. We will use the final 'analysis_data' for combinations.
    # If raw/refined plots are needed, the delayed function would need to return them.
    combined_analysis_data = combine_dataframes(analysis_data)


    # 4. Perform frequency analysis
    # STFT is now done in parallel within the Dask tasks.
    # We just need to ensure the results are correctly formatted.
    if args.stft:
        logging.info("STFT analysis was performed in parallel.")
    else:
        stft_results = {}


    # --- Perform CWT Analysis ---
    cwt_results = {}
    if args.cwt:
        logging.info("Performing CWT analysis...")
        analyzer = spectrum.SpectrumAnalysis()

        for filename, df in analysis_data.items():
            # Determine which columns to analyze
            all_data_cols = [col for col in df.columns]
            if args.cwt_cols is None:
                cols_to_analyze = all_data_cols
            else:
                cols_to_analyze = args.cwt_cols

            fs = 1 / df.index.to_series().diff().mean()
            cwt_results[filename] = {}

            for col_name in cols_to_analyze:
                if col_name not in df.columns:
                    logging.warning(f"Column '{col_name}' not found in {os.path.basename(filename)}. Skipping CWT.")
                    continue

                logging.info(f"  Analyzing {col_name} in {os.path.basename(filename)}...")
                signal = df[col_name].to_numpy()
                
                # Create a time vector corresponding to the signal
                t = df.index.to_numpy()

                cwt_matrix, freqs = analyzer.compute_cwt(signal, fs)
                cwt_results[filename][col_name] = {
                    "t": t,
                    "freqs": freqs,
                    "cwt_matrix": cwt_matrix,
                }
        logging.info("CWT analysis complete.")


    # 5. Calculate density
    density_dfs = []
    if args.density:
        if not stft_results:
            logging.error("Density calculation requires STFT results for center frequency. Please run with --stft.")
        else:
            logging.info("Calculating phase and density...")
            phase_converter = phi2ne.PhaseConverter()
            analyzer = spectrum.SpectrumAnalysis() # For finding ridge

            for filename, df in analysis_data.items():
                params = get_density_analysis_params(filename)
                if not params:
                    logging.warning(f"No density analysis recipe for {os.path.basename(filename)}. Skipping.")
                    continue

                fs = 1 / df['TIME'].diff().mean()
                
                ref_col_name = params['ref_col']
                if ref_col_name not in df.columns:
                    logging.warning(f"Reference column '{ref_col_name}' not found in {os.path.basename(filename)}. Skipping.")
                    continue
                ref_signal = df[ref_col_name].to_numpy()

                for probe_col_name in params['probe_cols']:
                    logging.info(f"  Processing {probe_col_name} in {os.path.basename(filename)} using {params['method']} method...")
                    
                    phase = None
                    # --- Dispatch to correct phase calculation method ---
                    if params['method'] == 'iq':
                        # For IQ, the probe_col_name is actually a tuple of (i_col, q_col)
                        if not isinstance(probe_col_name, tuple) or len(probe_col_name) != 2:
                            logging.warning(f"IQ method requires a tuple of (I, Q) column names. Got {probe_col_name}. Skipping.")
                            continue
                        
                        i_col, q_col = probe_col_name
                        if i_col not in df.columns or q_col not in df.columns:
                            logging.warning(f"I/Q columns '{i_col}' or '{q_col}' not found in {os.path.basename(filename)}. Skipping.")
                            continue
                        
                        i_signal = df[i_col].to_numpy()
                        q_signal = df[q_col].to_numpy()
                        phase = phase_converter.calc_phase_iq_asin2(i_signal, q_signal)

                    elif params['method'] == 'fpga':
                        # For FPGA, the probe_col_name is a tuple of (phase_col, amp_col)
                        if not isinstance(probe_col_name, tuple) or len(probe_col_name) != 2:
                            logging.warning(f"FPGA method requires a tuple of (phase, amp) column names. Got {probe_col_name}. Skipping.")
                            continue
                        
                        p_col, a_col = probe_col_name
                        if p_col not in df.columns or a_col not in df.columns or ref_col_name not in df.columns:
                            logging.warning(f"FPGA columns '{p_col}', '{a_col}', or ref '{ref_col_name}' not found. Skipping.")
                            continue
                        
                        ref_phase = df[ref_col_name].to_numpy()
                        probe_phase = df[p_col].to_numpy()
                        amp_signal = df[a_col].to_numpy()
                        time_signal = df['TIME'].to_numpy()
                        phase = phase_converter.calc_phase_fpga(ref_phase, probe_phase, time_signal, amp_signal)

                    elif params['method'] == 'cdm':
                        if probe_col_name not in df.columns:
                            logging.warning(f"Probe column '{probe_col_name}' not found in {os.path.basename(filename)}. Skipping.")
                            continue
                        probe_signal = df[probe_col_name].to_numpy()

                        # Get f_center from STFT results of the reference channel
                        if filename not in stft_results or ref_col_name not in stft_results[filename]:
                            logging.warning(f"STFT results for reference '{ref_col_name}' not found in {os.path.basename(filename)}. Cannot perform CDM. Skipping.")
                            continue
                        
                        stft_data = stft_results[filename][ref_col_name]
                        ridge = analyzer.find_freq_ridge(stft_data['Zxx'], stft_data['f'])
                        f_center = np.nanmean(ridge)
                        logging.info(f"    - Determined center frequency: {f_center/1e6:.2f} MHz")

                        phase = phase_converter.calc_phase_cdm(ref_signal, probe_signal, fs, f_center)

                    if phase is not None:
                        density = phase_converter.phase_to_density(phase, params['freq'])
                        # For IQ, the key is a string representation of the tuple
                        result_key = str(probe_col_name) if isinstance(probe_col_name, tuple) else probe_col_name
                        prefix = os.path.basename(filename).split('.')[0]
                        
                        density_df = pd.DataFrame({
                            f"{prefix}_{result_key}_phase": phase,
                            f"{prefix}_{result_key}_density": density
                        }, index=df.index)
                        density_dfs.append(density_df)

                        logging.info(f"    - Successfully calculated density for {result_key}.")
    
    combined_density_data = pd.DataFrame(index=ref_time_axis)
    if density_dfs:
        # Reindex each dataframe before concatenating
        reindexed_dfs = [df.reindex(ref_time_axis, method='nearest', limit=1) for df in density_dfs]
        combined_density_data = pd.concat(reindexed_dfs, axis=1)

        # --- Baseline Correction ---
        if args.baseline:
            logging.info(f"Performing '{args.baseline}' baseline correction on density data...")
            # The time axis for density is the same as the analysis_data index
            time_axis_for_density = combined_analysis_data.index.to_numpy()
            combined_density_data = phase_converter.correct_baseline(
                combined_density_data,
                time_axis_for_density,
                args.baseline,
                vest_data=vest_data
            )
    else:
        combined_density_data = pd.DataFrame()

    # 6. Load VEST data
    vest_data = pd.DataFrame(index=ref_time_axis)
    shot_num_for_vest = None
    if args.vest_fields:
        # We need a shot number to load VEST data. We'll use the first one from the query.
        if isinstance(processed_query, int):
            shot_num_for_vest = processed_query
        elif isinstance(processed_query, list) and isinstance(processed_query[0], int):
            shot_num_for_vest = processed_query[0]
        else:
            # Try to extract from string pattern
            match = re.match(r'(\d+)', str(processed_query))
            if match:
                shot_num_for_vest = int(match.group(1))

        if shot_num_for_vest:
            logging.info(f"Loading VEST data for shot {shot_num_for_vest}...")
            try:
                vest_db = VEST_DB(config_path='ifi/config.ini')
                vest_df_raw = vest_db.load_shot(shot_num_for_vest, args.vest_fields)
                
                if not vest_df_raw.empty:
                    # Interpolate VEST data onto the main time axis
                    interp_data = {}
                    for col in vest_df_raw.columns:
                        interp_data[col] = np.interp(
                            ref_time_axis.to_numpy(),
                            vest_df_raw.index.to_numpy(),
                            vest_df_raw[col].to_numpy(),
                            left=np.nan,
                            right=np.nan,
                        )
                    vest_data = pd.DataFrame(interp_data, index=ref_time_axis)
                    logging.info("Successfully loaded and processed VEST data.")
                else:
                    logging.warning("No VEST data returned for the given fields.")
            except Exception as e:
                logging.error(f"An error occurred while loading VEST data: {e}", exc_info=True)
        else:
            logging.warning("Could not determine a shot number from query to load VEST data.")


    # --- Plotting ---
    if not args.plot and not args.overview_plot:
        logging.info("No plots requested. Skipping visualization.")
    else:
        logging.info("Generating plots...")
        title_prefix = f"Shot #{shot_num_for_vest} - " if shot_num_for_vest else ""
        plt.ion()

        # Basic signal plots
        if args.plot:
            plots.plot_signals(
                {"Raw": combined_analysis_data}, # Use combined_analysis_data for raw data
                trigger_time=args.trigger_time,
                downsample=args.downsample,
                title_prefix=title_prefix
            )

        # STFT plots
        if stft_results:
            plots.plot_spectrograms(
                stft_results,
                downsample=args.downsample,
                title_prefix=title_prefix
            )

        # CWT plots
        if cwt_results:
            plots.plot_cwt(
                cwt_results,
                trigger_time=args.trigger_time,
                title_prefix=title_prefix
            )

        # Overview plot
        if args.overview_plot:
            plots.plot_analysis_overview(
                shot_num_for_vest,
                {"Processed Signals": combined_analysis_data},
                {"Density": combined_density_data},
                vest_data,
                trigger_time=args.trigger_time,
                downsample=args.downsample,
                title_prefix=title_prefix,
            )

        if args.save_plots and shot_num_for_vest:
            output_dir = os.path.join(args.results_dir, str(shot_num_for_vest))
            os.makedirs(output_dir, exist_ok=True)
            for i in plt.get_fignums():
                fig = plt.figure(i)
                # Use suptitle as filename if available
                filename = fig._suptitle.get_text() if fig._suptitle else f"figure_{i}"
                # Sanitize filename
                filename = "".join(
                    c for c in filename if c.isalnum() or c in (" ", "_", "-")
                ).rstrip()
                filename = filename.replace(" ", "_").replace("#", "")
                filepath = os.path.join(output_dir, f"{filename}.png")
                logging.info(f"Saving figure to {filepath}")
                fig.savefig(filepath, dpi=300)
        
        if plt.get_fignums():
            logging.info("Displaying plots. Close all plot windows to exit.")
            plt.ioff()
            plt.show()
        else:
            logging.info("No plots were generated.")


    # 7. Save results
    if args.save_data and shot_num_for_vest:
        logging.info("Saving analysis results to HDF5 files...")
        output_dir = os.path.join(args.results_dir, str(shot_num_for_vest))
        os.makedirs(output_dir, exist_ok=True)
        save_results_to_hdf5(
            output_dir,
            shot_num_for_vest,
            combined_analysis_data,
            stft_results,
            cwt_results,
            combined_density_data,
            vest_data,
        )

    logging.info("IFI Analysis Finished.")


def save_results_to_hdf5(
    output_dir, shot_num, processed_data, stft_results, cwt_results, density_data, vest_data
):
    """Saves the analysis results into multiple HDF5 files."""

    # Save Processed/Analysis Data (subset of NAS cache)
    # This might be redundant if the cache is already in the right place,
    # but saving it ensures the specific analysis version is stored.
    try:
        processed_path = os.path.join(output_dir, f"{shot_num}_processed.h5")
        if not processed_data.empty:
            processed_data.to_hdf(
                processed_path, key="processed", mode="w", complevel=9
            )
            logging.info(f"Saved processed data to {processed_path}")
    except Exception as e:
        logging.error(f"Failed to save processed data: {e}")

    # Save Frequency Transform Data (STFT and CWT)
    try:
        ft_path = os.path.join(output_dir, f"{shot_num}_FT.h5")
        with pd.HDFStore(ft_path, mode="w", complevel=9) as store:
            if stft_results:
                for filename, analysis in stft_results.items():
                    file_key = os.path.basename(filename).replace(".", "_")
                    for col_name, results in analysis.items():
                        group_key = f"/stft/{file_key}/{col_name}"
                        store.put(f"{group_key}/f", pd.Series(results["f"]))
                        store.put(f"{group_key}/t", pd.Series(results["t"]))
                        store.put(
                            f"{group_key}/Zxx_abs",
                            pd.DataFrame(np.abs(results["Zxx"])),
                        )
            if cwt_results:
                for filename, analysis in cwt_results.items():
                    file_key = os.path.basename(filename).replace(".", "_")
                    for col_name, results in analysis.items():
                        group_key = f"/cwt/{file_key}/{col_name}"
                        store.put(f"{group_key}/t", pd.Series(results["t"]))
                        store.put(f"{group_key}/freqs", pd.Series(results["freqs"]))
                        store.put(
                            f"{group_key}/cwt_abs",
                            pd.DataFrame(np.abs(results["cwt_matrix"])),
                        )
        if stft_results or cwt_results:
            logging.info(f"Saved Frequency Transform data to {ft_path}")
    except Exception as e:
        logging.error(f"Failed to save Frequency Transform data: {e}")

    # Save Density Data
    try:
        density_path = os.path.join(output_dir, f"{shot_num}_LID.h5")
        if not density_data.empty:
            density_data.to_hdf(density_path, key="density", mode="w", complevel=9)
            logging.info(f"Saved density data to {density_path}")
    except Exception as e:
        logging.error(f"Failed to save density data: {e}")

    # Save VEST Data
    try:
        vest_path = os.path.join(output_dir, f"{shot_num}_shot.h5")
        if not vest_data.empty:
            vest_data.to_hdf(vest_path, key="vest", mode="w", complevel=9)
            logging.info(f"Saved VEST data to {vest_path}")
    except Exception as e:
        logging.error(f"Failed to save VEST data: {e}")


def get_density_analysis_params(filename: str) -> dict:
    """
    Determines the correct density analysis parameters based on filename and shot number.
    """
    basename = os.path.basename(filename)
    match = re.match(r'(\d+)', basename)
    if not match:
        return None
    
    shot_num = int(match.group(1))
    
    # Rule 3: Shots 41542 and above
    if shot_num >= 41542:
        if "_ALL" in basename:
            # {shot번호}_ALL 파일: 280GHz / CDM 분석
            # MATLAB script: ref282 = csv_data282(:,2); prob282 = csv_data282(:,3);
            # This maps to CH1 and CH2 if TIME is the first column.
            return {
                'method': 'cdm',
                'freq': 280,
                'ref_col': 'CH1',
                'probe_cols': ['CH2']
            }
        else:
            # {shot번호}_{채널숫자}.csv형식은 94GHz, {채널숫자}에서 0은 refrence, 5~9는 각각 5~9번 채널 /CDM 분석
            # NOTE: This rule implies that data from multiple files (e.g., _0.csv for ref, _5.csv for probe)
            # needs to be combined into a single DataFrame before this step. The current pipeline
            # processes files one by one. We are ASSUMING that a future step will combine the data,
            # or that a single file contains all necessary columns.
            # The MATLAB script suggests ref is col 2, ch5 is col 3, etc.
            # We map this to CH1, CH2, ... assuming TIME is col 1.
            # User said "CH0" is ref, which is ambiguous. Assuming CH1 is ref for now.
            return {
                'method': 'cdm',
                'freq': 94,
                'ref_col': 'CH1',
                'probe_cols': ['CH2', 'CH3', 'CH4', 'CH5', 'CH6'] # Mapping for channels 5-9
            }

    # Rule 2: Shots 39302–41398
    elif 39302 <= shot_num <= 41398:
        # 94GHz / 5~9번 다채널 / FPGA 분석
        # Assuming phase/amp pairs with a common reference phase channel.
        # Placeholder names, e.g., CH1_p, CH1_a.
        return {
            'method': 'fpga',
            'freq': 94,
            'ref_col': 'CH_ref_p', # Placeholder for reference phase channel
            'probe_cols': [
                ('CH5_p', 'CH5_a'),
                ('CH6_p', 'CH6_a'),
                ('CH7_p', 'CH7_a'),
                ('CH8_p', 'CH8_a'),
                ('CH9_p', 'CH9_a')
            ]
        }

    # Rule 1: Shots 0–39265
    elif 0 <= shot_num <= 39265:
        # 94GHz / 5번 단일채널 / IQ 분석
        # Assuming the two columns for I and Q are named 'CH1' and 'CH2' for simplicity.
        # This might need to be adjusted based on actual data file structure.
        return {
            'method': 'iq',
            'freq': 94,
            'ref_col': None, # IQ method does not use a separate reference signal from another channel
            'probe_cols': [('CH1', 'CH2')]
        }
        
    return None


if __name__ == '__main__':
    main() 