#!/usr/bin/env python3
"""
IFI Analysis Main Script
========================

This script orchestrates the data analysis workflow for the IFI project.
It handles data loading, processing, analysis, and visualization.

Functions:
    load_and_process_file:
        Loads a single file using the NAS_DB instance,
        then processes it through refining, offset removal, and STFT/CWT analysis.
    run_analysis: Performs a complete analysis workflow for a given query.
    main: Main function that parses command line arguments and runs the analysis workflow.

Example:
    python main_analysis.py 45821
    python main_analysis.py 45821 --stft --cwt
    python main_analysis.py 45821 --stft --cwt --scheduler "threads"
    python main_analysis.py 45821 --stft --cwt --scheduler "processes"
    python main_analysis.py 45821 --stft --cwt --scheduler "single-threaded"
    python main_analysis.py 45821 --stft --cwt --scheduler "threads" --data_folders "data1,data2"
    python main_analysis.py 45821 --stft --cwt --scheduler "threads" --data_folders "data1,data2" --add_path
"""

import logging
from pathlib import Path
import argparse
import re
import time
from typing import List, Union
from collections import defaultdict
import pandas as pd
import dask
try:
    from ..db_controller.nas_db import NAS_DB
    from ..db_controller.vest_db import VEST_DB
    from . import processing, plots, spectrum, phi2ne
    from ..utils.common import LogManager, FlatShotList, log_tag
    from ..utils import file_io
    from .phi2ne import get_interferometry_params
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.db_controller.nas_db import NAS_DB
    from ifi.db_controller.vest_db import VEST_DB
    from ifi.analysis import processing, plots, spectrum, phi2ne
    from ifi.utils.common import LogManager, FlatShotList, log_tag
    from ifi.utils import file_io
    from ifi.analysis.phi2ne import get_interferometry_params

# The LogManager is better to be initialized inside the main() function
# or at the start of the script logic.
# LogManager(level="INFO") # This call is redundant if called in main()


@dask.delayed
def load_and_process_file(nas_instance, file_path, args):
    """
    Loads a single file using the NAS_DB instance, then processes it
    through refining, offset removal, and STFT/CWT analysis.
    This function is designed to be run in parallel by Dask.

    Args:
        nas_instance (NAS_DB): NAS_DB instance
        file_path (str): Path to the file to process
        args (argparse.Namespace): argparse.Namespace object containing the analysis parameters

    Returns:
        tuple: (file_path, df_processed, stft_result, cwt_result)
            file_path (str): Path to the file that was processed
            df_processed (pd.DataFrame): DataFrame containing the processed data
            stft_result (dict): Dictionary containing the STFT results
            cwt_result (dict): Dictionary containing the CWT results
    Raises:
        See log file for more details.
    """
    # Extract basename for consistent key usage
    # This avoids issues with network paths (e.g., //147.47.31.91/vest/...)
    # by using only the filename as the dictionary key
    file_basename = Path(file_path).name
    logging.info(f"{log_tag('ANALY','LOAD')} Starting processing for: {file_basename}")

    # 1. Read single file data
    # Use basename instead of full path to avoid network path issues
    # get_shot_data will use basename as the dictionary key
    data_dict = nas_instance.get_shot_data(file_basename, force_remote=args.force_remote)
    
    # Check with basename (consistent with get_shot_data return keys)
    if not data_dict or file_basename not in data_dict:
        logging.warning(
            "\n"
            + "=" * 80
            + "\n"
            + f"  Failed to read {file_basename}. Skipping.  ".center(80, "!")
            + "\n"
            + "=" * 80
            + "\n"
        )
        return None, None, None, None  # Return None for all results

    # Use basename to get data (consistent key)
    df_raw = data_dict[file_basename]

    # 2. Refine data (Removing nan values)
    df_refined = processing.refine_data(df_raw)
    logging.info("\n" + f"  Data refined for {file_path}  ".center(80, "=") + "\n")

    # 3. Remove offset
    if not args.no_offset_removal:
        df_processed = processing.remove_offset(
            df_refined, window_size=args.offset_window
        )
        logging.info(f"{log_tag('ANALY','LOAD')} Offset removed from {file_path}")
    else:
        df_processed = df_refined

    # 4. Perform STFT analysis if requested
    stft_result = None
    if args.stft:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != "TIME"]

        cols_to_analyze_idxs = range(len(all_data_cols))
        if args.stft_cols is not None:
            stft_cols_idxs = set(cols_to_analyze_idxs).intersection(set(args.stft_cols))
            if len(stft_cols_idxs) > 0:
                logging.info(f"{log_tag('ANALY','LOAD')} Analyzing columns: {stft_cols_idxs} vs input columns: {args.stft_cols}")
                cols_to_analyze_idxs = list(stft_cols_idxs)
            else:
                logging.warning(f"{log_tag('ANALY','LOAD')} No columns to analyze for STFT. Skipping STFT analysis.")
                cols_to_analyze_idxs = []

        fs = 1 / df_processed["TIME"].diff().mean()

        stft_result_for_file = {}
        for col_idx in cols_to_analyze_idxs:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            freq_STFT, time_STFT, STFT_matrix = analyzer.compute_stft(signal, fs)
            stft_result_for_file[col_name] = {
                "time_STFT": time_STFT,
                "freq_STFT": freq_STFT,
                "STFT_matrix": STFT_matrix,
            }

        stft_result = {file_path: stft_result_for_file}
        logging.info(f"{log_tag('ANALY','LOAD')} STFT analysis complete for {file_path}")

    # 5. Perform CWT analysis if requested
    cwt_result = None
    if args.cwt:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != "TIME"]

        cols_to_analyze_idxs = range(len(all_data_cols))
        if args.cwt_cols is not None:
            cwt_cols_idxs = set(cols_to_analyze_idxs).intersection(set(args.cwt_cols))
            if len(cwt_cols_idxs) > 0:
                logging.info(f"{log_tag('ANALY','LOAD')} Analyzing columns: {cwt_cols_idxs} vs input columns: {args.cwt_cols}")
                cols_to_analyze_idxs = list(cwt_cols_idxs)
            else:
                logging.warning(f"{log_tag('ANALY','LOAD')} No columns to analyze for CWT. Skipping CWT analysis.")
                cols_to_analyze_idxs = []

        fs = 1 / df_processed["TIME"].diff().mean()

        cwt_result_for_file = {}
        for col_idx in cols_to_analyze_idxs:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            
            # Find center frequency for memory optimization
            f_center = analyzer.find_center_frequency_fft(signal, fs)
            
            # Memory optimization parameters
            # Use decimation if signal is very long (> 100k samples)
            decimation_factor = 1
            if len(signal) > 100000:
                # Calculate decimation factor to reduce to ~50k samples
                decimation_factor = max(1, len(signal) // 10000)
                logging.info(
                    f"{log_tag('ANALY','LOAD')} Large signal detected ({len(signal)} samples). "
                    f"Using decimation factor {decimation_factor} for CWT."
                )
            
            # Compute CWT with memory optimization
            if f_center > 0:
                # Use f_center mode: only compute ±10% around center frequency
                freq_CWT, CWT_matrix = analyzer.compute_cwt(
                    signal, 
                    fs, 
                    f_center=f_center,
                    f_deviation=0.1,  # ±10% deviation
                    decimation_factor=decimation_factor
                )
            else:
                # Fallback: use default CWT if center frequency detection failed
                logging.warning(
                    f"{log_tag('ANALY','LOAD')} Center frequency detection failed for {col_name}. "
                    f"Using default CWT (may use more memory)."
                )
                freq_CWT, CWT_matrix = analyzer.compute_cwt(
                    signal, 
                    fs,
                    decimation_factor=decimation_factor
                )
            
            # Adjust time axis if decimation was applied
            if decimation_factor > 1:
                time_CWT = df_processed["TIME"].iloc[::decimation_factor].values
            else:
                time_CWT = df_processed["TIME"].values
            
            cwt_result_for_file[col_name] = {
                "time_CWT": time_CWT,
                "freq_CWT": freq_CWT,
                "CWT_matrix": CWT_matrix,
            }

        cwt_result = {file_path: cwt_result_for_file}
        logging.info(f"{log_tag('ANALY','LOAD')} CWT analysis complete for {file_path}")

    # Return a tuple of the processed data and any analysis results
    return file_path, df_processed, stft_result, cwt_result


def run_analysis(
    query: Union[int, str, List[Union[int, str]]],
    args: argparse.Namespace,
    nas_db: NAS_DB,
    vest_db: VEST_DB,
) -> dict | None:
    """
    Performs a complete analysis workflow for a given query.

    Args:
        query (Union[int, str, List[Union[int, str]]]): Shot number or pattern to analyze
        args (argparse.Namespace): argparse.Namespace object containing the analysis parameters
        nas_db (NAS_DB): NAS_DB instance
        vest_db (VEST_DB): VEST_DB instance

    Returns:
        dict | None: Dictionary containing the analysis data or None if no analysis was performed

    Raises:
        See log file for more details.
    """
    logging.info(f"{log_tag('ANALY','RUN')} Parsing Analysis Query")
    flat_list = FlatShotList(query)
    logging.info(
        f"{log_tag('ANALY','RUN')} Found {len(flat_list.nums)} unique shot numbers: {flat_list.nums}"
    )
    logging.info(
        f"{log_tag('ANALY','RUN')} Found {len(flat_list.paths)} unique file paths: {flat_list.paths}"
    )

    if not flat_list.all:
        logging.warning(
            "\n"
            + f"{log_tag('ANALY','RUN')} Query resulted in an empty list of targets. Nothing to do."
            + "\n"
        )
        return

    # --- 1. Find and Load Data ---
    target_files = nas_db.find_files(
        query=flat_list.all,
        data_folders=args.data_folders,
        add_path=args.add_path,
        force_remote=args.force_remote,
    )
    if not target_files:
        logging.warning(
            "\n"
            + f"{log_tag('ANALY','RUN')} No files found for the given query. Skipping."
            + "\n"
        )
        return

    # Group files by shot number and determine interferometry parameters
    files_by_shot = defaultdict(list)
    interferometry_params_by_file = {}
    for f in target_files:
        match = re.search(r"(\d{5,})", Path(f).name)
        if match:
            shot_num = int(match.group(1))
            files_by_shot[shot_num].append(f)
            # Get parameters for each individual file (not just shot number)
            params = get_interferometry_params(shot_num, Path(f).name)
            interferometry_params_by_file[f] = params
            logging.info(
                f"{log_tag('ANALY','RUN')} Interferometry (Shot #{shot_num}) params for {Path(f).name}: "
            )
            logging.info(f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']} GHz")
        else:
            files_by_shot["unknown"].append(f)
            # For unknown files, use shot number 0 as default
            params = get_interferometry_params(0, Path(f).name)
            interferometry_params_by_file[f] = params
            logging.info(
                f"{log_tag('ANALY','RUN')} Interferometry (Shot #00000) params for {Path(f).name}: "
            )
            logging.info(f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']} GHz")

    logging.info(
        f"{log_tag('ANALY','RUN')} Grouped files into {len(files_by_shot)} shot(s)."
        + "\n"
    )

    # Load VEST data for all relevant shots
    vest_data_by_shot = defaultdict(dict)
    if flat_list.nums:
        logging.info(f"{log_tag('ANALY','RUN')} Loading VEST data for shots: {flat_list.nums}")
        for shot_num in flat_list.nums:
            vest_data_by_shot[shot_num] = (
                vest_db.load_shot(shot=shot_num, fields=args.vest_fields)
                if shot_num > 0
                else {}
            )

    # --- Dask Task Creation with Progress Tracking ---
    tasks = [dask.delayed(load_and_process_file)(nas_db, f, args) for f in target_files]

    logging.info(
        f"{log_tag('ANALY','RUN')} Starting Dask computation for {len(tasks)} tasks..."
        + "\n"
    )
    logging.info(f"{log_tag('ANALY','RUN')} Using scheduler: {args.scheduler}")
    logging.info(f"{log_tag('ANALY','RUN')} Target files: {len(target_files)}")

    # Execute with progress tracking and optimized scheduler
    start_time = time.time()

    # Use optimal scheduler based on task count and system resources
    if len(tasks) <= 4:
        scheduler = "threads"  # Better for I/O bound tasks
    else:
        scheduler = args.scheduler if args.scheduler else "threads"

    logging.info(f"{log_tag('ANALY','RUN')} Using scheduler: {scheduler}")
    results = dask.compute(*tasks, scheduler=scheduler)
    end_time = time.time()

    logging.info(f"{log_tag('ANALY','RUN')} Dask computation finished.")
    logging.info(f"{log_tag('ANALY','RUN')} Processing time: {end_time - start_time:.2f} seconds")
    logging.info(
        f"{log_tag('ANALY','RUN')} Average time per file: {(end_time - start_time) / len(target_files):.2f} seconds"
    )

    # --- Process Dask Results ---
    analysis_data = defaultdict(dict)
    stft_results = defaultdict(dict)
    cwt_results = defaultdict(dict)
    for file_path, df, stft_result, cwt_result in results:
        if df is None:
            continue

        match = re.search(r"(\d{5,})", Path(file_path).name)
        shot_num = (
            int(match.group(1)) if match else 0
        )  # Group under 0 if no shot number

        analysis_data[shot_num][Path(file_path).name] = df
        if stft_result:
            stft_results[shot_num].update(stft_result)
        if cwt_result:
            cwt_results[shot_num].update(cwt_result)

    # --- 2. Process Each Shot ---
    all_analysis_bundles = {}
    shots_to_process = sorted(analysis_data.keys())

    for shot_num in shots_to_process:
        shot_nas_data = analysis_data[shot_num]
        shot_stft_data = stft_results.get(shot_num, {})
        shot_cwt_data = cwt_results.get(shot_num, {})

        logging.info(
            "\n"
            + f"{log_tag('ANALY','RUN')} Post-processing for Shot #{shot_num} with {len(shot_nas_data)} files"
            +"\n"
        )

        # --- Collect interferometry parameters for this shot ---
        shot_files = files_by_shot.get(shot_num, [])
        shot_interferometry_params = {}
        for file_path in shot_files:
            basename = Path(file_path).name
            if file_path in interferometry_params_by_file:
                params = interferometry_params_by_file[file_path]
                shot_interferometry_params[basename] = params
                logging.info(f"{log_tag('ANALY','RUN')} File {basename}: ")
                logging.info(
                    f"{log_tag('ANALY','RUN')} {params['method']} method, {params['freq_ghz']}GHz, ref={params['ref_col']}, probes={params['probe_cols']}"
                )

        # --- Group Data by Frequency ---
        # Group files by frequency based on interferometry parameters
        # Use flexible frequency detection: map actual frequencies to standard groups
        # 93-95 GHz range → use 94GHz group
        # 275-285 GHz range → use 280GHz group
        freq_groups = {}
        for basename, params in shot_interferometry_params.items():
            freq_ghz = params.get("freq_ghz", 94.0)  # Actual frequency from config
            
            # Map to standard frequency group
            if 93.0 <= freq_ghz <= 95.0:
                # Map to 94GHz group
                group_freq = 94.0
            elif 275.0 <= freq_ghz <= 285.0:
                # Map to 280GHz group
                group_freq = 280.0
            else:
                # Use actual frequency as group key (for other frequencies)
                group_freq = freq_ghz
                logging.warning(
                    f"{log_tag('ANALY','RUN')} Frequency {freq_ghz} GHz outside standard ranges "
                    f"(93-95 or 275-285). Using actual frequency as group key."
                )
            
            if group_freq not in freq_groups:
                freq_groups[group_freq] = {"files": [], "params": []}
            freq_groups[group_freq]["files"].append(basename)
            freq_groups[group_freq]["params"].append(params)
            
            # Log frequency mapping
            if freq_ghz != group_freq:
                logging.info(
                    f"{log_tag('ANALY','RUN')} Mapped frequency {freq_ghz} GHz → {group_freq} GHz group for {basename}"
                )

        logging.info(
            f"{log_tag('ANALY','RUN')} Frequency groups found: {list(freq_groups.keys())} GHz"
        )

        # Create frequency-specific combined signals
        freq_combined_signals = {}
        for freq_ghz, group_info in freq_groups.items():
            files_in_group = group_info["files"]
            logging.info(
                f"{log_tag('ANALY','RUN')} Processing {freq_ghz}GHz group with files: {files_in_group}"
            )

            if freq_ghz == 280.0:
                # 280GHz: Use _ALL file independently
                all_file = None
                for basename in files_in_group:
                    if "_ALL" in basename and basename in shot_nas_data:
                        all_file = basename
                        break

                if all_file:
                    df = shot_nas_data[all_file].copy()
                    if "TIME" in df.columns:
                        df.index = df["TIME"]
                        df = df.drop("TIME", axis=1)
                        df.index.name = "TIME"
                    freq_combined_signals[freq_ghz] = df
                    logging.info(f"{log_tag('ANALY','RUN')} 280GHz: Using {all_file} with shape {df.shape}")
                else:
                    logging.warning(f"{log_tag('ANALY','RUN')} No _ALL file found for 280GHz group")

            elif freq_ghz == 94.0:
                # 94GHz: Use _0XX file as reference time axis, combine with other files
                ref_file = None
                other_files = []

                # Find reference file (_0XX pattern) and other files
                for basename in files_in_group:
                    if "_0" in basename and basename in shot_nas_data:
                        ref_file = basename
                    elif basename in shot_nas_data:
                        other_files.append(basename)

                if ref_file:
                    # Use reference file's time axis
                    ref_df = shot_nas_data[ref_file].copy()
                    if "TIME" in ref_df.columns:
                        ref_time_axis = ref_df["TIME"]
                        ref_df.index = ref_time_axis
                        ref_df = ref_df.drop("TIME", axis=1)
                        ref_df.index.name = "TIME"
                    else:
                        ref_time_axis = ref_df.index

                    # Rename reference file columns
                    ref_suffix = ref_file.replace(".csv", "").replace(".dat", "")
                    ref_df.columns = [f"{col}_{ref_suffix}" for col in ref_df.columns]

                    combined_dfs = [ref_df]
                    logging.info(
                        f"{log_tag('ANALY','RUN')} 94GHz reference: {ref_file} with shape {ref_df.shape}"
                    )

                    # Add other files, reindexed to reference time axis
                    for other_file in other_files:
                        other_df = shot_nas_data[other_file].copy()
                        if "TIME" in other_df.columns:
                            other_df.index = other_df["TIME"]
                            other_df = other_df.drop("TIME", axis=1)

                        # Reindex to reference time axis
                        other_df_reindexed = other_df.reindex(
                            ref_time_axis, method="nearest", limit=1
                        )

                        # Rename columns to avoid conflicts
                        other_suffix = other_file.replace(".csv", "").replace(
                            ".dat", ""
                        )
                        other_df_reindexed.columns = [
                            f"{col}_{other_suffix}"
                            for col in other_df_reindexed.columns
                        ]

                        combined_dfs.append(other_df_reindexed)
                        logging.info(
                            f"{log_tag('ANALY','RUN')} 94GHz additional: {other_file} reindexed to reference"
                        )

                    # Combine all 94GHz files
                    freq_combined_signals[freq_ghz] = pd.concat(combined_dfs, axis=1)
                    freq_combined_signals[freq_ghz].index.name = "TIME"
                    logging.info(
                        f"{log_tag('ANALY','RUN')} 94GHz: Combined shape {freq_combined_signals[freq_ghz].shape}"
                    )
                else:
                    logging.warning(f"{log_tag('ANALY','RUN')} No reference _0XX file found for 94GHz group")
            else:
                logging.warning(f"{log_tag('ANALY','RUN')} Unknown frequency group: {freq_ghz} GHz")

        # Keep freq_combined_signals as dict structure (94GHz and 280GHz separate)
        # This avoids time resolution conflicts and preserves frequency separation
        combined_signals = freq_combined_signals  # dict[str, DataFrame] format: {94.0: DataFrame, 280.0: DataFrame}
        
        # For backward compatibility, also keep main_freq for plotting and baseline correction
        main_freq = max(freq_combined_signals.keys()) if freq_combined_signals else None
        main_combined_signals = freq_combined_signals[main_freq] if main_freq else pd.DataFrame()
        logging.info(f"{log_tag('ANALY','RUN')} Combined signals organized by frequency: {list(combined_signals.keys())} GHz")
        if main_freq:
            logging.info(f"{log_tag('ANALY','RUN')} Using {main_freq} GHz as main combined_signals for baseline correction")
        
        current_vest_data = vest_data_by_shot.get(shot_num, {})
        vest_ip_data = current_vest_data.get("25k")
        if vest_ip_data is None:
            vest_ip_data = current_vest_data.get("250k")

        # --- 5. Density Calculation & Baseline Correction ---
        # Organize density_data by frequency to match combined_signals structure
        phase_converter = phi2ne.PhaseConverter()
        density_data = {}  # dict[str, DataFrame] format: {94.0: DataFrame, 280.0: DataFrame}

        # DEBUG: Check combined_signals structure (now a dict)
        logging.debug(f"{log_tag('ANALY','RUN')} DEBUGGING combined_signals (dict structure)")
        for freq_key, freq_df in combined_signals.items():
            logging.debug(f"{log_tag('ANALY','RUN')} {freq_key} GHz - Shape: {freq_df.shape}, Columns: {list(freq_df.columns)}")
            if len(freq_df) > 0:
                time_diff = freq_df.index.to_series().diff().mean()
                logging.debug(f"{log_tag('ANALY','RUN')} {freq_key} GHz - Time resolution: {time_diff:.2e} s")
        logging.debug(f"{log_tag('ANALY','RUN')} END DEBUG")

        if args.density:
            # Process each frequency group separately for density calculation
            # Create frequency-specific density DataFrames
            for freq_ghz, freq_data in freq_combined_signals.items():
                logging.info(f"{log_tag('ANALY','RUN')} Processing density calculation for {freq_ghz} GHz")

                # Initialize density DataFrame for this frequency
                freq_density_data = pd.DataFrame(index=freq_data.index)

                # Get files for this frequency
                freq_files = freq_groups[freq_ghz]["files"]
                freq_params_list = freq_groups[freq_ghz]["params"]

                # Calculate time difference from this frequency's data
                time_diff = freq_data.index.to_series().diff().mean()
                if time_diff == 0 or pd.isna(time_diff):
                    # Fallback: use 4ns resolution from file metadata
                    fs = 250e6  # 250 MHz sampling rate
                    logging.warning(
                        f"{log_tag('ANALY','RUN')} Invalid time resolution detected for {freq_ghz} GHz, using default fs={fs / 1e6:.1f} MHz"
                    )
                else:
                    fs = 1 / time_diff

                logging.info(
                    f"{log_tag('ANALY','RUN')} {freq_ghz} GHz - Sampling frequency: {fs / 1e6:.2f} MHz (time_diff: {time_diff})"
                )

                # Process each file in this frequency group
                for i, basename in enumerate(freq_files):
                    params = freq_params_list[i]
                    if params and params["method"] != "unknown":
                        file_suffix = basename.replace(".csv", "").replace(".dat", "")
                        logging.info(
                            f"{log_tag('ANALY','RUN')} Processing {basename}: {params['method']} method, {freq_ghz} GHz"
                        )

                        if params["method"] == "CDM":
                            # Find reference signal for this frequency group
                            ref_signal = None
                            ref_col_name = None
                            f_center = None

                            if params["ref_col"]:
                                # This file has its own reference
                                if freq_ghz == 280.0:
                                    # Single file case - no suffix
                                    ref_col_name = params["ref_col"]
                                else:
                                    # Multi-file case - with suffix
                                    ref_col_name = f"{params['ref_col']}_{file_suffix}"

                                if ref_col_name in freq_data.columns:
                                    ref_signal = (
                                        freq_data[ref_col_name].dropna().to_numpy()
                                    )
                                    logging.info(
                                        f"{log_tag('ANALY','RUN')} Using own reference {ref_col_name} for {basename}"
                                    )
                                else:
                                    logging.warning(
                                        f"{log_tag('ANALY','RUN')} Reference column {ref_col_name} not found for {basename}"
                                    )
                                    continue
                            else:
                                # This file has no reference - try to use group reference
                                group_ref_signal = None
                                group_ref_col = None

                                # Find reference from other files in the same frequency group
                                for other_basename, other_params in zip(
                                    freq_files, freq_params_list
                                ):
                                    if (
                                        other_params["ref_col"]
                                        and other_basename != basename
                                    ):
                                        other_suffix = other_basename.replace(
                                            ".csv", ""
                                        ).replace(".dat", "")
                                        potential_ref_col = (
                                            f"{other_params['ref_col']}_{other_suffix}"
                                        )
                                        if potential_ref_col in freq_data.columns:
                                            group_ref_signal = (
                                                freq_data[potential_ref_col]
                                                .dropna()
                                                .to_numpy()
                                            )
                                            group_ref_col = potential_ref_col
                                            logging.info(
                                                f"{log_tag('ANALY','RUN')} Using shared reference {potential_ref_col} from {other_basename} for {basename}"
                                            )
                                            break

                                if group_ref_signal is not None:
                                    ref_signal = group_ref_signal
                                    ref_col_name = group_ref_col
                                else:
                                    logging.warning(
                                        f"{log_tag('ANALY','RUN')} No reference signal available for {basename} - skipping CDM analysis"
                                    )
                                    continue

                            # Calculate center frequency for this reference
                            analyzer = spectrum.SpectrumAnalysis()
                            f_center = analyzer.find_center_frequency_fft(
                                ref_signal, fs
                            )

                            # If center frequency detection fails, use a reasonable default
                            if f_center == 0.0:
                                f_center = min(fs / 8, 20e6)
                                logging.warning(
                                    f"{log_tag('ANALY','RUN')} Center frequency detection failed for {basename} - using default: {f_center / 1e6:.2f} MHz"
                                )
                            else:
                                logging.info(
                                    f"{log_tag('ANALY','RUN')} {basename}: f_center = {f_center / 1e6:.2f} MHz"
                                )

                            # Process each probe channel
                            for probe_col in params["probe_cols"]:
                                if freq_ghz == 280.0:
                                    # Single file case - no suffix
                                    probe_col_name = probe_col
                                else:
                                    # Multi-file case - with suffix
                                    probe_col_name = f"{probe_col}_{file_suffix}"

                                if probe_col_name in freq_data.columns:
                                    probe_signal = (
                                        freq_data[probe_col_name].dropna().to_numpy()
                                    )
                                    phase = phase_converter.calc_phase_cdm(
                                        ref_signal, probe_signal, fs, f_center
                                    )
                                    # Store in frequency-specific density DataFrame
                                    freq_density_data[f"ne_{probe_col}_{basename}"] = (
                                        phase_converter.phase_to_density(
                                            phase, analysis_params=params
                                        )
                                    )
                                    logging.info(
                                        f"{log_tag('ANALY','RUN')} CDM: Calculated density for {probe_col} in {basename}"
                                    )
                                else:
                                    logging.warning(
                                        f"{log_tag('ANALY','RUN')} Probe column {probe_col_name} not found for {basename}"
                                    )

                        elif params["method"] == "FPGA":
                            if params["ref_col"]:
                                ref_col_name = f"{params['ref_col']}_{file_suffix}"
                                if ref_col_name in freq_data.columns:
                                    ref_signal = (
                                        freq_data[ref_col_name].dropna().to_numpy()
                                    )
                                    time_axis = freq_data.index.to_numpy()

                                    # Process each probe channel
                                    for probe_col in params["probe_cols"]:
                                        probe_col_name = f"{probe_col}_{file_suffix}"
                                        if probe_col_name in freq_data.columns:
                                            probe_signal = (
                                                freq_data[probe_col_name]
                                                .dropna()
                                                .to_numpy()
                                            )
                                            phase = phase_converter.calc_phase_fpga(
                                                ref_signal,
                                                probe_signal,
                                                time_axis,
                                                probe_signal,
                                                isflip=True,
                                            )
                                            # Store in frequency-specific density DataFrame
                                            freq_density_data[
                                                f"ne_{probe_col}_{basename}"
                                            ] = phase_converter.phase_to_density(
                                                phase, analysis_params=params
                                            )
                                            logging.info(
                                                f"{log_tag('ANALY','RUN')} FPGA: Calculated density for {probe_col} in {basename}"
                                            )
                                        else:
                                            logging.warning(
                                                f"{log_tag('ANALY','RUN')} Probe column {probe_col_name} not found for {basename}"
                                            )
                                else:
                                    logging.warning(
                                        f"{log_tag('ANALY','RUN')} Reference column {ref_col_name} not found for {basename}"
                                    )
                            else:
                                logging.warning(
                                    f"{log_tag('ANALY','RUN')} No reference signal for {basename} - skipping FPGA analysis"
                                )

                        elif params["method"] == "IQ":
                            # IQ method expects probe_cols to contain tuples like [('CH0', 'CH1')]
                            if params["probe_cols"] and len(params["probe_cols"]) > 0:
                                probe_cols_tuple = params["probe_cols"][
                                    0
                                ]  # Should be a tuple (CH0, CH1)
                                if (
                                    isinstance(probe_cols_tuple, tuple)
                                    and len(probe_cols_tuple) == 2
                                ):
                                    i_col, q_col = probe_cols_tuple
                                    i_col_name = f"{i_col}_{file_suffix}"
                                    q_col_name = f"{q_col}_{file_suffix}"

                                    if (
                                        i_col_name in freq_data.columns
                                        and q_col_name in freq_data.columns
                                    ):
                                        i_signal = (
                                            freq_data[i_col_name].dropna().to_numpy()
                                        )
                                        q_signal = (
                                            freq_data[q_col_name].dropna().to_numpy()
                                        )
                                        phase = phase_converter.calc_phase_iq(
                                            i_signal, q_signal
                                        )
                                        # Store in frequency-specific density DataFrame
                                        freq_density_data[f"ne_IQ_{basename}"] = (
                                            phase_converter.phase_to_density(
                                                phase, analysis_params=params
                                            )
                                        )
                                        logging.info(
                                            f"{log_tag('ANALY','RUN')} IQ: Calculated density for {basename}"
                                        )
                                    else:
                                        logging.warning(
                                            f"{log_tag('ANALY','RUN')} IQ columns {i_col_name}, {q_col_name} not found for {basename}"
                                        )
                                else:
                                    logging.warning(
                                        f"{log_tag('ANALY','RUN')} Invalid IQ probe_cols format for {basename}: {probe_cols_tuple}"
                                    )
                            else:
                                logging.warning(
                                    f"{log_tag('ANALY','RUN')} No probe_cols defined for IQ method in {basename}"
                                )

                        else:
                            logging.warning(
                                f"{log_tag('ANALY','RUN')} Unknown interferometry method: {params['method']} for file {basename}"
                            )
                
                # Store frequency-specific density data
                if not freq_density_data.empty:
                    # Apply baseline correction to this frequency's density data
                    if args.baseline and vest_ip_data is not None:
                        time_axis = freq_data.index.to_numpy()
                        ip_column_name = None
                        if args.baseline == "ip":
                            # Try to find IP column in vest_ip_data - common names
                            for col_name in vest_ip_data.columns:
                                if "ip" in col_name.lower() or "current" in col_name.lower():
                                    ip_column_name = col_name
                                    break
                        freq_density_data = phase_converter.correct_baseline(
                            freq_density_data,
                            time_axis,
                            args.baseline,
                            shot_num=shot_num,
                            vest_data=vest_ip_data,
                            ip_column_name=ip_column_name,
                        )
                    
                    # Store in frequency-keyed dict
                    density_data[str(freq_ghz)] = freq_density_data
                    logging.info(
                        f"{log_tag('ANALY','RUN')} Stored density data for {freq_ghz} GHz with {len(freq_density_data.columns)} columns"
                    )

        # --- 6. Final Output Bundle ---
        analysis_bundle = {
            "shot_info": {"shot_num": shot_num},
            "raw_data": {"nas": shot_nas_data, "vest": current_vest_data},
            "processed_data": {"signals": combined_signals, "density": density_data},
            "analysis_results": {"stft": shot_stft_data, "cwt": shot_cwt_data},
            "interferometry_params": shot_interferometry_params,
        }
        all_analysis_bundles[shot_num] = analysis_bundle

        # --- 7. Plotting and Saving ---
        if args.plot or args.save_plots:
            logging.info(f"{log_tag('ANALY','RUN')} Generating plots...")
            title_prefix = f"Shot #{shot_num} - " if shot_num else ""

            # Use a context manager to handle plot creation and showing/saving
            # block=False if --no_plot_block is specified to prevent execution blocking
            with plots.ifion_plotting(
                show_plots=args.plot,
                save_dir=Path(args.results_dir) / str(shot_num)
                if args.save_plots
                else None,
                save_prefix=title_prefix,
                block=not args.no_plot_block,  # Block unless --no_plot_block is specified
            ):
                if not args.no_plot_ft:
                    if shot_stft_data:
                        plots.plot_spectrograms(
                            shot_stft_data,
                            title_prefix=title_prefix,
                            trigger_time=args.trigger_time,
                            downsample=args.downsample,
                        )
                    if shot_cwt_data:
                        plots.plot_cwt(
                            shot_cwt_data,
                            trigger_time=args.trigger_time,
                            title_prefix=title_prefix,
                        )

                if not args.no_plot_raw:
                    # For plotting, use main_combined_signals (single DataFrame) for backward compatibility
                    # If plotting functions need dict structure, they can be updated later
                    plot_signals = main_combined_signals if isinstance(combined_signals, dict) else combined_signals
                    plot_density = density_data.get(str(main_freq), pd.DataFrame()) if isinstance(density_data, dict) else density_data
                    plots.plot_analysis_overview(
                        shot_num,
                        {"Processed Signals": plot_signals},
                        {"Density": plot_density},
                        vest_ip_data,
                        trigger_time=args.trigger_time,
                        title_prefix=title_prefix,
                        downsample=args.downsample,
                    )

        # --- Saving Logic Update ---
        if args.save_data:
            # The logic inside save_results_to_hdf5 now handles shot_num=0 correctly
            output_dir = (
                Path(args.results_dir) / str(shot_num)
                if shot_num != 0
                else "unknown_shots"
            )
            
            # Convert combined_signals dict to format expected by save_results_to_hdf5
            # save_results_to_hdf5 expects signals as dict[str, DataFrame]
            # combined_signals is now dict[float, DataFrame] (keys are 94.0, 280.0)
            # Convert to dict[str, DataFrame] with string keys
            signals_dict = {}
            if isinstance(combined_signals, dict):
                for freq_key, freq_df in combined_signals.items():
                    # Convert frequency key to string and prepare DataFrame
                    freq_str = str(freq_key)
                    # Convert index to TIME column if needed
                    freq_df_copy = freq_df.copy()
                    if freq_df_copy.index.name == "TIME":
                        freq_df_copy = freq_df_copy.reset_index()
                    elif "TIME" not in freq_df_copy.columns:
                        if freq_df_copy.index.name is None or freq_df_copy.index.name == "":
                            freq_df_copy = freq_df_copy.reset_index()
                            freq_df_copy.rename(columns={freq_df_copy.columns[0]: "TIME"}, inplace=True)
                    # Use frequency as part of the key name
                    signals_dict[f"freq_{freq_str}_GHz"] = freq_df_copy
            else:
                # Fallback: if combined_signals is still DataFrame (shouldn't happen)
                combined_df = combined_signals.copy()
                if combined_df.index.name == "TIME":
                    combined_df = combined_df.reset_index()
                signals_dict["combined"] = combined_df
            
            # Convert density_data dict to format expected by save_results_to_hdf5
            # For now, combine all frequency density data into one DataFrame
            # TODO: Update save_results_to_hdf5 to handle dict structure
            if isinstance(density_data, dict):
                # Combine all frequency density DataFrames
                # Use the main frequency's index as reference
                if main_freq and str(main_freq) in density_data:
                    combined_density = density_data[str(main_freq)].copy()
                    # Add other frequencies' density data if they exist
                    for freq_key, freq_df in density_data.items():
                        if freq_key != str(main_freq) and not freq_df.empty:
                            # Reindex to main frequency's time axis
                            freq_df_reindexed = freq_df.reindex(
                                combined_density.index, method="nearest", limit=1
                            )
                            # Add columns with frequency prefix
                            for col in freq_df_reindexed.columns:
                                combined_density[f"{freq_key}GHz_{col}"] = freq_df_reindexed[col]
                    density_data_for_save = combined_density
                elif density_data:
                    # Use first available frequency
                    first_freq = list(density_data.keys())[0]
                    density_data_for_save = density_data[first_freq]
                else:
                    density_data_for_save = pd.DataFrame()
            else:
                density_data_for_save = density_data
            
            file_io.save_results_to_hdf5(
                output_dir,
                shot_num,
                signals_dict,
                shot_stft_data,
                shot_cwt_data,
                density_data_for_save,
                vest_ip_data,
            )

    logging.info(f"{log_tag('ANALY','RUN')} Full Analysis Finished")
    return all_analysis_bundles


def main():
    """
    Main entry point for the IFI analysis script.
    Parses command-line arguments and orchestrates the analysis process.

    Args:
        None

    Returns:
        None

    Raises:
        See log file for more details.
    """
    LogManager().get_logger(__name__, level="INFO")

    parser = argparse.ArgumentParser(
        description="IFI Analysis Program",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Argument Parsing ---
    parser.add_argument(
        "query",
        nargs="+",
        help="One or more analysis targets, which can be:\n"
        "- Shot numbers (e.g., 45821)\n"
        '- A range of shots (e.g., "45821:45823")\n'
        '- A glob pattern for shots (e.g., "45*")\n'
        "- Full paths to specific data files.",
    )
    parser.add_argument(
        "--data_folders",
        type=str,
        default=None,
        help="Comma-separated list of data folders to search. Overrides config.ini defaults.",
    )
    parser.add_argument(
        "--add_path",
        action="store_true",
        help="""If specified, adds the --data_folders paths to the default paths 
            from config.ini instead of overriding them.""",
    )
    parser.add_argument(
        "--force_remote",
        action="store_true",
        help="Force fetching data from the remote NAS, ignoring any local cache.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="ifi/results",
        help="The directory to store analysis results and cached data.",
    )

    # Flags for data processing
    parser.add_argument(
        "--no_offset_removal",
        action="store_true",
        help="Disable the moving average offset removal step.",
    )
    parser.add_argument(
        "--offset_window",
        type=int,
        default=2001,
        help="The window size for the moving average offset removal.",
    )

    # Flags for data processing with freq.-time transform
    parser.add_argument(
        "--stft",
        action="store_true",
        help="Perform STFT analysis when retrieving data.",
    )
    parser.add_argument(
        "--stft_cols",
        nargs="*",
        type=int,
        default=[],
        help="""The indices of the columns to perform STFT analysis on.
            If empty, all columns will be analyzed.""",
    )
    parser.add_argument(
        "--cwt", action="store_true", help="Perform CWT analysis when retrieving data."
    )
    parser.add_argument(
        "--cwt_cols",
        nargs="*",
        type=int,
        default=[],
        help="""The indices of the columns to perform CWT analysis on.
            If empty, all columns will be analyzed.""",
    )

    # Flags for plotting
    parser.add_argument(
        "--plot", action="store_true", help="Show plots of the analysis results."
    )
    parser.add_argument(
        "--no_plot_block",
        action="store_true",
        help="""Don't block execution when showing plots. 
            Plots will be displayed in non-blocking mode, allowing the analysis to continue without waiting for plot windows to be closed.""",
    )
    parser.add_argument(
        "--no_plot_raw",
        action="store_true",
        help="""Don't show the plots of the raw data.""",
    )
    parser.add_argument(
        "--no_plot_ft",
        action="store_true",
        help="""Don't show the plots of the time-frequency trasnforms.""",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Downsample factor for plotting to improve performance. Default: 10.",
    )
    parser.add_argument(
        "--trigger_time",
        type=float,
        default=0.290,
        help="Trigger time in seconds to be added to the time axis for plotting.",
    )

    # Analysis-specific flags
    parser.add_argument(
        "--density", action="store_true", help="Perform phase and density calculation."
    )
    parser.add_argument(
        "--vest_fields",
        nargs="*",
        type=int,
        default=[],
        help="Space-separated list of VEST DB field IDs to load and process.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["ip", "trig"],
        default=None,
        help="Perform baseline correction on density data.",
    )

    # Output flags
    parser.add_argument(
        "--save_plots", action="store_true", help="Save generated plots to files."
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save processed data to HDF5 files."
    )

    # Dask scheduler

    parser.add_argument(
        "--scheduler",
        type=str,
        default="threads",
        choices=["threads", "processes", "single-threaded"],
        help="Dask scheduler to use for parallel processing.",
    )

    args = parser.parse_args()

    # --- Initialize DB Controllers ---
    try:
        nas_db = NAS_DB(config_path="ifi/config.ini")
        vest_db = VEST_DB(config_path="ifi/config.ini")
    except FileNotFoundError:
        logging.error(f"{log_tag('ANALY','MAIN')} Configuration file 'ifi/config.ini' not found. Exiting.")
        return
    except Exception as e:
        logging.error(f"{log_tag('ANALY','MAIN')} Failed to initialize database controllers: {e}")
        return

    # The main change is here: call run_analysis with the raw query
    run_analysis(args.query, args, nas_db, vest_db)


if __name__ == "__main__":
    main()
