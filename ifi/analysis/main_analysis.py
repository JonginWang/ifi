#!/usr/bin/env python3
"""
    IFI Analysis Main Script
    ========================

    This script orchestrates the data analysis workflow for the IFI project.
    It handles data loading, processing, analysis, and visualization.

    Functions:
        run_analysis: Runs the analysis pipeline.
            - Inputs:
                - query: Shot number or pattern
                - args: argparse.Namespace object containing the analysis parameters
                - nas_db: NAS_DB instance
                - vest_db: VEST_DB instance
            - Outputs:
                - analysis_data: Dictionary containing the analysis data
                - stft_results: Dictionary containing the STFT results
                - cwt_results: Dictionary containing the CWT results
        load_and_process_file: Loads a single file using the NAS_DB instance, then processes it through refining, offset removal, and STFT analysis.
            - Inputs:
                - nas_instance: NAS_DB instance
                - file_path: Path to the file to process
                - args: argparse.Namespace object containing the analysis parameters
            - Outputs:
                - file_path: Path to the file that was processed
                - df_processed: DataFrame containing the processed data
                - stft_result: Dictionary containing the STFT results
                - cwt_result: Dictionary containing the CWT results
        main: Main function that parses command line arguments and runs the analysis pipeline.
            - Inputs:
                - query: Shot number or pattern
                - args: argparse.Namespace object containing the analysis parameters
                - nas_db: NAS_DB instance
                - vest_db: VEST_DB instance
            - Outputs:
                - analysis_data: Dictionary containing the analysis data
                - stft_results: Dictionary containing the STFT results
                - cwt_results: Dictionary containing the CWT results
"""
# ============================================================================
# CRITICAL: Set up numba cache BEFORE any imports that use numba/ssqueezepy
# ============================================================================

import logging
from pathlib import Path

from ifi.utils.cache_setup import setup_project_cache
cache_config = setup_project_cache()

# Now safe to import other modules
import argparse
import re
import time
from typing import List, Union
from collections import defaultdict
import pandas as pd
import dask

from ifi.db_controller.nas_db import NAS_DB
from ifi.db_controller.vest_db import VEST_DB
from ifi.analysis import processing, plots, spectrum, phi2ne
from ifi.utils.common import LogManager, FlatShotList
from ifi.utils import file_io
from ifi.analysis.phi2ne import get_interferometry_params

# The LogManager is better to be initialized inside the main() function 
# or at the start of the script logic.
# LogManager(level="INFO") # This call is redundant if called in main()


@dask.delayed
def load_and_process_file(nas_instance, file_path, args):
    """
    Loads a single file using the NAS_DB instance, then processes it
    through refining, offset removal, and STFT analysis.
    This function is designed to be run in parallel by Dask.
    """
    logging.info("\n" + "="*80 + "\n" + f" Starting processing for: {Path(file_path).name} ".center(80, "=") + "\n" + "="*80 + "\n")
    
    # 1. Read single file data
    # Use get_shot_data for better path handling and caching
    data_dict = nas_instance.get_shot_data(file_path, force_remote=args.force_remote)
    if not data_dict or file_path not in data_dict:
        logging.warning("\n" + "="*80 + "\n" + f"  Failed to read {file_path}. Skipping.  ".center(80, "!") + "\n" + "="*80 + "\n")
        return None, None, None, None # Return None for all results
    
    df_raw = data_dict[file_path]

    # 2. Refine data (Removing nan values)
    df_refined = processing.refine_data(df_raw)
    logging.info("\n" + f"  Data refined for {file_path}  ".center(80, "=") + "\n")

    # 3. Remove offset
    if not args.no_offset_removal:
        df_processed = processing.remove_offset(df_refined, window_size=args.offset_window)
        logging.info("\n" + f"  Offset removed from {file_path}  ".center(80, "=") + "\n")
    else:
        df_processed = df_refined
    
    # 4. Perform STFT analysis if requested
    stft_result = None
    if args.stft:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != 'TIME']
        
        cols_to_analyze_idxs = range(len(all_data_cols))
        if args.ft_cols is not None:
            cols_to_analyze_idxs = args.ft_cols

        fs = 1 / df_processed['TIME'].diff().mean()
        
        stft_result_for_file = {}
        for col_idx in cols_to_analyze_idxs:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            freq_STFT, time_STFT, STFT_matrix = analyzer.compute_stft(signal, fs)
            stft_result_for_file[col_name] = {'time_STFT': time_STFT, 'freq_STFT': freq_STFT, 'STFT_matrix': STFT_matrix}
         
        stft_result = {file_path: stft_result_for_file}
        logging.info("\n" + f"  STFT analysis complete for {file_path}  ".center(80, "=") + "\n")

    # 5. Perform CWT analysis if requested
    cwt_result = None
    if args.cwt:
        analyzer = spectrum.SpectrumAnalysis()
        all_data_cols = [col for col in df_processed.columns if col != 'TIME']
        
        cols_to_analyze_idxs = range(len(all_data_cols))
        if args.ft_cols is not None:
            cols_to_analyze_idxs = args.ft_cols

        fs = 1 / df_processed['TIME'].diff().mean()
        
        cwt_result_for_file = {}
        for col_idx in cols_to_analyze_idxs:
            if col_idx < 0 or col_idx >= len(all_data_cols):
                continue
            col_name = all_data_cols[col_idx]
            signal = df_processed[col_name].to_numpy()
            freq_CWT, CWT_matrix = analyzer.compute_cwt(signal, fs)
            cwt_result_for_file[col_name] = {"time_CWT": df_processed['TIME'], "freq_CWT": freq_CWT, "CWT_matrix": CWT_matrix}
         
        cwt_result = {file_path: cwt_result_for_file}
        logging.info(f"  CWT analysis complete for {file_path}  ".center(80, "="))

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
    logging.info( "\n" + " Parsing Analysis Query ".center(80, "=") + "\n")
    flat_list = FlatShotList(query)
    logging.info(f"\n- Found {len(flat_list.nums)} unique shot numbers: \n{flat_list.nums}\n")
    logging.info(f"\n- Found {len(flat_list.paths)} unique file paths: \n{flat_list.paths}\n")

    if not flat_list.all:
        logging.warning("\n" + "  Query resulted in an empty list of targets. Nothing to do.  ".center(80, "!") + "\n")
        return

    # --- 1. Find and Load Data ---
    target_files = nas_db.find_files(
        query=flat_list.all,
        data_folders=args.data_folders,
        add_path=args.add_path,
        force_remote=args.force_remote
    )
    if not target_files:
        logging.warning("\n" + "  No files found for the given query. Skipping.  ".center(80, "!") + "\n")
        return

    # Group files by shot number and determine interferometry parameters
    files_by_shot = defaultdict(list)
    interferometry_params_by_file = {}
    for f in target_files:
        match = re.search(r'(\d{5,})', Path(f).name)
        if match:
            shot_num = int(match.group(1))
            files_by_shot[shot_num].append(f)
            # Get parameters for each individual file (not just shot number)
            params = get_interferometry_params(shot_num, Path(f).name)
            interferometry_params_by_file[f] = params
            logging.info(f"\n- Interferometry (Shot #{shot_num}) params for {Path(f).name}: ")
            logging.info(f"{params['method']} method, {params['freq_ghz']} GHz" + "\n")
        else:
            files_by_shot['unknown'].append(f)
            # For unknown files, use shot number 0 as default
            params = get_interferometry_params(0, Path(f).name)
            interferometry_params_by_file[f] = params
            logging.info(f"\n- Interferometry (Shot #00000) params for {Path(f).name}: ")
            logging.info(f"{params['method']} method, {params['freq_ghz']} GHz" + "\n")

    logging.info("\n" + f"  Grouped files into {len(files_by_shot)} shot(s).  ".center(80, "=") + "\n")

    # Load VEST data for all relevant shots
    vest_data_by_shot = defaultdict(dict)
    if flat_list.nums:
        logging.info(f"\n- Loading VEST data for shots: \n{flat_list.nums}\n")
        for shot_num in flat_list.nums:
            vest_data_by_shot[shot_num] = vest_db.load_shot(shot=shot_num, fields=args.vest_fields) if shot_num > 0 else {}

    # --- Dask Task Creation with Progress Tracking ---
    tasks = [dask.delayed(load_and_process_file)(nas_db, f, args) for f in target_files]
    
    logging.info("\n" + f"Starting Dask computation for {len(tasks)} tasks...".center(80, "=") + "\n")
    logging.info(f"Using scheduler: {args.scheduler}")
    logging.info(f"Target files: {len(target_files)}")
    
    # Execute with progress tracking and optimized scheduler
    start_time = time.time()
    
    # Use optimal scheduler based on task count and system resources
    if len(tasks) <= 4:
        scheduler = 'threads'  # Better for I/O bound tasks
    else:
        scheduler = args.scheduler if args.scheduler else 'threads'
    
    logging.info(f"Using scheduler: {scheduler}")
    results = dask.compute(*tasks, scheduler=scheduler)
    end_time = time.time()
    
    logging.info("\n" + "Dask computation finished.".center(80, "=") + "\n")
    logging.info(f"Processing time: {end_time - start_time:.2f} seconds")
    logging.info(f"Average time per file: {(end_time - start_time) / len(target_files):.2f} seconds")

    # --- Process Dask Results ---
    analysis_data = defaultdict(dict)
    stft_results = defaultdict(dict)
    cwt_results = defaultdict(dict)
    for file_path, df, stft_result, cwt_result in results:
        if df is None:
            continue
        
        match = re.search(r'(\d{5,})', Path(file_path).name)
        shot_num = int(match.group(1)) if match else 0 # Group under 0 if no shot number
        
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
        
        logging.info("\n" + f"  Post-processing for Shot #{shot_num} with {len(shot_nas_data)} files  ".center(80, "=") + "\n")
        
        # --- Collect interferometry parameters for this shot ---
        shot_files = files_by_shot.get(shot_num, [])
        shot_interferometry_params = {}
        for file_path in shot_files:
            basename = Path(file_path).name
            if file_path in interferometry_params_by_file:
                params = interferometry_params_by_file[file_path]
                shot_interferometry_params[basename] = params
                logging.info(f"File {basename}: ")
                logging.info(f"{params['method']} method, {params['freq_ghz']}GHz, ref={params['ref_col']}, probes={params['probe_cols']}" + "\n")
        
        # --- Group Data by Frequency ---
        # Group files by frequency based on interferometry parameters
        freq_groups = {}
        for basename, params in shot_interferometry_params.items():
            freq_ghz = params.get('freq_ghz', 94.0)  # Default to 94GHz
            if freq_ghz not in freq_groups:
                freq_groups[freq_ghz] = {'files': [], 'params': []}
            freq_groups[freq_ghz]['files'].append(basename)
            freq_groups[freq_ghz]['params'].append(params)
        
        logging.info(f"\n- Frequency groups found: {list(freq_groups.keys())} GHz" + "\n")
        
        # Create frequency-specific combined signals
        freq_combined_signals = {}
        for freq_ghz, group_info in freq_groups.items():
            files_in_group = group_info['files']
            logging.info(f"\n- Processing {freq_ghz}GHz group with files: \n{files_in_group}\n")
            
            if freq_ghz == 280.0:
                # 280GHz: Use _ALL file independently
                all_file = None
                for basename in files_in_group:
                    if '_ALL' in basename and basename in shot_nas_data:
                        all_file = basename
                        break
                
                if all_file:
                    df = shot_nas_data[all_file].copy()
                    if 'TIME' in df.columns:
                        df.index = df['TIME']
                        df = df.drop('TIME', axis=1)
                        df.index.name = 'TIME'
                    freq_combined_signals[freq_ghz] = df
                    logging.info(f"- 280GHz: Using {all_file} with shape {df.shape}")
                else:
                    logging.warning("! No _ALL file found for 280GHz group")
                    
            elif freq_ghz == 94.0:
                # 94GHz: Use _0XX file as reference time axis, combine with other files
                ref_file = None
                other_files = []
                
                # Find reference file (_0XX pattern) and other files
                for basename in files_in_group:
                    if '_0' in basename and basename in shot_nas_data:
                        ref_file = basename
                    elif basename in shot_nas_data:
                        other_files.append(basename)
                
                if ref_file:
                    # Use reference file's time axis
                    ref_df = shot_nas_data[ref_file].copy()
                    if 'TIME' in ref_df.columns:
                        ref_time_axis = ref_df['TIME']
                        ref_df.index = ref_time_axis
                        ref_df = ref_df.drop('TIME', axis=1)
                        ref_df.index.name = 'TIME'
                    else:
                        ref_time_axis = ref_df.index
                    
                    # Rename reference file columns
                    ref_suffix = ref_file.replace('.csv', '').replace('.dat', '')
                    ref_df.columns = [f"{col}_{ref_suffix}" for col in ref_df.columns]
                    
                    combined_dfs = [ref_df]
                    logging.info(f"\n- 94GHz reference: {ref_file} with shape {ref_df.shape}\n")
                    
                    # Add other files, reindexed to reference time axis
                    for other_file in other_files:
                        other_df = shot_nas_data[other_file].copy()
                        if 'TIME' in other_df.columns:
                            other_df.index = other_df['TIME']
                            other_df = other_df.drop('TIME', axis=1)
                        
                        # Reindex to reference time axis
                        other_df_reindexed = other_df.reindex(ref_time_axis, method="nearest", limit=1)
                        
                        # Rename columns to avoid conflicts
                        other_suffix = other_file.replace('.csv', '').replace('.dat', '')
                        other_df_reindexed.columns = [f"{col}_{other_suffix}" for col in other_df_reindexed.columns]
                        
                        combined_dfs.append(other_df_reindexed)
                        logging.info(f"\n- 94GHz additional: \n{other_file} reindexed to reference\n")
                    
                    # Combine all 94GHz files
                    freq_combined_signals[freq_ghz] = pd.concat(combined_dfs, axis=1)
                    freq_combined_signals[freq_ghz].index.name = 'TIME'
                    logging.info(f"- 94GHz: Combined shape {freq_combined_signals[freq_ghz].shape}")
                else:
                    logging.warning("! No reference _0XX file found for 94GHz group")
            else:
                logging.warning(f"! Unknown frequency group: {freq_ghz} GHz")
        
        # For backward compatibility, use the largest frequency group as main combined_signals
        if freq_combined_signals:
            main_freq = max(freq_combined_signals.keys())
            combined_signals = freq_combined_signals[main_freq]
            logging.info(f"\n- Using {main_freq} GHz as main combined_signals\n")
        current_vest_data = vest_data_by_shot.get(shot_num, {})
        vest_ip_data = current_vest_data.get('25k')
        if vest_ip_data is None:
            vest_ip_data = current_vest_data.get('250k')
        
        # --- 5. Density Calculation & Baseline Correction ---
        phase_converter = phi2ne.PhaseConverter()
        density_data = pd.DataFrame(index=combined_signals.index)
        
        # DEBUG: Check combined_signals structure
        logging.debug("\n" + "  DEBUGGING combined_signals  ".center(80, "^") + "\n")
        logging.debug(f"- Combined signals shape: {combined_signals.shape}")
        logging.debug(f"- Available columns: {list(combined_signals.columns)}")
        logging.debug(f"- Index info: {combined_signals.index.name}\n- range: {combined_signals.index.min():.6f} to {combined_signals.index.max():.6f}")
        logging.debug(f"- Index length: {len(combined_signals.index)}")
        
        if len(combined_signals) > 0:
            logging.debug("First 5 rows of combined_signals:")
            logging.debug(f"{combined_signals.head()}")
            
            # Check time resolution
            time_diff = combined_signals.index.to_series().diff().mean()
            logging.debug("     Time resolution analysis:")
            logging.debug(f"     - Mean time diff: {time_diff}")
            logging.debug(f"     - Time diff type: {type(time_diff)}")
            logging.debug(f"     - Is NaN?: {pd.isna(time_diff)}")
            logging.debug(f"     - Is zero?: {time_diff == 0}")
            
            # Check first few time differences
            time_diffs = combined_signals.index.to_series().diff().head(10)
            logging.debug(f"   - First 10 time diffs: {time_diffs.tolist()}")
        
        logging.debug("\n" + "  END DEBUG  ".center(80, "^") + "\n")
        
        if args.density:
            # Process each frequency group separately for density calculation
            for freq_ghz, freq_data in freq_combined_signals.items():
                logging.info(f"- Processing density calculation for {freq_ghz} GHz")
                
                # Get files for this frequency
                freq_files = freq_groups[freq_ghz]['files']
                freq_params_list = freq_groups[freq_ghz]['params']
                
                # Calculate time difference from this frequency's data
                time_diff = freq_data.index.to_series().diff().mean()
                if time_diff == 0 or pd.isna(time_diff):
                    # Fallback: use 4ns resolution from file metadata
                    fs = 250e6  # 250 MHz sampling rate
                    logging.warning(f"! Invalid time resolution detected for {freq_ghz} GHz, using default fs={fs/1e6:.1f} MHz")
                else:
                    fs = 1 / time_diff
                
                logging.info(f"- {freq_ghz} GHz - Sampling frequency: {fs/1e6:.2f} MHz (time_diff: {time_diff})")
                
                # Process each file in this frequency group
                for i, basename in enumerate(freq_files):
                    params = freq_params_list[i]
                    if params and params['method'] != 'unknown':
                        file_suffix = basename.replace('.csv', '').replace('.dat', '')
                        logging.info(f"- Processing {basename}: \n{params['method']} method, {freq_ghz} GHz")
                        
                        if params['method'] == 'CDM':
                            # Find reference signal for this frequency group
                            ref_signal = None
                            ref_col_name = None
                            f_center = None
                            
                            if params['ref_col']:
                                # This file has its own reference
                                if freq_ghz == 280.0:
                                    # Single file case - no suffix
                                    ref_col_name = params['ref_col']
                                else:
                                    # Multi-file case - with suffix
                                    ref_col_name = f"{params['ref_col']}_{file_suffix}"
                                
                                if ref_col_name in freq_data.columns:
                                    ref_signal = freq_data[ref_col_name].dropna().to_numpy()
                                    logging.info(f"- Using own reference {ref_col_name} for {basename}")
                                else:
                                    logging.warning(f"! Reference column {ref_col_name} not found for {basename}")
                                    continue
                            else:
                                # This file has no reference - try to use group reference
                                group_ref_signal = None
                                group_ref_col = None
                                
                                # Find reference from other files in the same frequency group
                                for other_basename, other_params in zip(freq_files, freq_params_list):
                                    if other_params['ref_col'] and other_basename != basename:
                                        other_suffix = other_basename.replace('.csv', '').replace('.dat', '')
                                        potential_ref_col = f"{other_params['ref_col']}_{other_suffix}"
                                        if potential_ref_col in freq_data.columns:
                                            group_ref_signal = freq_data[potential_ref_col].dropna().to_numpy()
                                            group_ref_col = potential_ref_col
                                            logging.info(f"- Using shared reference {potential_ref_col} from {other_basename} for {basename}")
                                            break
                                
                                if group_ref_signal is not None:
                                    ref_signal = group_ref_signal
                                    ref_col_name = group_ref_col
                                else:
                                    logging.warning(f"! No reference signal available for {basename} \n    - skipping CDM analysis")
                                    continue
                            
                            # Calculate center frequency for this reference
                            analyzer = spectrum.SpectrumAnalysis()
                            f_center = analyzer.find_center_frequency_fft(ref_signal, fs)
                            
                            # If center frequency detection fails, use a reasonable default
                            if f_center == 0.0:
                                f_center = min(fs / 8, 20e6)
                                logging.warning(f"! Center frequency detection failed for {basename}\n    - using default: {f_center/1e6:.2f} MHz")
                            else:
                                logging.info(f"- {basename}: f_center = {f_center/1e6:.2f} MHz")
                            
                            # Process each probe channel
                            for probe_col in params['probe_cols']:
                                if freq_ghz == 280.0:
                                    # Single file case - no suffix
                                    probe_col_name = probe_col
                                else:
                                    # Multi-file case - with suffix
                                    probe_col_name = f"{probe_col}_{file_suffix}"
                                
                                if probe_col_name in freq_data.columns:
                                    probe_signal = freq_data[probe_col_name].dropna().to_numpy()
                                    phase = phase_converter.calc_phase_cdm(ref_signal, probe_signal, fs, f_center)
                                    density_data[f"ne_{probe_col}_{basename}"] = phase_converter.phase_to_density(phase, analysis_params=params)
                                    logging.info(f"- CDM: Calculated density for {probe_col} in {basename}")
                                else:
                                    logging.warning(f"! Probe column {probe_col_name} not found for {basename}")
                        
                        elif params['method'] == 'FPGA':
                            if params['ref_col']:
                                ref_col_name = f"{params['ref_col']}_{file_suffix}"
                                if ref_col_name in freq_data.columns:
                                    ref_signal = freq_data[ref_col_name].dropna().to_numpy()
                                    time_axis = freq_data.index.to_numpy()
                                    
                                    # Process each probe channel
                                    for probe_col in params['probe_cols']:
                                        probe_col_name = f"{probe_col}_{file_suffix}"
                                        if probe_col_name in freq_data.columns:
                                            probe_signal = freq_data[probe_col_name].dropna().to_numpy()
                                            phase = phase_converter.calc_phase_fpga(ref_signal, probe_signal, time_axis, probe_signal, isflip=True)
                                            density_data[f"ne_{probe_col}_{basename}"] = phase_converter.phase_to_density(phase, analysis_params=params)
                                            logging.info(f"- FPGA: Calculated density for {probe_col} in {basename}")
                                        else:
                                            logging.warning(f"! Probe column {probe_col_name} not found for {basename}")
                                else:
                                    logging.warning(f"! Reference column {ref_col_name} not found for {basename}")
                            else:
                                logging.warning(f"! No reference signal for {basename} \n    - skipping FPGA analysis")
                        
                        elif params['method'] == 'IQ':
                            # IQ method expects probe_cols to contain tuples like [('CH0', 'CH1')]
                            if params['probe_cols'] and len(params['probe_cols']) > 0:
                                probe_cols_tuple = params['probe_cols'][0]  # Should be a tuple (CH0, CH1)
                                if isinstance(probe_cols_tuple, tuple) and len(probe_cols_tuple) == 2:
                                    i_col, q_col = probe_cols_tuple
                                    i_col_name = f"{i_col}_{file_suffix}"
                                    q_col_name = f"{q_col}_{file_suffix}"
                                    
                                    if i_col_name in freq_data.columns and q_col_name in freq_data.columns:
                                        i_signal = freq_data[i_col_name].dropna().to_numpy()
                                        q_signal = freq_data[q_col_name].dropna().to_numpy()
                                        phase = phase_converter.calc_phase_iq(i_signal, q_signal)
                                        density_data[f"ne_IQ_{basename}"] = phase_converter.phase_to_density(phase, analysis_params=params)
                                        logging.info(f"- IQ: Calculated density for {basename}")
                                    else:
                                        logging.warning(f"! IQ columns {i_col_name}, {q_col_name} not found for {basename}")
                                else:
                                    logging.warning(f"! Invalid IQ probe_cols format for {basename}: {probe_cols_tuple}")
                            else:
                                logging.warning(f"! No probe_cols defined for IQ method in {basename}")
                        
                        else:
                            logging.warning(f"! Unknown interferometry method: {params['method']} for file {basename}")

            # Apply baseline correction to all density columns
            if args.baseline and vest_ip_data is not None and not density_data.empty:
                time_axis = combined_signals.index.to_numpy()
                ip_column_name = None
                if args.baseline == 'ip':
                    # Try to find IP column in vest_ip_data - common names
                    for col_name in vest_ip_data.columns:
                        if 'ip' in col_name.lower() or 'current' in col_name.lower():
                            ip_column_name = col_name
                            break
                density_data = phase_converter.correct_baseline(
                    density_data, time_axis, args.baseline, 
                    shot_num=shot_num, vest_data=vest_ip_data, ip_column_name=ip_column_name
                )
                
        # --- 6. Final Output Bundle ---
        analysis_bundle = {
            'shot_info': {'shot_num': shot_num},
            'raw_data': {'nas': shot_nas_data, 'vest': current_vest_data},
            'processed_data': {'signals': combined_signals, 'density': density_data},
            'analysis_results': {'stft': shot_stft_data, 'cwt': shot_cwt_data},
            'interferometry_params': shot_interferometry_params
        }
        all_analysis_bundles[shot_num] = analysis_bundle

        # --- 7. Plotting and Saving ---
        if args.plot or args.save_plots:
            logging.info("\n" + "  Generating plots...  ".center(80, "=") + "\n")
            title_prefix = f"Shot #{shot_num} - " if shot_num else ""
            
            # Use a context manager to handle plot creation and showing/saving
            with plots.ifion_plotting(interactive=args.plot, save_dir=Path(args.results_dir) / str(shot_num) if args.save_plots else None, save_prefix=title_prefix):
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
                        vest_ip_data,
                        trigger_time=args.trigger_time,
                        title_prefix=title_prefix,
                        downsample=args.downsample
                    )

        # --- Saving Logic Update ---
        if args.save_data:
            # The logic inside save_results_to_hdf5 now handles shot_num=0 correctly
            output_dir = Path(args.results_dir) / str(shot_num) if shot_num != 0 else 'unknown_shots'
            file_io.save_results_to_hdf5(
                output_dir, shot_num, combined_signals, shot_stft_data,
                shot_cwt_data, density_data, vest_ip_data
            )

    logging.info("\n" + "  Full Analysis Finished  ".center(80, "=") + "\n")
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