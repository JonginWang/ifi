#!/usr/bin/env python3
"""
IFI Analysis Main Script
========================

This script orchestrates the data analysis workflow for the IFI project.
It handles data loading, processing, analysis, and visualization.

Functions:
    load_and_process_file:
        Loads a single file using the NasDB instance,
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

import argparse
import logging
from pathlib import Path
from typing import Any

import dask

from ..db_controller.nas_db import NasDB
from ..db_controller.vest_db import VestDB
from ..utils.log_manager import LogManager, log_tag
from ..utils.vest_utils import FlatShotList
from .analysis_pipeline.cache_phase import (
    resolve_analysis_requirements,
    scan_cached_results_for_query,
)
from .analysis_pipeline.compute_phase import (
    aggregate_dask_results,
    run_parallel_file_processing,
)
from .analysis_pipeline.discovery_phase import (
    build_analysis_query,
    group_files_and_interferometry,
    load_vest_data_for_shots,
)
from .analysis_pipeline.file_phase import (
    compute_cwt_results,
    compute_stft_results,
    load_single_file_raw_data,
    refine_and_preprocess_signal,
)
from .analysis_pipeline.output_phase import (
    export_envelope_outputs,
    merge_cached_results_with_bundles,
    plot_shot_outputs,
    save_shot_outputs,
)
from .analysis_pipeline.shot_phase import (
    build_frequency_context_for_shot,
    collect_shot_interferometry_params,
    resolve_vest_ip_data,
)
from .workflow import build_cached_analysis_bundles
from .workflow_density import calculate_density_data_by_frequency


@dask.delayed
def load_and_process_file(config_path, file_path, args):
    """
    Loads a single file using a dedicated NasDB instance, then processes it
    through refining, offset removal, and STFT/CWT analysis.
    This function is designed to be run in parallel by Dask.
    
    Each task creates its own NasDB connection to avoid SFTP channel conflicts
    when multiple tasks run in parallel.

    Args:
        config_path (str): Path to the config file for creating NasDB instance
        file_path (str): Path to the file to process
        args (argparse.Namespace): argparse.Namespace object containing the analysis parameters

    Returns:
        tuple: (file_path, df_processed, df_raw, stft_result, cwt_result)
            file_path (str): Path to the file that was processed
            df_processed (pd.DataFrame): DataFrame containing the processed data
            df_raw (pd.DataFrame): DataFrame containing the raw data (before refining/offset removal)
            stft_result (dict): Dictionary containing the STFT results
            cwt_result (dict): Dictionary containing the CWT results
    """
    nas_instance = NasDB(config_path=config_path)
    file_basename = Path(file_path).name
    logging.info(f"{log_tag('ANALY','LOAD')} Starting processing for: {file_basename} @{file_path}")

    try:
        file_basename, df_raw = load_single_file_raw_data(nas_instance, file_path, args)
        if df_raw is None:
            return None, None, None, None, None

        df_processed = refine_and_preprocess_signal(df_raw, file_path, args)
        stft_result = compute_stft_results(file_basename, df_processed, args)
        cwt_result = compute_cwt_results(file_basename, df_processed, args)
        return file_path, df_processed, df_raw, stft_result, cwt_result
    finally:
        nas_instance.disconnect()


def run_analysis(
    query: int | str | list[int | str],
    args: argparse.Namespace,
    nas_db: NasDB,
    vest_db: VestDB,
) -> dict | None:
    """
    Performs a complete analysis workflow for a given query.

    Args:
        query (int | str | list[int | str]): Shot number or pattern to analyze
        args (argparse.Namespace): argparse.Namespace object containing the analysis parameters
        nas_db (NasDB): NasDB instance
        vest_db (VestDB): VestDB instance

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

    need_stft, need_cwt, need_density, requested_freqs = resolve_analysis_requirements(args)
    results_data_by_shot, shots_to_analyze = scan_cached_results_for_query(
        flat_list=flat_list,
        args=args,
        requested_freqs=requested_freqs,
        need_stft=need_stft,
        need_cwt=need_cwt,
        need_density=need_density,
    )

    if not shots_to_analyze and results_data_by_shot:
        logging.info(
            f"{log_tag('ANALY','RSLT')} All shots have complete results in results directory. "
            f"Returning cached results without re-analysis."
        )
        return build_cached_analysis_bundles(results_data_by_shot)

    if shots_to_analyze:
        analysis_query = build_analysis_query(flat_list, shots_to_analyze)
        target_files = nas_db.find_files(
            query=analysis_query,
            data_folders=args.data_folders,
            add_path=args.add_path,
            force_remote=args.force_remote,
        )
        if not target_files:
            logging.warning(
                "\n"
                + f"{log_tag('ANALY','RUN')} No files found for shots that need analysis. Skipping analysis."
                + "\n"
            )
            if results_data_by_shot:
                return build_cached_analysis_bundles(results_data_by_shot)
            return
    else:
        if results_data_by_shot:
            return build_cached_analysis_bundles(results_data_by_shot)
        return

    files_by_shot, interferometry_params_by_file = group_files_and_interferometry(target_files)
    vest_data_by_shot = load_vest_data_for_shots(flat_list, vest_db, args)
    dask_results = run_parallel_file_processing(
        target_files,
        args,
        load_and_process_file,
    )
    analysis_data, raw_data, stft_results, cwt_results = aggregate_dask_results(dask_results)

    all_analysis_bundles: dict[int, dict[str, Any]] = {}
    for shot_num in sorted(analysis_data.keys()):
        shot_nas_data = analysis_data[shot_num]
        shot_raw_data = raw_data.get(shot_num, {})
        shot_stft_data = stft_results.get(shot_num, {})
        shot_cwt_data = cwt_results.get(shot_num, {})
        logging.info(
            "\n"
            + f"{log_tag('ANALY','RUN')} Post-processing for Shot #{shot_num} with {len(shot_nas_data)} files"
            + "\n"
        )

        shot_files = files_by_shot.get(shot_num, [])
        shot_interferometry_params = collect_shot_interferometry_params(
            shot_files,
            interferometry_params_by_file,
        )
        freq_groups, freq_combined_signals = build_frequency_context_for_shot(
            shot_nas_data=shot_nas_data,
            shot_interferometry_params=shot_interferometry_params,
            requested_freqs=requested_freqs,
        )
        combined_signals = freq_combined_signals
        logging.info(
            f"{log_tag('ANALY','RUN')} Combined signals organized by frequency: {list(combined_signals.keys())} GHz"
        )

        current_vest_data = vest_data_by_shot.get(shot_num, {})
        vest_ip_data = resolve_vest_ip_data(current_vest_data)
        density_data = calculate_density_data_by_frequency(
            freq_combined_signals=freq_combined_signals,
            freq_groups=freq_groups,
            shot_num=shot_num,
            args=args,
            vest_ip_data=vest_ip_data,
        )

        all_analysis_bundles[shot_num] = {
            "shot_info": {"shot_num": shot_num},
            "raw_data": {"nas": shot_nas_data, "vest": current_vest_data},
            "processed_data": {"signals": combined_signals, "density": density_data},
            "analysis_results": {"stft": shot_stft_data, "cwt": shot_cwt_data},
            "interferometry_params": shot_interferometry_params,
        }

        plot_shot_outputs(
            shot_num=shot_num,
            args=args,
            shot_stft_data=shot_stft_data,
            shot_cwt_data=shot_cwt_data,
            combined_signals=combined_signals,
            density_data=density_data,
            vest_ip_data=vest_ip_data,
        )
        export_envelope_outputs(
            shot_num=shot_num,
            args=args,
            shot_nas_data=shot_nas_data,
            shot_interferometry_params=shot_interferometry_params,
        )
        save_shot_outputs(
            shot_num=shot_num,
            args=args,
            shot_raw_data=shot_raw_data,
            freq_groups=freq_groups,
            shot_interferometry_params=shot_interferometry_params,
            shot_stft_data=shot_stft_data,
            shot_cwt_data=shot_cwt_data,
            density_data=density_data,
            current_vest_data=current_vest_data,
        )

    logging.info(f"{log_tag('ANALY','RUN')} Full Analysis Finished")
    return merge_cached_results_with_bundles(
        all_analysis_bundles,
        results_data_by_shot,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for IFI analysis."""
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
        "--plot_envelope",
        action="store_true",
        help="Overlay spike-robust envelopes on waveform plots.",
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
        "--envelope",
        action="store_true",
        help="Export low-envelope probe segments as JSON files.",
    )
    parser.add_argument(
        "--freq",
        nargs="+",
        type=float,
        choices=[94.0, 280.0],
        default=None,
        help="Filter analysis to specific frequency groups. "
             "Options: 94.0 (for 94GHz), 280.0 (for 280GHz). "
             "Can specify both: --freq 94.0 280.0. "
             "If not specified, all frequencies are processed.",
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

    # Density plotting options
    parser.add_argument(
        "--color_density_by_amplitude",
        action="store_true",
        help="Color-code density plots by probe signal amplitude. "
             "Requires probe signal data to be available.",
    )
    parser.add_argument(
        "--amplitude_colormap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap for amplitude-based density coloring. "
             "Options: 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', etc. "
             "Default: 'coolwarm'.",
    )
    parser.add_argument(
        "--amplitude_impedance",
        type=float,
        default=50.0,
        help="System impedance in ohms [Ohm] for amplitude-to-dB conversion. "
             "Default: 50.0 Ohm.",
    )

    # Output flags
    parser.add_argument(
        "--save_plots", action="store_true", help="Save generated plots to files."
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save processed data to HDF5 files."
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="threads",
        choices=["threads", "processes", "single-threaded"],
        help="Dask scheduler to use for parallel processing.",
    )
    return parser


def main(argv: list[str] | None = None):
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

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # --- Initialize DB Controllers ---
    try:
        nas_db = NasDB(config_path="ifi/config.ini")
        vest_db = VestDB(config_path="ifi/config.ini")
    except FileNotFoundError:
        logging.error(f"{log_tag('ANALY','MAIN')} Configuration file 'ifi/config.ini' not found. Exiting.")
        return
    except Exception as e:
        logging.error(f"{log_tag('ANALY','MAIN')} Failed to initialize database controllers: {e}")
        return

    try:
        run_analysis(args.query, args, nas_db, vest_db)
    finally:
        nas_db.disconnect()
        vest_db.disconnect()


if __name__ == "__main__":
    main()


