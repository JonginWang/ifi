#!/usr/bin/env python3
"""
Data Processing Functions
=========================

This module contains functions for cleaning, refining, and transforming
the raw interferometer data.

Functions:
    refine_data: Applies a series of cleaning and refining steps to the raw DataFrame.
    remove_offset: Removes the low-frequency offset from data channels using a moving average.
"""

import pandas as pd
import numpy as np
try:
    from ..utils.common import LogManager, log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, log_tag

logger = LogManager().get_logger(__name__)


def refine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a series of cleaning and refining steps to the raw DataFrame.

    This function rounds all numeric data to 10 decimal places and reconstructs the 'TIME' column to be a perfect arithmetic sequence.
    It also removes any rows containing NaN values.

    Args:
        df(pandas.DataFrame): The raw input DataFrame.

    Returns:
        pandas.DataFrame: The refined DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning(
            f"{log_tag('PROCS', 'REFIN')} refine_data called with non-DataFrame input. Skipping."
        )
        return df

    refined_df = df.copy()

    # 1. Round all numeric columns to 10 decimal places
    numeric_cols = refined_df.select_dtypes(include=np.number).columns
    refined_df[numeric_cols] = refined_df[numeric_cols].round(10)
    logger.info(
        f"{log_tag('PROCS', 'REFIN')} Rounded numeric data to 10 decimal places."
    )

    # 2. Reconstruct TIME column
    if "TIME" in refined_df.columns and len(refined_df["TIME"]) > 1:
        # Calculate time resolution from the first few differences for stability
        time_diffs = refined_df["TIME"].diff().dropna()
        if not time_diffs.empty:
            time_resolution = time_diffs.head(100).mean()
            start_time = refined_df["TIME"].iloc[0]
            num_points = len(refined_df)

            new_time_col = np.linspace(
                start=start_time,
                stop=start_time + time_resolution * (num_points - 1),
                num=num_points,
            )
            refined_df["TIME"] = new_time_col
            logger.info(
                f"{log_tag('PROCS', 'REFIN')} Reconstructed 'TIME' column with resolution: {time_resolution:.4e} s"
            )
        else:
            logger.warning(
                f"{log_tag('PROCS', 'REFIN')} Could not determine time resolution. Skipping TIME reconstruction."
            )

    # 3. Drop rows with NaN values
    initial_rows = len(refined_df)
    refined_df.dropna(inplace=True)
    final_rows = len(refined_df)

    if initial_rows > final_rows:
        logger.info(
            f"{log_tag('PROCS', 'REFIN')} Dropped {initial_rows - final_rows} rows with NaN values."
        )

    return refined_df


def remove_offset(df: pd.DataFrame, window_size: int = 2001) -> pd.DataFrame:
    """
    Removes the low-frequency offset from data channels using a moving average.

    Args:
        df(pandas.DataFrame): The input DataFrame, expected to have a 'TIME' column and data channels.
        window_size(int): The size of the moving average window. Must be an odd integer.

    Returns:
        pandas.DataFrame: A new DataFrame with the offset removed from data channels.
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning(
            f"{log_tag('PROCS', 'RMOFF')} remove_offset called with non-DataFrame input. Skipping."
        )
        return df

    # Skip offset removal for FPGA data (.dat files)
    if df.attrs.get("source_file_type") == "dat":
        logger.info(
            f"{log_tag('PROCS', 'RMOFF')} Skipping offset removal for FPGA data file."
        )
        return df.copy()

    # If window_size is negative, use default.
    if window_size < 0:
        logger.warning(
            f"{log_tag('PROCS', 'RMOFF')} Negative window_size ({window_size}) is invalid. Using default 2001."
        )
        window_size = 2001

    # Dynamically adjust window size for smaller datasets
    num_rows = len(df)
    if 0 < num_rows < 1_000_000:
        dynamic_window_size = int(np.floor(num_rows / 5000))
        # Ensure it's not zero and is odd
        if dynamic_window_size == 0:
            dynamic_window_size = 1
        if dynamic_window_size % 2 == 0:
            dynamic_window_size += 1

        # Only use the dynamic size if it's smaller than the user-provided/default size
        if dynamic_window_size < window_size:
            window_size = dynamic_window_size
            logger.info(
                f"{log_tag('PROCS', 'RMOFF')} Dynamically adjusted window size to {window_size} for smaller dataset ({num_rows} rows)."
            )

    if window_size <= 0:
        logger.info(
            f"{log_tag('PROCS', 'RMOFF')} Offset removal skipped as window_size is zero or negative."
        )
        return df.copy()

    if window_size % 2 == 0:
        window_size += 1
        logger.warning(
            f"{log_tag('PROCS', 'RMOFF')} Window size must be odd. Adjusting to {window_size}."
        )

    offset_removed_df = df.copy()
    initial_rows = len(offset_removed_df)
    data_cols = [col for col in df.columns if col != "TIME"]

    for col in data_cols:
        moving_avg = (
            offset_removed_df[col]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        offset_removed_df[col] = offset_removed_df[col] - moving_avg

    # The rolling operation with center=True will produce NaNs at the edges, so drop them.
    offset_removed_df.dropna(inplace=True)
    rows_removed = initial_rows - len(offset_removed_df)

    logger.info(
        f"{log_tag('PROCS', 'RMOFF')} Removed offset using a {window_size}-point moving average. ({rows_removed} rows removed at edges)"
    )

    return offset_removed_df
