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
    from ..utils.log_manager import LogManager, log_tag
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.log_manager import LogManager, log_tag

logger = LogManager().get_logger(__name__)


def _interpolate_signal_non_finite(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate non-finite gaps for numeric signal columns while preserving TIME."""
    out = df.copy()
    numeric_cols = [col for col in out.select_dtypes(include=np.number).columns if col != "TIME"]
    if not numeric_cols:
        return out

    if "TIME" in out.columns and len(out["TIME"]) > 1:
        time_index = pd.Index(out["TIME"].to_numpy(), name="TIME")
        interpolated = out[numeric_cols].copy()
        interpolated.index = time_index
        interpolated = interpolated.interpolate(
            method="index",
            limit_direction="both",
        )
        out[numeric_cols] = interpolated.to_numpy()
        return out

    out[numeric_cols] = out[numeric_cols].interpolate(
        method="linear",
        limit_direction="both",
        axis=0,
    )
    return out


def refine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a series of cleaning and refining steps to the raw DataFrame.

    This function rounds numeric data, converts `Inf/-Inf` into `NaN`, reconstructs
    the `TIME` column when possible, and drops columns that remain invalid.

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

    # 2. Convert non-finite numeric values into NaN so downstream cleanup handles
    # both NaN and Inf consistently.
    if len(numeric_cols) > 0:
        finite_mask = np.isfinite(refined_df[numeric_cols])
        non_finite_count = int((~finite_mask).to_numpy().sum())
        if non_finite_count:
            refined_df[numeric_cols] = refined_df[numeric_cols].where(finite_mask, np.nan)
            logger.warning(
                f"{log_tag('PROCS', 'REFIN')} Replaced {non_finite_count} non-finite "
                f"numeric value(s) with NaN."
            )

    # Preserve TIME as a valid axis by dropping rows that have invalid time stamps.
    dropped_time_rows = 0
    if "TIME" in refined_df.columns:
        invalid_time_mask = refined_df["TIME"].isna()
        dropped_time_rows = int(invalid_time_mask.sum())
        if dropped_time_rows:
            refined_df = refined_df.loc[~invalid_time_mask].copy()
            logger.warning(
                f"{log_tag('PROCS', 'REFIN')} Dropped {dropped_time_rows} row(s) with "
                "invalid TIME values."
            )

    # 3. Interpolate signal non-finite values after TIME cleanup.
    refined_df = _interpolate_signal_non_finite(refined_df)

    # 4. Reconstruct TIME column
    if "TIME" in refined_df.columns and len(refined_df["TIME"]) > 1:
        # Calculate time resolution from the first few differences for stability
        time_diffs = refined_df["TIME"].diff().replace([np.inf, -np.inf], np.nan).dropna()
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

    # 5. Drop columns that remain invalid after refinement.
    initial_rows = len(refined_df)
    if refined_df.shape[1] > 4:
        col_to_drop = []
        for col in refined_df.columns[4:]:
            if refined_df[col].isna().sum() > initial_rows // 2:
                col_to_drop.append(col)
        refined_df.drop(columns=col_to_drop, inplace=True)
        logger.info(
            f"{log_tag('PROCS', 'REFIN')} Dropped {col_to_drop} with more than half of its rows are NaN. Shape: {refined_df.shape}."
        )
    refined_df.dropna(axis=1, inplace=True)
    final_rows = len(refined_df)

    if initial_rows > final_rows or dropped_time_rows:
        logger.info(
            f"{log_tag('PROCS', 'REFIN')} Dropped {initial_rows - final_rows} row(s) "
            "during refinement cleanup."
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
