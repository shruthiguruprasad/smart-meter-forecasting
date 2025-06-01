"""
src/utils/sequence_builder.py

Utility functions to convert a feature‐engineered DataFrame into sliding‐window
(“sequence”) inputs for LSTM training. Supports both global (all households)
and per‐household sequence construction. Handles optional household_code alignment.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    group_col: str = "LCLid",
    include_household: bool = False,
    household_col: str = "household_code"
):
    """
    Build sliding‐window sequences for LSTM input from a long DataFrame sorted by group and time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data. Must be sorted by [group_col, date_col].
    feature_cols : list of str
        List of feature column names (e.g., lag_*, weather, calendar flags, etc.).
    target_col : str
        Name of the target column (e.g., 'label_1' or 'label_7').
    seq_len : int
        Number of consecutive time steps per sequence (e.g., 14).
    group_col : str, default "LCLid"
        Column name indicating household or group identifier.
    include_household : bool, default False
        If True, also return an array of household codes aligned to each sequence.
    household_col : str, default "household_code"
        Column name containing integer‐coded household IDs (0..n_households-1).

    Returns
    -------
    X : np.ndarray, shape (N_samples, seq_len, n_features)
        Sequence input array.
    y : np.ndarray, shape (N_samples,)
        Target values corresponding to each sequence.
    hh_codes : np.ndarray, shape (N_samples,), optional
        Household codes aligned to each sequence (only if include_household=True).

    Notes
    -----
    - The DataFrame df should be sorted first by group_col, then by date (chronological).
    - This function will only build sequences where all seq_len steps and the target step exist.
    - If any NaNs appear in the features or target within the window, that sequence is skipped.
    - If include_household=True, hh_codes[i] will be the household_code at the final time step of the i-th sequence.
    """
    X_list = []
    y_list = []
    hh_list = []

    # Ensure correct order: group_col, then date
    # (Assumes df already sorted by date; otherwise specify date_col sorting externally)
    # Example: df.sort_values([group_col, "day"], inplace=True)

    # Group by each household (LCLid)
    for hh, group_df in df.groupby(group_col):
        # Work only with one household’s chronology
        group_df = group_df.reset_index(drop=True)

        # If including household codes, extract that series
        if include_household:
            hh_series = group_df[household_col].values

        # Extract feature array and target array for this household
        features_array = group_df[feature_cols].values  # shape: (T, n_features)
        target_array = group_df[target_col].values      # shape: (T,)

        # Slide window over time steps
        for i in range(len(group_df) - seq_len):
            window_feats = features_array[i : i + seq_len]
            window_target = target_array[i + seq_len]  # one‐step ahead (for label_1) or 7‐ahead (for label_7)

            # Skip if any NaNs in window_feats or window_target
            if np.isnan(window_feats).any() or np.isnan(window_target):
                continue

            X_list.append(window_feats)
            y_list.append(window_target)
            if include_household:
                # Align hh code to the target step (i + seq_len)
                hh_list.append(hh_series[i + seq_len])

    X = np.stack(X_list, axis=0)  # (N, seq_len, n_features)
    y = np.array(y_list, dtype=float)  # (N,)

    if include_household:
        hh_codes = np.array(hh_list, dtype=int)  # (N,)
        return X, y, hh_codes

    return X, y


def build_global_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    group_col: str = "LCLid",
    household_col: str = "household_code"
):
    """
    Convenience function to build sequences for train/val/test splits all at once.
    Returns X_train, y_train, X_val, y_val, X_test, y_test (and optionally household codes).

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Pre‐split DataFrames. Each should be sorted by [group_col, date].
    feature_cols : list of str
        List of feature column names.
    target_col : str
        Name of the target column.
    seq_len : int
        Sequence length.
    group_col : str, default "LCLid"
        Household identifier column.
    household_col : str, default "household_code"
        Integer‐coded household ID column.

    Returns
    -------
    (X_train, y_train, hh_train,
     X_val,   y_val,   hh_val,
     X_test,  y_test,  hh_test)
    where each X_* is shape (N_*, seq_len, n_features),
          each y_* is shape (N_*,),
          each hh_* is shape (N_*,) of integer codes.
    """
    # Build sequences with household codes for each split
    X_train, y_train, hh_train = build_sequences(
        train_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col
    )
    X_val, y_val, hh_val = build_sequences(
        val_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col
    )
    X_test, y_test, hh_test = build_sequences(
        test_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col
    )

    return (X_train, y_train, hh_train,
            X_val,   y_val,   hh_val,
            X_test,  y_test,  hh_test)
