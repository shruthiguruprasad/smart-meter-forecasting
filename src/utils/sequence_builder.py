"""
src/utils/sequence_builder.py

Utility functions to convert a feature‐engineered DataFrame into sliding‐window
(“sequence”) inputs for LSTM training. Supports both global (all households)
and per‐household sequence construction. Handles optional household_code alignment,
and can optionally return the target dates for plotting.

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

    for hh, group_df in df.groupby(group_col):
        group_df = group_df.reset_index(drop=True)

        if include_household:
            hh_series = group_df[household_col].values

        features_array = group_df[feature_cols].values
        target_array = group_df[target_col].values

        for i in range(len(group_df) - seq_len):
            window_feats = features_array[i : i + seq_len]
            window_target = target_array[i + seq_len]

            if np.isnan(window_feats).any() or np.isnan(window_target):
                continue

            X_list.append(window_feats)
            y_list.append(window_target)
            if include_household:
                hh_list.append(hh_series[i + seq_len])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=float)

    if include_household:
        hh_codes = np.array(hh_list, dtype=int)
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
    Returns X_train, y_train, hh_train, X_val, y_val, hh_val, X_test, y_test, hh_test.

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
    """
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


def build_sequences_with_dates(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    group_col: str = "LCLid",
    include_household: bool = False,
    household_col: str = "household_code",
    date_col: str = "day"
):
    """
    Like build_sequences, but also returns the target date for each window.

    Returns:
      X : np.ndarray (N, seq_len, n_features)
      y : np.ndarray (N,)
      hh_codes : np.ndarray (N,)  # only if include_household=True
      dates  : np.ndarray (N,)    # the 'day' corresponding to the target for each sequence
    """
    X_list = []
    y_list = []
    hh_list = []
    date_list = []

    for hh, group_df in df.groupby(group_col):
        group_df = group_df.reset_index(drop=True)

        if include_household:
            hh_series = group_df[household_col].values

        features_array = group_df[feature_cols].values
        target_array = group_df[target_col].values
        date_array = pd.to_datetime(group_df[date_col]).values

        for i in range(len(group_df) - seq_len):
            window_feats = features_array[i : i + seq_len]
            window_target = target_array[i + seq_len]
            window_date = date_array[i + seq_len]

            if np.isnan(window_feats).any() or np.isnan(window_target):
                continue

            X_list.append(window_feats)
            y_list.append(window_target)
            date_list.append(window_date)

            if include_household:
                hh_list.append(hh_series[i + seq_len])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=float)
    dates = np.array(date_list, dtype="datetime64[ns]")

    if include_household:
        hh_codes = np.array(hh_list, dtype=int)
        return X, y, hh_codes, dates

    return X, y, dates


def build_global_sequences_with_dates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    seq_len: int,
    group_col: str = "LCLid",
    household_col: str = "household_code",
    date_col: str = "day"
):
    """
    Calls build_sequences_with_dates on train/val/test and returns
    (X_train, y_train, hh_train, date_train,
     X_val,   y_val,   hh_val,   date_val,
     X_test,  y_test,  hh_test,  date_test)
    """
    X_train, y_train, hh_train, date_train = build_sequences_with_dates(
        train_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col,
        date_col=date_col
    )
    X_val, y_val, hh_val, date_val = build_sequences_with_dates(
        val_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col,
        date_col=date_col
    )
    X_test, y_test, hh_test, date_test = build_sequences_with_dates(
        test_df, feature_cols, target_col, seq_len,
        group_col=group_col,
        include_household=True,
        household_col=household_col,
        date_col=date_col
    )

    return (
        X_train, y_train, hh_train, date_train,
        X_val,   y_val,   hh_val,   date_val,
        X_test,  y_test,  hh_test,  date_test
    )
