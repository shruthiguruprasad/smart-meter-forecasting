"""
DATA SPLITTERS - Leakage-Safe Train/Validation/Test Splits for Forecasting
==========================================================================

Data splitting functions for day-ahead and week-ahead electricity consumption forecasting.
Handles chronological splits and leakage-safe feature addition.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
from .feature_pipeline import (
    get_forecasting_features,
    get_forecasting_feature_groups,
    add_group_and_household_features
)
import warnings

warnings.filterwarnings("ignore")


def prepare_forecasting_data(
    df: pd.DataFrame,
    target_col: str = "total_kwh",
    test_days: int = 90,
    val_days: int = 30
) -> tuple:
    """
    Prepare data for day‐ahead forecasting (h=1) in a leakage‐safe manner.
    Steps:
      1) Ensure 'day' is datetime, sort by 'day'
      2) Compute test_start = max(day) - test_days, val_start = test_start - val_days
      3) Split into train_df, val_df, test_df
      4) Merge group/household features from train_df into val_df/test_df
      5) Compute final predictor list and feature groups
    
    Args:
        df: DataFrame after create_comprehensive_features (must still contain 'total_kwh').
        target_col: Name of the target column ('total_kwh').
        test_days: Number of most recent days for test set.
        val_days: Number of days before test for validation set.
    
    Returns:
        train_df, val_df, test_df: Split DataFrames (with group/household features added).
        feature_cols: List of predictor column names.
        target_col: Echoed target column name.
        feature_groups: Dict of category→list of feature columns.
    """
    print("📊 Preparing data for day-ahead forecasting (h=1)...")
    
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values("day")

    # Initial feature scan (still includes target_col)
    initial_features = get_forecasting_features(df, target_col=target_col)
    feature_groups_initial = get_forecasting_feature_groups(df, target_col=target_col)

    # Determine date splits
    max_date = df["day"].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["day"] < val_start].copy()
    val_df = df[(df["day"] >= val_start) & (df["day"] < test_start)].copy()
    test_df = df[df["day"] >= test_start].copy()

    # Add group/household features (compute on train only)
    train_df, merged_valtest = add_group_and_household_features(
        train_df, pd.concat([val_df, test_df], ignore_index=True)
    )
    # Re-split merged_valtest into val_df and test_df
    val_df = merged_valtest[merged_valtest["day"] < test_start].copy()
    test_df = merged_valtest[merged_valtest["day"] >= test_start].copy()

    # Final predictor list (drops target_col from train_df)
    feature_cols = get_forecasting_features(train_df, target_col=target_col)
    feature_groups = get_forecasting_feature_groups(train_df, target_col=target_col)

    print(f"   ✅ Initial features: {len(initial_features)}")
    print(f"   ✅ Final features: {len(feature_cols)} (+{len(feature_cols) - len(initial_features)} household/group)")
    print(f"   ✅ Train rows: {len(train_df):,}  (Households: {train_df['LCLid'].nunique()})")
    print(f"   ✅ Val rows:   {len(val_df):,}  (Households: {val_df['LCLid'].nunique()})")
    print(f"   ✅ Test rows:  {len(test_df):,}  (Households: {test_df['LCLid'].nunique()})")
    print(f"   ✅ Train period: {train_df['day'].min().date()} to {train_df['day'].max().date()}")
    print(f"   ✅ Val period:   {val_df['day'].min().date()} to {val_df['day'].max().date()}")
    print(f"   ✅ Test period:  {test_df['day'].min().date()} to {test_df['day'].max().date()}")
    print(f"   ✅ Features:      {len(feature_cols)} columns")
    print(f"   ✅ Feature groups: {len(feature_groups)} groups")
    print(f"   ✅ Target:       {target_col}")

    return train_df, val_df, test_df, feature_cols, target_col, feature_groups


def prepare_weekahead_data(
    df: pd.DataFrame,
    test_days: int = 90,
    val_days: int = 30
) -> tuple:
    """
    Prepare data for week‐ahead forecasting (h=7) in a leakage‐safe manner.
    Steps:
      1) Sort by 'LCLid', 'day' and create 'label_7' = total_kwh.shift(-7)
      2) Drop rows where 'label_7' is NaN (last 7 days per household)
      3) Drop 'total_kwh'
      4) Compute test_start = max(day) - test_days, val_start = test_start - val_days
      5) Split into train_df, val_df, test_df
      6) Merge group/household features
      7) Compute final predictor list and feature groups
    
    Args:
        df: DataFrame after create_comprehensive_features (must contain 'total_kwh').
        test_days: Number of most recent days (post-shift) for test set.
        val_days: Number of days before test for validation.
    
    Returns:
        train_df, val_df, test_df: Split DataFrames with group/household features.
        feature_cols: List of predictor columns (drops 'label_7').
        "label_7": The name of the week‐ahead target.
        feature_groups: Dict of category→list of features.
    """
    print("📊 Preparing data for week-ahead forecasting (h=7)...")
    
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values(["LCLid", "day"])

    # Create week-ahead label
    print("   🔄 Creating week-ahead labels (7-day shift)...")
    df["label_7"] = df.groupby("LCLid")["total_kwh"].shift(-7)
    
    # Drop rows where we can't compute the 7-day ahead target
    initial_rows = len(df)
    df = df[df["label_7"].notna()].copy()
    print(f"   📉 Dropped {initial_rows - len(df):,} rows due to missing 7-day ahead targets")

    # Drop raw target to avoid leakage
    df = df.drop(columns=["total_kwh"], errors="ignore")

    # Determine date splits on shifted DataFrame
    df = df.sort_values("day")
    max_date = df["day"].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["day"] < val_start].copy()
    val_df = df[(df["day"] >= val_start) & (df["day"] < test_start)].copy()
    test_df = df[df["day"] >= test_start].copy()

    # Add group/household features (compute on train only)
    train_df, merged_valtest = add_group_and_household_features(
        train_df, pd.concat([val_df, test_df], ignore_index=True)
    )
    # Re-split merged_valtest into val_df and test_df
    val_df = merged_valtest[merged_valtest["day"] < test_start].copy()
    test_df = merged_valtest[merged_valtest["day"] >= test_start].copy()

    # Final predictor list (drops 'label_7')
    feature_cols = get_forecasting_features(train_df, target_col="label_7")
    feature_groups = get_forecasting_feature_groups(train_df, target_col="label_7")

    print("✅ Week‐ahead data prepared in a leakage‐safe manner")
    print(f"   ✅ Train rows: {len(train_df):,}  (Households: {train_df['LCLid'].nunique()})")
    print(f"   ✅ Val rows:   {len(val_df):,}  (Households: {val_df['LCLid'].nunique()})")
    print(f"   ✅ Test rows:  {len(test_df):,}  (Households: {test_df['LCLid'].nunique()})")
    print(f"   ✅ Train period: {train_df['day'].min().date()} to {train_df['day'].max().date()}")
    print(f"   ✅ Val period:   {val_df['day'].min().date()} to {val_df['day'].max().date()}")
    print(f"   ✅ Test period:  {test_df['day'].min().date()} to {test_df['day'].max().date()}")
    print(f"   ✅ Features:    {len(feature_cols)}")
    print(f"   ✅ Label:       'label_7'")

    return train_df, val_df, test_df, feature_cols, "label_7", feature_groups


if __name__ == "__main__":
    print("🔧 Data Splitters - Leakage-Safe Train/Val/Test Splits for Forecasting")
    print("Usage:")
    print("  from src.data.splitters import prepare_forecasting_data, prepare_weekahead_data") 