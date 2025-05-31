"""
FEATURE PIPELINE - Comprehensive Leakage-Safe Features for Forecasting
======================================================================

Comprehensive feature pipeline for electricity consumption forecasting.
Creates leakage-safe features by avoiding direct consumption exposure.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from .consumption_features import (
    create_consumption_features, 
    create_consumption_patterns
)
from .temporal_features import create_all_temporal_features
from .weather_features import create_all_weather_features
import warnings

warnings.filterwarnings("ignore")


def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create leakage‐safe, comprehensive features for electricity forecasting.
    Components (in order):
      1) Temporal features
      2) Consumption‐pattern features (intermediate, then dropped)
      3) Weather features
      4) Time‐series features (lags, rolls)
      5) Leakage‐safe interactions (using lagged consumption)
      6) Drop any direct‐today consumption columns
    
    Args:
        df: Input DataFrame containing at least:
            - 'LCLid' (household ID)
            - 'day'   (date or datetime)
            - 'total_kwh' (raw daily usage) or half‐hourly columns for consumption
            - Weather columns: 'temperatureMax', 'temperatureMin', 'humidity', 'windSpeed', 'cloudCover'
            - Any temporal flags (if pre‐computed) or they will be created
    Returns:
        A new DataFrame with all leakage‐safe features added and raw consumption columns removed.
    """
    print("🚀 CREATING COMPREHENSIVE LEAKAGE-SAFE FEATURES")
    print("=" * 50)
    
    # 1. Temporal features (dayofweek, month, is_holiday, is_weekend, etc.)
    print("📅 Creating temporal features...")
    df = create_all_temporal_features(df)
    
    # 2. Consumption-pattern features (create features like peak_kwh, daily_variability, etc.)
    #    These are intermediate—some will be dropped below to avoid leakage.
    print("⚡ Creating consumption pattern features (intermediate)...")
    df = create_consumption_features(df)
    df = create_consumption_patterns(df)
    
    # 3. Weather features (temp_avg, heating_degree_days, cooling_degree_days, etc.)
    print("🌤️ Creating weather features...")
    df = create_all_weather_features(df)
    
    # 4. Time-series features (lags and rolling windows)
    #    Must shift by 1 (or more) to avoid using total_kwh[t] directly.
    print("📈 Creating time-series features (lags and rolling windows)...")
    df = create_timeseries_features_safe(df, target_col="total_kwh", lags=[1, 7, 14], windows=[7, 14])

    # 5. Leakage‐safe interaction features (all referencing lag1_total)
    print("🔗 Creating leakage-safe interaction features...")
    eps = 1e-6
    if {"is_weekend", "heating_degree_days", "lag1_total"}.issubset(df.columns):
        df["lag1_weekend_heating"] = (
            df["is_weekend"] 
            * df["heating_degree_days"] 
            * (df["lag1_total"] / (df["lag1_total"] + eps))
        ).fillna(0)

    if {"is_holiday", "lag1_total"}.issubset(df.columns):
        df["lag1_holiday_consumption"] = (df["is_holiday"] * df["lag1_total"]).fillna(0)

    if {"is_summer", "cooling_degree_days", "lag1_total"}.issubset(df.columns):
        df["lag1_summer_cooling"] = (
            df["is_summer"]
            * df["cooling_degree_days"]
            * (df["lag1_total"] / (df["lag1_total"] + eps))
        ).fillna(0)

    # 6. Drop any columns that directly expose today's consumption or raw half-hourly readings
    print("🧹 Removing leakage-prone features...")
    to_drop = []
    # Direct consumption columns
    forbidden_prefixes = [
        "total_kwh", "mean_kwh", "std_kwh", "peak_kwh", "min_kwh",
        "morning_kwh", "afternoon_kwh", "evening_kwh", "night_kwh",
        "peak_period_kwh", "off_peak_kwh", "base_load", "load_factor",
        "daily_variability", "coefficient_of_variation",
        "usage_concentration", "peak_sharpness",
        "peak_to_mean_ratio", "peak_to_total_ratio", "day_night_ratio",
        "holiday_consumption_boost", "base_load_ratio", "consumption_sharpness"
    ]
    for col in df.columns:
        for prefix in forbidden_prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                to_drop.append(col)
        # Raw half-hourly columns, e.g. 'hh_0' ... 'hh_47'
        if col.startswith("hh_") and col.replace("hh_", "").isdigit():
            to_drop.append(col)

    dropped_cols = list(set(to_drop))
    df = df.drop(columns=dropped_cols, errors="ignore")
    
    print(f"   🚫 Dropped {len(dropped_cols)} leakage-prone columns")
    print("✅ COMPREHENSIVE LEAKAGE-SAFE FEATURES CREATED")
    print(f"📊 Final shape: {df.shape}")
    
    return df


def create_timeseries_features_safe(
    df: pd.DataFrame,
    target_col: str,
    lags: list,
    windows: list
) -> pd.DataFrame:
    """
    Create leakage‐safe time-series features: lags and rolling means.
    Each feature for day t references only total_kwh[t - k].
    
    Args:
        df: DataFrame containing 'LCLid', 'day', and target_col (total_kwh).
        target_col: Name of the raw consumption column, e.g. 'total_kwh'.
        lags: List of integer lag days (e.g. [1, 7, 14]).
        windows: List of integer rolling window sizes (e.g. [7, 14]).
    Returns:
        df with new columns:
            - 'lag{lag}_total' for each lag
            - 'roll{window}_total_mean' for each window (computed on shifted series)
            - Optionally pct_change and delta fields
    """
    df = df.sort_values(["LCLid", "day"])
    
    # Create lag features
    for lag in lags:
        col_name = f"lag{lag}_total"
        df[col_name] = df.groupby("LCLid")[target_col].shift(lag)

    # Create rolling-window features on shifted target (shift by 1 to exclude current day)
    for window in windows:
        col_name = f"roll{window}_total_mean"
        shifted = df.groupby("LCLid")[target_col].shift(1)
        df[col_name] = (
            shifted
            .groupby(df["LCLid"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Optional: percent change and delta from previous lag (e.g. lag1 vs lag2)
    if 1 in lags:
        df["lag2_total"] = df.groupby("LCLid")[target_col].shift(2)
        df["delta1_total"] = (df["lag1_total"] - df["lag2_total"]).fillna(0)
        df["pct_change1_total"] = np.where(
            df["lag2_total"].abs() > 0,
            (df["lag1_total"] - df["lag2_total"]) / (df["lag2_total"] + 1e-6),
            0
        )
        df = df.drop(columns=["lag2_total"], errors="ignore")

    return df


def get_forecasting_features(df: pd.DataFrame, target_col: str = None) -> list:
    """
    Return a leakage‐safe list of predictor column names.
    Drops any column equal to target_col, and excludes:
        - Columns starting with 'LCLid', 'day', or 'hh_'
    
    Args:
        df: Input DataFrame including features and possibly target_col.
        target_col: Name of the target column to drop (e.g. 'total_kwh' or 'label_7').
        
    Returns:
        feature_cols: List of column names that can be used as model inputs.
    """
    temp_df = df.copy()
    if target_col is not None and target_col in temp_df.columns:
        temp_df = temp_df.drop(columns=[target_col])

    exclude_prefixes = ["LCLid", "day", "hh_"]
    feature_cols = [
        col for col in temp_df.columns
        if not any(col.startswith(pref) for pref in exclude_prefixes)
    ]

    print(f"📊 Selected {len(feature_cols)} forecasting features")
    return feature_cols


def get_forecasting_feature_groups(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Organize predictor columns into logical categories for interpretation.
    Drops target_col, then groups by substring patterns.
    
    Categories and matching substrings:
      - 'temporal': ["dayofweek", "month", "is_holiday", "is_weekend", "season", "quarter"]
      - 'weather': ["temp", "heating", "cooling", "humidity", "wind", "cloud"]
      - 'time_series': ["lag", "roll", "delta", "pct_change", "weekly"]
      - 'consumption_patterns': ["ratio", "variability", "concentration", "sharpness"]
      - 'household_group': ["acorn", "hh_avg", "hh_std", "hh_max", "hh_min", "relative_variability"]
      - 'interactions': ["lag1_weekend_heating", "lag1_holiday_consumption", "lag1_summer_cooling"]
    
    Args:
        df: DataFrame containing predictor columns and possibly target_col.
        target_col: Name of the target column to drop first.
    
    Returns:
        feature_groups: Dict mapping category names to lists of column names.
    """
    temp_df = df.copy()
    if target_col is not None and target_col in temp_df.columns:
        temp_df = temp_df.drop(columns=[target_col])

    exclude_prefixes = ["LCLid", "day", "hh_"]
    all_predictors = [
        col for col in temp_df.columns
        if not any(col.startswith(pref) for pref in exclude_prefixes)
    ]

    feature_groups = {
        "temporal": [
            col for col in all_predictors
            if any(substr in col for substr in ["dayofweek", "month", "is_holiday", "is_weekend", "season", "quarter"])
        ],
        "weather": [
            col for col in all_predictors
            if any(substr in col for substr in ["temp", "heating", "cooling", "humidity", "wind", "cloud"])
        ],
        "time_series": [
            col for col in all_predictors
            if any(substr in col for substr in ["lag", "roll", "delta", "pct_change", "weekly"])
        ],
        "consumption_patterns": [
            col for col in all_predictors
            if any(substr in col for substr in ["ratio", "variability", "concentration", "sharpness"])
        ],
        "household_group": [
            col for col in all_predictors
            if any(substr in col for substr in ["acorn", "hh_avg", "hh_std", "hh_max", "hh_min", "relative_variability"])
        ],
        "interactions": [
            col for col in all_predictors
            if any(substr in col for substr in ["lag1_weekend_heating", "lag1_holiday_consumption", "lag1_summer_cooling"])
        ]
    }

    return feature_groups


def add_group_and_household_features(train_df, test_df):
    """
    Add group-level (ACORN) and household-level features in a leakage-safe way.
    Compute stats on train only, merge into both train and test.
    """
    print("🏠 Adding group and household features...")
    
    # ACORN group features
    if 'Acorn_grouped' in train_df.columns:
        print("   📍 Adding ACORN group features...")
        # Use lag1_total instead of total_kwh to avoid leakage
        target_for_stats = "lag1_total" if "lag1_total" in train_df.columns else "total_kwh"
        
        acorn_means = train_df.groupby("Acorn_grouped")[target_for_stats].mean().rename("acorn_avg_consumption")
        train_df = train_df.merge(acorn_means, on="Acorn_grouped", how="left")
        test_df = test_df.merge(acorn_means, on="Acorn_grouped", how="left")
        
        if target_for_stats in train_df.columns:
            train_df["acorn_consumption_ratio"] = train_df[target_for_stats] / (train_df["acorn_avg_consumption"] + 1e-6)
            test_df["acorn_consumption_ratio"] = test_df[target_for_stats] / (test_df["acorn_avg_consumption"] + 1e-6)
    
    # Household-level features
    if 'LCLid' in train_df.columns:
        print("   🏡 Adding household features...")
        # Use lag1_total instead of total_kwh to avoid leakage
        target_for_stats = "lag1_total" if "lag1_total" in train_df.columns else "total_kwh"
        
        hh_stats = train_df.groupby("LCLid")[target_for_stats].agg([
            ("hh_avg_consumption", "mean"),
            ("hh_std_consumption", "std"),
            ("hh_max_consumption", "max"),
            ("hh_min_consumption", "min")
        ]).reset_index()
        
        train_df = train_df.merge(hh_stats, on="LCLid", how="left")
        test_df = test_df.merge(hh_stats, on="LCLid", how="left")
        
        if target_for_stats in train_df.columns:
            train_df["daily_vs_hh_avg"] = train_df[target_for_stats] / (train_df["hh_avg_consumption"] + 1e-6)
            test_df["daily_vs_hh_avg"] = test_df[target_for_stats] / (test_df["hh_avg_consumption"] + 1e-6)
            train_df["daily_vs_hh_max"] = train_df[target_for_stats] / (train_df["hh_max_consumption"] + 1e-6)
            test_df["daily_vs_hh_max"] = test_df[target_for_stats] / (test_df["hh_max_consumption"] + 1e-6)
    
    return train_df, test_df


if __name__ == "__main__":
    print("🔧 Feature Pipeline - Comprehensive Leakage-Safe Features for Forecasting")
    print("Usage: from src.features.feature_pipeline import create_comprehensive_features") 