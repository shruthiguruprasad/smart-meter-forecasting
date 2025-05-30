"""
⚡ CONSUMPTION FEATURES - Enhanced Consumption Features for Forecasting
======================================================================

Enhanced consumption features for electricity consumption forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')

def create_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create essential consumption features for forecasting
    
    Args:
        df: Dataframe with hh_0 to hh_47 columns
        
    Returns:
        Dataframe with consumption features added
    """
    print("⚡ Creating consumption features...")
    
    # Define half-hour columns
    hh_cols = [f"hh_{i}" for i in range(48)]
    hh_data = df[hh_cols].to_numpy()
    
    # Basic consumption stats
    df["total_kwh"] = hh_data.sum(axis=1)
    df["mean_kwh"] = hh_data.mean(axis=1)
    df["std_kwh"] = hh_data.std(axis=1)
    df["peak_kwh"] = np.nanmax(hh_data, axis=1)
    df["peak_hour"] = np.nanargmax(hh_data, axis=1)
    df["min_kwh"] = np.nanmin(hh_data, axis=1)
    
    # Time-of-day consumption (essential for forecasting)
    df["morning_kwh"] = hh_data[:, 6:12].sum(axis=1)       # 3-6 AM
    df["afternoon_kwh"] = hh_data[:, 12:18].sum(axis=1)    # 6-9 AM  
    df["evening_kwh"] = hh_data[:, 18:24].sum(axis=1)      # 9 AM-12 PM
    night_idx = np.r_[0:6, 24:48]                          # Night hours
    df["night_kwh"] = hh_data[:, night_idx].sum(axis=1)
    
    # Peak period consumption (UK specific)
    df["peak_period_kwh"] = hh_data[:, 34:40].sum(axis=1)  # 5-8 PM peak
    df["off_peak_kwh"] = hh_data[:, 0:12].sum(axis=1)      # Midnight-6 AM
    
    # Peak characteristics (important for forecasting)
    if "dayofweek" in df.columns:
        df["is_weekday_peak"] = (
            (df["dayofweek"] < 5) &
            df["peak_hour"].isin([17, 18, 19, 20])
        ).astype(int)
    
    # Essential ratios
    df["peak_to_mean_ratio"] = df["peak_kwh"] / df["mean_kwh"]
    df["peak_to_mean_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df["peak_to_total_ratio"] = df["peak_kwh"] / df["total_kwh"]
    df["peak_to_total_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Day/night patterns
    df["day_night_ratio"] = (
        (df["morning_kwh"] + df["afternoon_kwh"]) /
        (df["evening_kwh"] + df["night_kwh"])
    )
    df["day_night_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Load factor (important for energy analysis)
    df["load_factor"] = df["mean_kwh"] / df["peak_kwh"]
    df["load_factor"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Daily variability (important for forecasting uncertainty)
    df["daily_variability"] = hh_data.std(axis=1)
    df["coefficient_of_variation"] = df["std_kwh"] / df["mean_kwh"]
    df["coefficient_of_variation"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"   ✅ Created consumption features")
    return df

def create_consumption_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create consumption pattern features for household archetypes
    
    Args:
        df: Dataframe with consumption features
        
    Returns:
        Dataframe with pattern features
    """
    print("📊 Creating consumption patterns...")
    
    hh_cols = [f"hh_{i}" for i in range(48)]
    if not all(col in df.columns for col in hh_cols[:5]):  # Check if we have hh data
        print("   ⚠️ Half-hourly data not found")
        return df
    
    hh_data = df[hh_cols].to_numpy()
    
    # Usage concentration (how concentrated is usage across the day)
    df["usage_concentration"] = np.sum(hh_data ** 2, axis=1) / (np.sum(hh_data, axis=1) ** 2)
    df["usage_concentration"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Peak sharpness (how sharp are the peaks)
    df["peak_sharpness"] = df["peak_kwh"] / (df["mean_kwh"] + 1e-6)
    
    # Base load (minimum sustained consumption)
    df["base_load"] = np.percentile(hh_data, 10, axis=1)  # 10th percentile
    df["base_load_ratio"] = df["base_load"] / df["total_kwh"]
    
    print("   ✅ Created consumption patterns")
    return df

def create_timeseries_features(df: pd.DataFrame, 
                              target_col: str = "total_kwh",
                              lags: List[int] = [1, 7, 14],
                              windows: List[int] = [7, 14]) -> pd.DataFrame:
    """
    Create time series features essential for forecasting
    
    Args:
        df: Input dataframe
        target_col: Target column for lags/rolling
        lags: Lag periods in days
        windows: Rolling window sizes in days
        
    Returns:
        Dataframe with time series features
    """
    print(f"📈 Creating time series features...")
    
    if target_col not in df.columns or "LCLid" not in df.columns:
        print(f"   ⚠️ Required columns not found")
        return df
    
    # Sort by household and date
    df = df.sort_values(["LCLid", "day"]).reset_index(drop=True)
    
    # Lag features (essential for forecasting)
    for lag in lags:
        lag_col = f"lag{lag}_{target_col.split('_')[0]}"
        df[lag_col] = df.groupby("LCLid")[target_col].shift(lag)
    
    # Rolling features (capture trends)
    for window in windows:
        roll_col = f"roll{window}_{target_col.split('_')[0]}_mean"
        df[roll_col] = (
            df.groupby("LCLid")[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # Rolling std (capture volatility)
        roll_std_col = f"roll{window}_{target_col.split('_')[0]}_std"
        df[roll_std_col] = (
            df.groupby("LCLid")[target_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
    
    # Change features (capture volatility)
    if f"lag1_{target_col.split('_')[0]}" in df.columns:
        delta_col = f"delta1_{target_col.split('_')[0]}"
        df[delta_col] = df[target_col] - df[f"lag1_{target_col.split('_')[0]}"]
        
        # Percentage change
        pct_change_col = f"pct_change1_{target_col.split('_')[0]}"
        df[pct_change_col] = df[delta_col] / (df[f"lag1_{target_col.split('_')[0]}"] + 1e-6)
        df[pct_change_col].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Weekly comparison (compare to same day last week)
    if f"lag7_{target_col.split('_')[0]}" in df.columns:
        weekly_change_col = f"weekly_change_{target_col.split('_')[0]}"
        df[weekly_change_col] = df[target_col] - df[f"lag7_{target_col.split('_')[0]}"]
    
    print(f"   ✅ Created time series features")
    return df

def create_household_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create household-specific consumption characteristics
    
    Args:
        df: Input dataframe with consumption data
        
    Returns:
        Dataframe with household characteristics
    """
    print("🏠 Creating household characteristics...")
    
    if "LCLid" not in df.columns or "total_kwh" not in df.columns:
        print("   ⚠️ Required columns not found")
        return df
    
    # Household-level statistics (computed once per household)
    household_stats = df.groupby("LCLid")["total_kwh"].agg([
        ("hh_avg_consumption", "mean"),
        ("hh_std_consumption", "std"),
        ("hh_max_consumption", "max"),
        ("hh_min_consumption", "min")
    ]).reset_index()
    
    # Merge back to main dataframe
    df = df.merge(household_stats, on="LCLid", how="left")
    
    # Relative consumption features
    df["daily_vs_hh_avg"] = df["total_kwh"] / df["hh_avg_consumption"]
    df["daily_vs_hh_max"] = df["total_kwh"] / df["hh_max_consumption"]
    
    print("   ✅ Created household characteristics")
    return df

if __name__ == "__main__":
    print("⚡ Consumption Features Module")
    print("Usage: from src.features.consumption_features import create_consumption_features") 