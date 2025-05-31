"""
📅 TEMPORAL FEATURES - Enhanced Temporal Features for Forecasting
================================================================

Enhanced temporal features for electricity consumption forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_basic_temporal_features(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Create basic temporal features for forecasting
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        
    Returns:
        Dataframe with temporal features added
    """
    print("📅 Creating basic temporal features...")
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic temporal features
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["day_of_year"] = df[date_col].dt.dayofyear
    
    # Weekend/weekday patterns
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_weekday"] = 1 - df["is_weekend"]
    
    # Specific weekdays (important for consumption patterns)
    df["is_monday"] = (df["dayofweek"] == 0).astype(int)
    df["is_friday"] = (df["dayofweek"] == 4).astype(int)
    
    print("   ✅ Created basic temporal features")
    return df

def create_seasonal_features(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Create seasonal features for energy consumption
    
    Args:
        df: Input dataframe  
        date_col: Name of date column
        
    Returns:
        Dataframe with seasonal features
    """
    print("🌀 Creating seasonal features...")
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Cyclical encoding for month (important for seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Cyclical encoding for day of week
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # Seasonal indicators
    if "season" in df.columns:
        df["is_winter"] = (df["season"] == "Winter").astype(int)
        df["is_summer"] = (df["season"] == "Summer").astype(int)
        df["is_shoulder_season"] = df["season"].isin(["Spring", "Autumn"]).astype(int)
    
    print("   ✅ Created seasonal features")
    return df

def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create holiday-related features
    
    Args:
        df: Dataframe with holiday_category column
        
    Returns:
        Dataframe with holiday features
    """
    print("🎉 Creating holiday features...")
    
    if "holiday_category" not in df.columns:
        print("   ⚠️ holiday_category column not found")
        return df
    
    # Holiday indicators
    df["is_holiday"] = (df["holiday_category"] != "Regular Day").astype(int)
    df["is_christmas"] = (df["holiday_category"] == "Christmas").astype(int)
    df["is_new_year"] = (df["holiday_category"] == "New Year").astype(int)
    df["is_bank_holiday"] = (df["holiday_category"] == "Bank Holiday").astype(int)
    
    # Holiday interactions with weekdays
    df["holiday_weekday"] = df["is_holiday"] * df["is_weekday"]
    df["holiday_weekend"] = df["is_holiday"] * df["is_weekend"]
    
    print("   ✅ Created holiday features")
    return df

def create_peak_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create peak period features for UK electricity consumption
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with peak period features
    """
    print("⚡ Creating peak period features...")
    
    if "peak_hour" not in df.columns:
        print("   ⚠️ peak_hour column not found")
        return df
    
    # UK-specific peak periods
    df["is_morning_peak"] = df["peak_hour"].isin([7, 8, 9]).astype(int)      # 7-9 AM
    df["is_evening_peak"] = df["peak_hour"].isin([17, 18, 19, 20]).astype(int)  # 5-8 PM
    df["is_off_peak"] = df["peak_hour"].isin(range(0, 6)).astype(int)        # Midnight-6 AM
    
    # Peak period interactions with weekdays
    if "is_weekday" in df.columns:
        df["weekday_evening_peak"] = df["is_weekday"] * df["is_evening_peak"]
        df["weekend_peak_shift"] = df["is_weekend"] * df["is_morning_peak"]
    
    print("   ✅ Created peak period features")
    return df

def create_time_trend_features(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Create time trend features for long-term patterns
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        
    Returns:
        Dataframe with trend features
    """
    print("📈 Creating time trend features...")
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Linear time trend (days since start)
    start_date = df[date_col].min()
    df["days_since_start"] = (df[date_col] - start_date).dt.days
    
    # Quarterly features (for quarterly patterns)
    df["quarter"] = df[date_col].dt.quarter
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)  # Winter quarter
    
    print("   ✅ Created time trend features")
    return df

def create_all_temporal_features(df: pd.DataFrame, date_col: str = "day") -> pd.DataFrame:
    """
    Create all temporal features for forecasting
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        
    Returns:
        Dataframe with all temporal features
    """
    print("🚀 Creating All Temporal Features")
    print("=" * 35)
    
    # Create all temporal feature types
    df = create_basic_temporal_features(df, date_col)
    df = create_seasonal_features(df, date_col)
    df = create_holiday_features(df)
    df = create_peak_period_features(df)
    df = create_time_trend_features(df, date_col)
    
    print("✅ All temporal features created")
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
    
if __name__ == "__main__":
    print("📅 Temporal Features Module")
    print("Usage: from src.features.temporal_features import create_all_temporal_features") 