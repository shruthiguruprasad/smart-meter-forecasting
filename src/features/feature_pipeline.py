"""
FEATURE PIPELINE - Comprehensive Features (no dropping)
=======================================================

Comprehensive feature pipeline for electricity consumption forecasting.
**This version does _not_ drop any columns.** All raw and intermediate features
remain in the output DataFrame. Downstream splitters must handle any leakage‐prone
columns if desired.

Author: Shruthi Simha Chippagiri (updated)
Date: 2025
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
#  IMPORT ALL HELPER FUNCTIONS NORMALLY USED TO BUILD FEATURES:
# -----------------------------------------------------------------------

from .consumption_features import (
    create_consumption_features,
    create_consumption_patterns
)

from .temporal_features import (
    create_all_temporal_features
)

from .weather_features import (
    create_all_weather_features
)

# (If there are any other imports in the original file—e.g. utilities for merging household stats—keep them as they were.)


# -----------------------------------------------------------------------------
#  MAIN FUNCTION: CREATE ALL FEATURES, BUT DO NOT DROP ANY COLUMNS AT THE END
# -----------------------------------------------------------------------------
def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a "full" set of leakage‐safe features for electricity forecasting,
    but do NOT drop ANY columns. Downstream logic (splitters) will have to
    drop raw targets if they shouldn't be exposed.

    Components (in order):
      1) Temporal features
      2) Consumption‐pattern features (intermediate)
      3) Weather features
      4) Time‐series features (lags, rolling windows)
      5) Leakage‐safe interaction features

    Args:
        df: Input DataFrame containing at least:
            - 'LCLid' (household ID)
            - 'day'   (date or datetime)
            - 'total_kwh' (raw daily usage) or half‐hourly columns for consumption
            - Any raw weather columns (if present)
            - Any pre‐computed flags (e.g., is_holiday, is_summer) or else the
              temporal helper will create them.

    Returns:
        A new DataFrame with every feature added (including the raw columns).
        No leakage‐prone columns are dropped here.
    """
    print("🚀 CREATING COMPREHENSIVE FEATURES (all features retained)")
    print("=" * 50)

    # ---------------------------------------------------------------------
    #  1) TEMPORAL FEATURES
    # ---------------------------------------------------------------------
    print("📅 Creating temporal features...")
    # This will add features such as dayofweek, month, is_holiday, is_weekend, etc.
    df = create_all_temporal_features(df)


    # ---------------------------------------------------------------------
    #  2) CONSUMPTION‐PATTERN FEATURES (INTERMEDIATE)
    # ---------------------------------------------------------------------
    print("⚡ Creating consumption‐pattern features...")
    # These might compute things like "peak_kwh," "daily_variability," etc.
    # (They do not drop raw columns yet—just augment the DataFrame.)
    df = create_consumption_features(df)
    df = create_consumption_patterns(df)


    # ---------------------------------------------------------------------
    #  3) WEATHER FEATURES
    # ---------------------------------------------------------------------
    print("🌤️ Creating weather features...")
    # This will merge or compute derived weather variables (e.g. degree days).
    df = create_all_weather_features(df)


    # ---------------------------------------------------------------------
    #  4) TIME‐SERIES FEATURES (LAGS, ROLLING WINDOWS)
    # ---------------------------------------------------------------------
    print("📈 Creating time‐series features (lags and rolling windows)…")
    # We pass target_col="total_kwh" so that the helper shifts by 1, 7, 14, etc.
    # This will create columns like lag1_total, roll7_mean, roll14_std, etc.
    df = create_timeseries_features_safe(
        df,
        target_col="total_kwh",
        lags=[1, 7, 14],
        windows=[7, 14]
    )

    # ---------------------------------------------------------------------
    #  5) LEAKAGE‐SAFE INTERACTION FEATURES
    # ---------------------------------------------------------------------
    print("🔗 Creating leakage‐safe interaction features...")
    eps = 1e-6

    # Example 1: Weekend × Heating × (lag1_total / (lag1_total + eps))
    if {"is_weekend", "heating_degree_days", "lag1_total"}.issubset(df.columns):
        df["lag1_weekend_heating"] = (
            df["is_weekend"]
            * df["heating_degree_days"]
            * (df["lag1_total"] / (df["lag1_total"] + eps))
        ).fillna(0)

    # Example 2: Holiday × lag1_total
    if {"is_holiday", "lag1_total"}.issubset(df.columns):
        df["lag1_holiday_consumption"] = (df["is_holiday"] * df["lag1_total"]).fillna(0)

    # Example 3: Summer × Cooling × (lag1_total / (lag1_total + eps))
    if {"is_summer", "cooling_degree_days", "lag1_total"}.issubset(df.columns):
        df["lag1_summer_cooling"] = (
            df["is_summer"]
            * df["cooling_degree_days"]
            * (df["lag1_total"] / (df["lag1_total"] + eps))
        ).fillna(0)


    print("✅ FEATURE PIPELINE COMPLETE (all features retained)")
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

    # Create rolling-window features on shifted target (
    for window in windows:
        # Make a “t−1” series, then rolling on that
        temp = df.groupby("LCLid")[target_col].shift(1)
        df[f"roll{window}_total_mean"] = (
            temp.groupby(df["LCLid"])
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
    
# -----------------------------------------------------------------------------
#  IF THIS MODULE IS RUN DIRECTLY
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Feature Pipeline - No‐drop version")
    print("Usage: from src.features.feature_pipeline import create_comprehensive_features")
