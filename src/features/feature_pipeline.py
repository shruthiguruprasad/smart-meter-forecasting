"""
FEATURE PIPELINE - Comprehensive Features for Forecasting
===========================================================

Comprehensive feature pipeline for electricity consumption forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from .consumption_features import (
    create_consumption_features, 
    create_consumption_patterns,
    create_timeseries_features,
    create_household_characteristics
)
from .temporal_features import create_all_temporal_features
from .weather_features import create_all_weather_features
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive features for electricity consumption forecasting
    (except group/household features, which are handled after splitting to avoid leakage)
    
    Args:
        df: Input dataframe with cleaned data
        
    Returns:
        Dataframe with comprehensive features for modeling
    """
    print("🚀 CREATING COMPREHENSIVE FEATURES FOR FORECASTING")
    print("=" * 55)
    
    # 1. Temporal features (foundation for time series)
    df = create_all_temporal_features(df)
    
    # 2. Core consumption features
    df = create_consumption_features(df)
    
    # 3. Consumption patterns (for household archetypes)
    df = create_consumption_patterns(df)
    
    # 4. Weather features (important external factors)
    df = create_all_weather_features(df)
    
    # 5. ACORN socio-economic features (REMOVED: now handled after split)
    # 6. Household characteristics (REMOVED: now handled after split)
    
    # 7. Time series features (critical for forecasting)
    df = create_timeseries_features(df, target_col="total_kwh", lags=[1, 7, 14], windows=[7, 14])
    
    # 8. Interaction features
    print("🔗 Creating interaction features...")
    
    # Weather-temporal interactions
    if all(col in df.columns for col in ["is_weekend", "heating_degree_days"]):
        df["weekend_heating"] = df["is_weekend"] * df["heating_degree_days"]
        df["weekday_heating"] = df["is_weekday"] * df["heating_degree_days"]
    
    if all(col in df.columns for col in ["is_summer", "cooling_degree_days"]):
        df["summer_cooling"] = df["is_summer"] * df["cooling_degree_days"]
    
    # Holiday-consumption interactions
    if all(col in df.columns for col in ["is_holiday", "total_kwh"]):
        df["holiday_consumption_boost"] = df["is_holiday"] * df["total_kwh"]
    
    # 🔧 CRITICAL FIX: Add missing critical features expected by notebooks
    # Holiday-heating interaction (critical feature)
    if all(col in df.columns for col in ["is_holiday", "heating_degree_days"]):
        df["holiday_heating_interaction"] = df["is_holiday"] * df["heating_degree_days"]
    
    print("✅ ALL COMPREHENSIVE FEATURES CREATED")
    print(f"📊 Final shape: {df.shape}")
    
    # Show detailed feature summary
    feature_counts = {
        'Consumption Basic': len([c for c in df.columns if any(x in c for x in ['total', 'mean', 'peak', 'min', 'std'])]),
        'Time-of-Day': len([c for c in df.columns if any(x in c for x in ['morning', 'afternoon', 'evening', 'night'])]),
        'Consumption Patterns': len([c for c in df.columns if any(x in c for x in ['ratio', 'variability', 'concentration', 'sharpness', 'load_factor'])]),
        'Weather': len([c for c in df.columns if any(x in c for x in ['temp', 'heating', 'cooling', 'humidity', 'wind', 'cloud'])]),
        'Temporal': len([c for c in df.columns if any(x in c for x in ['dayofweek', 'weekend', 'month', 'season', 'holiday', 'quarter'])]),
        'Time Series': len([c for c in df.columns if any(x in c for x in ['lag', 'roll', 'delta', 'pct_change', 'weekly'])]),
        'Interactions': len([c for c in df.columns if any(x in c for x in ['weekend_heating', 'summer_cooling', 'holiday_'])]),
        'Peak Timing': len([c for c in df.columns if any(x in c for x in ['peak_hour', 'peak_period', 'off_peak'])])
    }
    # Note: ACORN and Household features are now handled after split
    
    print("\n📋 COMPREHENSIVE FEATURE SUMMARY:")
    total_features = sum(feature_counts.values())
    for category, count in feature_counts.items():
        print(f"   {category}: {count} features")
    print(f"   TOTAL: {total_features} features")
    
    return df

def get_forecasting_feature_groups(df: pd.DataFrame) -> dict:
    """
    Organize features into logical groups for model interpretation
    
    Args:
        df: Dataframe with features
        
    Returns:
        Dictionary of feature groups
    """
    exclude_patterns = ["LCLid", "day", "holiday_type"]
    all_cols = [col for col in df.columns 
                if not any(pattern in col for pattern in exclude_patterns)
                and not (col.startswith('hh_') and col.replace('hh_', '').isdigit())]
    
    feature_groups = {
        'consumption_basic': [c for c in all_cols if any(x in c for x in ['total', 'mean', 'peak', 'min', 'std'])],
        'time_of_day': [c for c in all_cols if any(x in c for x in ['morning', 'afternoon', 'evening', 'night'])],
        'consumption_patterns': [c for c in all_cols if any(x in c for x in ['ratio', 'variability', 'concentration', 'sharpness'])],
        'weather': [c for c in all_cols if any(x in c for x in ['temp', 'heating', 'cooling', 'humidity', 'wind', 'cloud'])],
        'temporal': [c for c in all_cols if any(x in c for x in ['dayofweek', 'weekend', 'month', 'season', 'holiday'])],
        'time_series': [c for c in all_cols if any(x in c for x in ['lag', 'roll', 'delta', 'pct_change'])],
        'household': [c for c in all_cols if any(x in c for x in ['acorn', 'hh_avg', 'hh_std', 'hh_max', 'hh_min', 'daily_vs'])]
    }
    
    return feature_groups

def get_forecasting_features(df: pd.DataFrame) -> list:
    """
    Get list of features suitable for forecasting (exclude IDs, dates, targets)
    
    Args:
        df: Dataframe with features
        
    Returns:
        List of feature column names
    """
    exclude_patterns = ["LCLid", "day", "holiday_type"]
    feature_cols = []
    
    for col in df.columns:
        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip if it looks like raw half-hourly data (hh_0, hh_1, etc.) but NOT derived features
        if col.startswith('hh_') and col.replace('hh_', '').isdigit():
            continue
        feature_cols.append(col)
    
    print(f"📊 Selected {len(feature_cols)} forecasting features")
    return feature_cols

def add_group_and_household_features(train_df, test_df):
    """
    Add group-level (ACORN) and household-level features in a leakage-safe way.
    Compute stats on train only, merge into both train and test.
    """
    # ACORN group features
    if 'Acorn_grouped' in train_df.columns:
        acorn_means = train_df.groupby("Acorn_grouped")["total_kwh"].mean().rename("acorn_avg_consumption")
        train_df = train_df.merge(acorn_means, on="Acorn_grouped", how="left")
        test_df = test_df.merge(acorn_means, on="Acorn_grouped", how="left")
        train_df["acorn_consumption_ratio"] = train_df["total_kwh"] / (train_df["acorn_avg_consumption"] + 1e-6)
        test_df["acorn_consumption_ratio"] = test_df["total_kwh"] / (test_df["acorn_avg_consumption"] + 1e-6)
        # Peak behavior by ACORN group
        acorn_peak = train_df.groupby("Acorn_grouped")["peak_kwh"].mean().rename("acorn_peak_kwh")
        acorn_total = train_df.groupby("Acorn_grouped")["total_kwh"].mean().rename("acorn_total_kwh")
        acorn_peak_ratio = (acorn_peak / (acorn_total + 1e-6)).rename("acorn_peak_ratio")
        train_df = train_df.merge(acorn_peak_ratio, on="Acorn_grouped", how="left")
        test_df = test_df.merge(acorn_peak_ratio, on="Acorn_grouped", how="left")
        # Variability by ACORN group
        acorn_var = train_df.groupby("Acorn_grouped")["daily_variability"].mean().rename("acorn_variability")
        train_df = train_df.merge(acorn_var, on="Acorn_grouped", how="left")
        test_df = test_df.merge(acorn_var, on="Acorn_grouped", how="left")
        train_df["relative_variability"] = train_df["daily_variability"] / (train_df["acorn_variability"] + 1e-6)
        test_df["relative_variability"] = test_df["daily_variability"] / (test_df["acorn_variability"] + 1e-6)
    # Household-level features
    if 'LCLid' in train_df.columns:
        hh_stats = train_df.groupby("LCLid")["total_kwh"].agg([
            ("hh_avg_consumption", "mean"),
            ("hh_std_consumption", "std"),
            ("hh_max_consumption", "max"),
            ("hh_min_consumption", "min")
        ]).reset_index()
        train_df = train_df.merge(hh_stats, on="LCLid", how="left")
        test_df = test_df.merge(hh_stats, on="LCLid", how="left")
        train_df["daily_vs_hh_avg"] = train_df["total_kwh"] / train_df["hh_avg_consumption"]
        test_df["daily_vs_hh_avg"] = test_df["total_kwh"] / test_df["hh_avg_consumption"]
        train_df["daily_vs_hh_max"] = train_df["total_kwh"] / train_df["hh_max_consumption"]
        test_df["daily_vs_hh_max"] = test_df["total_kwh"] / test_df["hh_max_consumption"]
    return train_df, test_df

def prepare_forecasting_data(df: pd.DataFrame, 
                           target_col: str = "total_kwh",
                           test_days: int = 90,
                           val_days: int = 30) -> tuple:
    """
    Prepare data for forecasting models with chronological split (leakage-safe)
    Includes train/validation/test splits for proper model evaluation
    
    Args:
        df: Dataframe with features
        target_col: Target variable for forecasting
        test_days: Number of days to use for test set (default: 90 days)
        val_days: Number of days to use for validation set (default: 30 days)
        
    Returns:
        Tuple of (train_df, val_df, test_df, feature_cols, target_col, feature_groups)
    """
    print("📊 Preparing data for forecasting...")
    
    # Get initial feature columns and groups (will be updated after household features)
    initial_feature_cols = get_forecasting_features(df)
    feature_groups = get_forecasting_feature_groups(df)
    
    # Ensure day column is datetime
    df["day"] = pd.to_datetime(df["day"])
    
    # Sort by date to ensure chronological order
    df = df.sort_values("day")
    
    # Calculate split dates
    test_start = df["day"].max() - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    
    # Split by date for time series
    train_df = df[df["day"] < val_start].copy()
    val_df = df[(df["day"] >= val_start) & (df["day"] < test_start)].copy()
    test_df = df[df["day"] >= test_start].copy()
    
    # --- Add group/household features in a leakage-safe way ---
    # Compute features on train only, then apply to val and test
    train_df, _ = add_group_and_household_features(train_df, pd.concat([val_df, test_df]))
    val_df, test_df = add_group_and_household_features(val_df, test_df)
    
    # 🔧 CRITICAL FIX: Update feature columns AFTER household features are added
    print("🔄 Updating feature list to include household features...")
    feature_cols = get_forecasting_features(train_df)  # Use train_df which now has all features
    feature_groups = get_forecasting_feature_groups(train_df)  # Update groups too
    
    print(f"   ✅ Initial features: {len(initial_feature_cols)}")
    print(f"   ✅ Final features: {len(feature_cols)} (+{len(feature_cols) - len(initial_feature_cols)} household features)")
    
    print(f"   ✅ Train: {len(train_df):,} rows ({train_df['LCLid'].nunique()} households)")
    print(f"   ✅ Validation: {len(val_df):,} rows ({val_df['LCLid'].nunique()} households)")
    print(f"   ✅ Test: {len(test_df):,} rows ({test_df['LCLid'].nunique()} households)")
    print(f"   ✅ Train period: {train_df['day'].min()} to {train_df['day'].max()}")
    print(f"   ✅ Validation period: {val_df['day'].min()} to {val_df['day'].max()}")
    print(f"   ✅ Test period: {test_df['day'].min()} to {test_df['day'].max()}")
    print(f"   ✅ Features: {len(feature_cols)} columns")
    print(f"   ✅ Feature groups: {len(feature_groups)} groups")
    print(f"   ✅ Target: {target_col}")
    
    return train_df, val_df, test_df, feature_cols, target_col, feature_groups

if __name__ == "__main__":
    print("🔧 Feature Pipeline - Comprehensive Features for Forecasting")
    print("Usage: from src.features.feature_pipeline import create_comprehensive_features") 