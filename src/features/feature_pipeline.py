"""
�� FEATURE PIPELINE - Comprehensive Features for Forecasting
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
    
    # 5. ACORN socio-economic features
    print("🏠 Creating ACORN features...")
    if 'Acorn_grouped' in df.columns:
        # Group-level consumption patterns
        df["acorn_avg_consumption"] = df.groupby("Acorn_grouped")["total_kwh"].transform("mean")
        df["acorn_consumption_ratio"] = df["total_kwh"] / (df["acorn_avg_consumption"] + 1e-6)
        
        # Peak behavior by ACORN group
        df["acorn_peak_ratio"] = (
            df.groupby("Acorn_grouped")["peak_kwh"].transform("mean") /
            (df.groupby("Acorn_grouped")["total_kwh"].transform("mean") + 1e-6)
        )
        
        # Variability by ACORN group
        df["acorn_variability"] = df.groupby("Acorn_grouped")["daily_variability"].transform("mean")
        df["relative_variability"] = df["daily_variability"] / (df["acorn_variability"] + 1e-6)
    
    # 6. Household characteristics
    df = create_household_characteristics(df)
    
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
        'ACORN': len([c for c in df.columns if 'acorn' in c.lower()]),
        'Household': len([c for c in df.columns if any(x in c for x in ['hh_avg', 'hh_std', 'hh_max', 'daily_vs'])]),
        'Interactions': len([c for c in df.columns if any(x in c for x in ['weekend_heating', 'summer_cooling', 'holiday_'])]),
        'Peak Timing': len([c for c in df.columns if any(x in c for x in ['peak_hour', 'peak_period', 'off_peak'])])
    }
    
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
    exclude_patterns = ["LCLid", "day", "hh_", "holiday_type"]
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
        'household': [c for c in all_cols if any(x in c for x in ['acorn', 'hh_avg', 'daily_vs'])]
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
    exclude_patterns = ["LCLid", "day", "hh_", "holiday_type"]
    feature_cols = []
    
    for col in df.columns:
        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip if it looks like raw half-hourly data
        if col.startswith('hh_') and col.replace('hh_', '').isdigit():
            continue
        feature_cols.append(col)
    
    print(f"📊 Selected {len(feature_cols)} forecasting features")
    return feature_cols

def prepare_forecasting_data(df: pd.DataFrame, 
                           target_col: str = "total_kwh",
                           test_start: str = "2014-01-01") -> tuple:
    """
    Prepare data for forecasting models
    
    Args:
        df: Dataframe with features
        target_col: Target variable for forecasting
        test_start: Start date for test set
        
    Returns:
        Tuple of (train_df, test_df, feature_cols, target_col, feature_groups)
    """
    print("📊 Preparing data for forecasting...")
    
    # Get feature columns and groups
    feature_cols = get_forecasting_features(df)
    feature_groups = get_forecasting_feature_groups(df)
    
    # Split by date for time series
    df["day"] = pd.to_datetime(df["day"])
    test_start = pd.to_datetime(test_start)
    
    train_df = df[df["day"] < test_start].copy()
    test_df = df[df["day"] >= test_start].copy()
    
    print(f"   ✅ Train: {len(train_df):,} rows ({train_df['LCLid'].nunique()} households)")
    print(f"   ✅ Test: {len(test_df):,} rows ({test_df['LCLid'].nunique()} households)")
    print(f"   ✅ Features: {len(feature_cols)} columns")
    print(f"   ✅ Feature groups: {len(feature_groups)} groups")
    print(f"   ✅ Target: {target_col}")
    
    return train_df, test_df, feature_cols, target_col, feature_groups

if __name__ == "__main__":
    print("🔧 Feature Pipeline - Comprehensive Features for Forecasting")
    print("Usage: from src.features.feature_pipeline import create_comprehensive_features") 