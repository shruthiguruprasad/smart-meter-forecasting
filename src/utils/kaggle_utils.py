"""
🔧 KAGGLE UTILS - Streamlined Smart Meter Data Preparation
=========================================================

Streamlined utility for Kaggle notebooks focusing on essential features for forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/kaggle/working')
import warnings
warnings.filterwarnings('ignore')

# Try to import from local modules, fall back to inline definitions
try:
    from src.data.data_loader import load_all_raw_data
    from src.data.data_cleaner import clean_and_merge_all_data
    from src.features.feature_pipeline import create_essential_features, prepare_forecasting_data
except ImportError:
    print("⚠️ Using inline functions (local imports not available)")

def load_smart_meter_data_kaggle(data_path: str = "/kaggle/input/smart-meters-in-london/") -> pd.DataFrame:
    """
    Load and prepare smart meter data for Kaggle (essential features only)
    
    Args:
        data_path: Path to smart meter dataset
        
    Returns:
        Complete dataframe with essential features for forecasting
    """
    print("🚀 LOADING SMART METER DATA FOR KAGGLE")
    print("=" * 45)
    
    try:
        # Try using modular approach
        raw_data = load_all_raw_data(data_path)
        df = clean_and_merge_all_data(raw_data)
        df = create_essential_features(df)
        
    except NameError:
        # Fall back to inline loading
        print("📂 Using inline data loading...")
        df = load_and_clean_inline(data_path)
        df = create_essential_features_inline(df)
    
    print(f"🎉 DATA READY FOR MODELING: {df.shape}")
    return df

def load_and_clean_inline(data_path: str) -> pd.DataFrame:
    """
    Inline data loading and cleaning for Kaggle
    """
    import glob
    
    # Load consumption data
    print("📂 Loading consumption data...")
    files = glob.glob(f"{data_path}hhblock_dataset/hhblock_dataset/block_*.csv")
    df_list = []
    for f in files:
        df_part = pd.read_csv(f, parse_dates=["day"])
        df_list.append(df_part)
    df = pd.concat(df_list, ignore_index=True)
    
    # Clean consumption data
    print("🧹 Cleaning consumption data...")
    hh_cols = [f"hh_{i}" for i in range(48)]
    df["missing_ratio"] = df[hh_cols].isna().mean(axis=1)
    df = df[df["missing_ratio"] < 0.2].copy()
    
    valid_counts = df["LCLid"].value_counts()
    keep_ids = valid_counts[valid_counts >= 30].index
    df = df[df["LCLid"].isin(keep_ids)]
    
    # Interpolate missing values
    hh_matrix = df[hh_cols].to_numpy()
    for i in range(hh_matrix.shape[0]):
        x = hh_matrix[i]
        if np.isnan(x).any():
            n = len(x)
            not_nan = ~np.isnan(x)
            if not_nan.sum() > 0:
                hh_matrix[i] = np.interp(np.arange(n), np.where(not_nan)[0], x[not_nan])
    df[hh_cols] = hh_matrix
    df = df.drop(['missing_ratio'], axis=1)
    
    # Merge household data
    print("🏠 Merging household data...")
    household_data = pd.read_csv(f"{data_path}informations_households.csv")
    df = df.merge(household_data[['LCLid', 'Acorn', 'Acorn_grouped', 'stdorToU']], on='LCLid', how='left')
    
    # Merge weather data
    print("🌤️ Merging weather data...")
    weather_data = pd.read_csv(f"{data_path}weather_daily_darksky.csv", parse_dates=["time"])
    weather_cols = ["time", "temperatureMax", "temperatureMin", "humidity", "windSpeed", "cloudCover"]
    weather_data = weather_data[weather_cols].rename(columns={"time": "day"})
    weather_data["day"] = pd.to_datetime(weather_data["day"]).dt.date
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df = df.merge(weather_data, on="day", how="left")
    
    # Merge holiday data
    print("🎉 Merging holiday data...")
    holiday_data = pd.read_csv(f"{data_path}uk_bank_holidays.csv", parse_dates=["Bank holidays"])
    holiday_data = holiday_data.rename(columns={"Bank holidays": "day", "Type": "holiday_type"})
    holiday_data["day"] = pd.to_datetime(holiday_data["day"]).dt.date
    
    holiday_mapping = {
        'New Year?s Day': 'New Year', 'Christmas Day': 'Christmas', 'Boxing Day': 'Christmas',
        'Easter Monday': 'Easter', 'Good Friday': 'Easter', 'Summer bank holiday': 'Bank Holiday',
        'Early May bank holiday': 'Bank Holiday', 'Spring bank holiday': 'Bank Holiday'
    }
    holiday_data['holiday_category'] = holiday_data['holiday_type'].map(holiday_mapping).fillna('Regular Day')
    df = df.merge(holiday_data[["day", "holiday_category"]], on="day", how="left")
    df['holiday_category'] = df['holiday_category'].fillna('Regular Day')
    
    # Add temporal features
    df["day"] = pd.to_datetime(df["day"])
    df["month"] = df["day"].dt.month
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    
    return df

def create_essential_features_inline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create essential features inline for Kaggle
    """
    print("🚀 Creating essential features...")
    
    hh_cols = [f"hh_{i}" for i in range(48)]
    hh_data = df[hh_cols].to_numpy()
    
    # Basic consumption features
    df["total_kwh"] = hh_data.sum(axis=1)
    df["mean_kwh"] = hh_data.mean(axis=1)
    df["std_kwh"] = hh_data.std(axis=1)
    df["peak_kwh"] = np.nanmax(hh_data, axis=1)
    df["peak_hour"] = np.nanargmax(hh_data, axis=1)
    
    # Time-of-day features
    df["morning_kwh"] = hh_data[:, 6:12].sum(axis=1)
    df["afternoon_kwh"] = hh_data[:, 12:18].sum(axis=1)
    df["evening_kwh"] = hh_data[:, 18:24].sum(axis=1)
    night_idx = np.r_[0:6, 24:48]
    df["night_kwh"] = hh_data[:, night_idx].sum(axis=1)
    
    # Calendar features
    df["dayofweek"] = df["day"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_weekday"] = 1 - df["is_weekend"]
    
    # Weather features
    df["temp_range"] = df["temperatureMax"] - df["temperatureMin"]
    df["heating_degree_days"] = np.maximum(15 - df["temperatureMax"], 0)
    df["cooling_degree_days"] = np.maximum(df["temperatureMax"] - 22, 0)
    
    # Peak features
    df["is_weekday_peak"] = (
        (df["dayofweek"] < 5) & df["peak_hour"].isin([17, 18, 19, 20])
    ).astype(int)
    
    # Essential ratios
    df["peak_to_mean_ratio"] = df["peak_kwh"] / df["mean_kwh"]
    df["peak_to_mean_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Variability
    df["daily_variability"] = hh_data.std(axis=1)
    
    # ACORN features
    if 'Acorn_grouped' in df.columns:
        df["acorn_avg_consumption"] = df.groupby("Acorn_grouped")["total_kwh"].transform("mean")
        df["acorn_consumption_ratio"] = df["total_kwh"] / df["acorn_avg_consumption"]
        df["acorn_peak_ratio"] = (
            df.groupby("Acorn_grouped")["peak_kwh"].transform("mean") /
            df.groupby("Acorn_grouped")["total_kwh"].transform("mean")
        )
    
    # Time series features
    df = df.sort_values(["LCLid", "day"]).reset_index(drop=True)
    df["lag1_total"] = df.groupby("LCLid")["total_kwh"].shift(1)
    df["lag7_total"] = df.groupby("LCLid")["total_kwh"].shift(7)
    df["roll7_total_mean"] = (
        df.groupby("LCLid")["total_kwh"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["delta1_total"] = df["total_kwh"] - df["lag1_total"]
    
    print("✅ Essential features created")
    return df

def get_essential_features(df: pd.DataFrame) -> list:
    """Get list of essential feature columns for forecasting"""
    exclude_patterns = ["LCLid", "day", "hh_", "holiday_type"]
    feature_cols = []
    for col in df.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            if not (col.startswith('hh_') and col.replace('hh_', '').isdigit()):
                feature_cols.append(col)
    return feature_cols

def train_test_split_by_date(df: pd.DataFrame, test_start: str = "2014-01-01"):
    """Split data by date for time series"""
    df["day"] = pd.to_datetime(df["day"])
    test_start = pd.to_datetime(test_start)
    
    train_df = df[df["day"] < test_start].copy()
    test_df = df[df["day"] >= test_start].copy()
    
    print(f"📊 Train: {len(train_df):,} rows ({train_df['LCLid'].nunique()} households)")
    print(f"📊 Test: {len(test_df):,} rows ({test_df['LCLid'].nunique()} households)")
    
    return train_df, test_df

def sample_households(df: pd.DataFrame, n_households: int = 100):
    """Get sample of households for testing"""
    households = df["LCLid"].unique()
    sampled = np.random.choice(households, size=min(n_households, len(households)), replace=False)
    return df[df["LCLid"].isin(sampled)].copy()

def prepare_smart_meter_data(data_path: str = "/kaggle/input/smart-meters-in-london/",
                           sample_size: int = None) -> pd.DataFrame:
    """
    One-stop function to prepare smart meter data for forecasting
    
    Args:
        data_path: Path to dataset
        sample_size: Number of households to sample (None = all)
        
    Returns:
        Ready-to-use dataframe with essential features for forecasting
    """
    # Load and prepare data
    df = load_smart_meter_data_kaggle(data_path)
    
    # Sample if requested
    if sample_size:
        df = sample_households(df, sample_size)
        print(f"🎲 Sampled {sample_size} households")
    
    print("🎉 DATA PREPARATION COMPLETE!")
    print(f"📊 Shape: {df.shape}")
    print(f"🏠 Households: {df['LCLid'].nunique()}")
    print(f"📅 Date range: {df['day'].min()} to {df['day'].max()}")
    
    # Show feature summary
    features = get_essential_features(df)
    print(f"🔧 Features ready for modeling: {len(features)}")
    
    return df

if __name__ == "__main__":
    print("🔧 Kaggle Utils - Streamlined Smart Meter Data Preparation")
    print("Usage: df = prepare_smart_meter_data()")
    print("       train_df, test_df = train_test_split_by_date(df)")
    print("       features = get_essential_features(df)") 