"""
🧹 DATA CLEANER - Data Cleaning and Merging Functions
====================================================

Functions for cleaning and merging raw smart meter data.

Author: Shruthi Simha Chippagiri  
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

def clean_consumption_data(df: pd.DataFrame, 
                         missing_threshold: float = 0.2,
                         min_days: int = 30) -> pd.DataFrame:
    """
    Clean consumption data - remove bad households and interpolate gaps
    
    Args:
        df: Raw consumption dataframe with half-hourly data
        missing_threshold: Max missing ratio per day (0.2 = 20%)
        min_days: Minimum days per household
        
    Returns:
        Cleaned consumption dataframe
    """
    print("🧹 Cleaning consumption data...")
    
    # Define half-hour columns
    hh_cols = [f"hh_{i}" for i in range(48)]
    
    initial_rows = len(df)
    
    # Step 1: Drop rows with >20% missing values
    df["missing_ratio"] = df[hh_cols].isna().mean(axis=1)
    df = df[df["missing_ratio"] < missing_threshold].copy()
    print(f"   ✅ Removed {initial_rows - len(df):,} rows with >{missing_threshold*100}% missing")
    
    # Step 2: Keep only households with ≥30 valid days
    valid_counts = df["LCLid"].value_counts()
    keep_ids = valid_counts[valid_counts >= min_days].index
    df = df[df["LCLid"].isin(keep_ids)]
    print(f"   ✅ Kept {len(keep_ids):,} households with ≥{min_days} days")
    
    # Step 3: Interpolate short gaps efficiently
    print("   🔧 Interpolating missing values...")
    hh_matrix = df[hh_cols].to_numpy()
    for i in range(hh_matrix.shape[0]):
        x = hh_matrix[i]
        if np.isnan(x).any():
            n = len(x)
            not_nan = ~np.isnan(x)
            if not_nan.sum() > 0:
                hh_matrix[i] = np.interp(np.arange(n), np.where(not_nan)[0], x[not_nan])
    df[hh_cols] = hh_matrix
    
    # Drop temporary column
    df = df.drop(['missing_ratio'], axis=1)
    
    print(f"✅ Consumption data cleaned: {df.shape}")
    return df

def prepare_household_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare household data for merging
    
    Args:
        df: Raw household dataframe
        
    Returns:
        Prepared household dataframe
    """
    print("🏠 Preparing household data...")
    
    # Select key columns
    household_cols = ['LCLid', 'Acorn', 'Acorn_grouped', 'stdorToU']
    df_household = df[household_cols].copy()
    
    print(f"✅ Household data prepared: {df_household.shape}")
    return df_household

def prepare_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean weather data
    
    Args:
        df: Raw weather dataframe
        
    Returns:
        Cleaned weather dataframe
    """
    print("🌤️ Preparing weather data...")
    
    # Select key columns and rename
    weather_cols = ["time", "temperatureMax", "temperatureMin", "humidity", "windSpeed", "cloudCover"]
    df_weather = df[weather_cols].copy().rename(columns={"time": "day"})
    
    # Normalize day to date-only
    df_weather["day"] = pd.to_datetime(df_weather["day"]).dt.date
    
    # 🔍 DIAGNOSTIC: Check for duplicate dates (DST transitions)
    day_counts = df_weather.groupby("day").size().sort_values(ascending=False)
    duplicate_days = day_counts[day_counts > 1]
    
    if len(duplicate_days) > 0:
        print(f"   ⚠️ Found {len(duplicate_days)} dates with multiple records (likely DST transitions):")
        duplicate_dates = [pd.Timestamp(day) for day in duplicate_days.index]
        print(f"   📝 Duplicate dates: {duplicate_dates}")
        print(f"   📊 Counts per duplicate date:\n{duplicate_days}")
        
        # Remove duplicates, keeping the first occurrence
        initial_count = len(df_weather)
        df_weather = df_weather.drop_duplicates(subset=["day"], keep='first')
        final_count = len(df_weather)
        
        print(f"   ✅ Removed {initial_count - final_count} DST duplicate rows")
        print(f"   📊 Weather data: {initial_count} → {final_count} rows")
    else:
        print("   ✅ No duplicate dates found - weather data looks clean")
    
    # Clean weather data - interpolate missing values
    df_weather['day'] = pd.to_datetime(df_weather['day'], errors='coerce')
    df_weather = df_weather.sort_values('day').set_index('day')
    
    weather_feature_cols = ['cloudCover', 'windSpeed', 'humidity', 'temperatureMin', 'temperatureMax']
    df_weather[weather_feature_cols] = (
        df_weather[weather_feature_cols]
        .interpolate(method='time')
        .fillna(method='bfill')
        .fillna(method='ffill')
    )
    
    df_weather = df_weather.reset_index()
    df_weather["day"] = df_weather["day"].dt.date
    
    print(f"✅ Weather data prepared: {df_weather.shape}")
    return df_weather

def prepare_holiday_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare holiday data for merging
    
    Args:
        df: Raw holiday dataframe
        
    Returns:
        Prepared holiday dataframe
    """
    print("🎉 Preparing holiday data...")
    
    # Rename for consistency
    df_holiday = df.rename(columns={
        "Bank holidays": "day",
        "Type": "holiday_type"
    }).copy()
    
    df_holiday["day"] = pd.to_datetime(df_holiday["day"]).dt.date
    
    # Map holidays to broader categories
    holiday_mapping = {
        'New Year?s Day': 'New Year',
        'New Year?s Day (substitute day)': 'New Year',
        'Christmas Day': 'Christmas',
        'Boxing Day': 'Christmas',
        'Easter Monday': 'Easter',
        'Good Friday': 'Easter',
        'Summer bank holiday': 'Bank Holiday',
        'Early May bank holiday': 'Bank Holiday',
        'Spring bank holiday': 'Bank Holiday',
        'Spring bank holiday (substitute day)': 'Bank Holiday',
        'Queen?s Diamond Jubilee (extra bank holiday)': 'Bank Holiday'
    }
    
    df_holiday['holiday_category'] = df_holiday['holiday_type'].map(holiday_mapping).fillna('Regular Day')
    
    print(f"✅ Holiday data prepared: {df_holiday.shape}")
    return df_holiday[["day", "holiday_category"]]

def merge_all_data(consumption_df: pd.DataFrame,
                  household_df: pd.DataFrame,
                  weather_df: pd.DataFrame,
                  holiday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all cleaned datasets together
    
    Args:
        consumption_df: Cleaned consumption data
        household_df: Prepared household data
        weather_df: Prepared weather data
        holiday_df: Prepared holiday data
        
    Returns:
        Merged dataframe
    """
    print("🔗 Merging all datasets...")
    
    # Normalize day column in consumption data
    consumption_df["day"] = pd.to_datetime(consumption_df["day"]).dt.date
    
    # Merge household info
    df_merged = consumption_df.merge(household_df, on='LCLid', how='left')
    print(f"   ✅ After household merge: {df_merged.shape}")
    
    # Merge weather data
    df_merged = df_merged.merge(weather_df, on="day", how="left")
    print(f"   ✅ After weather merge: {df_merged.shape}")
    
    # Merge holiday data
    df_merged = df_merged.merge(holiday_df, on="day", how="left")
    print(f"   ✅ After holiday merge: {df_merged.shape}")
    
    # Fill missing holiday category
    df_merged['holiday_category'] = df_merged['holiday_category'].fillna('Regular Day')
    
    print("✅ All data merged successfully")
    return df_merged

def add_basic_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic temporal features after cleaning
    
    Args:
        df: Merged dataframe
        
    Returns:
        Dataframe with temporal features
    """
    print("📅 Adding basic temporal features...")
    
    # Convert to datetime for processing
    df["day"] = pd.to_datetime(df["day"])
    
    # Add temporal features
    df["month"] = df["day"].dt.month
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    
    print(f"✅ Temporal features added")
    return df

def clean_and_merge_all_data(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Main function to clean and merge all raw data
    
    Args:
        raw_data: Dictionary of raw dataframes from data_loader
        
    Returns:
        Complete cleaned and merged dataframe
    """
    print("🚀 CLEANING AND MERGING ALL DATA")
    print("=" * 40)
    
    # Clean and prepare each dataset
    consumption_clean = clean_consumption_data(raw_data['consumption'])
    household_clean = prepare_household_data(raw_data['household'])
    weather_clean = prepare_weather_data(raw_data['weather'])
    holiday_clean = prepare_holiday_data(raw_data['holiday'])
    
    # Merge everything
    df_final = merge_all_data(consumption_clean, household_clean, weather_clean, holiday_clean)
    
    # Add basic temporal features
    df_final = add_basic_temporal_features(df_final)
    
    print(f"🎉 FINAL CLEANED DATASET: {df_final.shape}")
    return df_final

if __name__ == "__main__":
    print("🧹 Data Cleaner Module")
    print("Usage: from src.data.data_cleaner import clean_and_merge_all_data") 