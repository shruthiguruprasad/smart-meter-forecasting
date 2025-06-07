import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
from scipy import stats

# Use existing hh_cols definition from guru.py
hh_cols = [f"hh_{i}" for i in range(48)]

def check_dataframe_columns(df_hh):
    """
    Check what columns are available in the dataframe for plotting
    """
    print("🔍 DATAFRAME COLUMN CHECK")
    print("=" * 50)
    
    required_columns = {
        "Basic Features": [
            "LCLid", "day", "total_kwh", "mean_kwh", "std_kwh", "peak_kwh"
        ],
        "Time Features": [
            "dayofweek", "is_weekend", "season", "month", "is_holiday"
        ],
        "Weather Features": [
            "temperatureMax", "temperatureMin", "humidity"
        ],
        "Time-of-Day Features": [
            "morning_kwh", "afternoon_kwh", "evening_kwh", "night_kwh"
        ],
        "Advanced Features": [
            "is_weekday", "peak_to_mean_ratio", "day_night_ratio", 
            "daily_variability", "temp_impact"
        ],
        "Socio-Economic Features": [
            "Acorn_grouped", "stdorToU"
        ],
        "Half-Hourly Columns": hh_cols[:5] + ["..."] + hh_cols[-5:]  # Show first and last 5
    }
    
    for category, columns in required_columns.items():
        print(f"\n📊 {category}:")
        if category == "Half-Hourly Columns":
            # Special handling for half-hourly columns
            available_hh = [col for col in hh_cols if col in df_hh.columns]
            print(f"   ✅ Available: {len(available_hh)}/48 half-hourly columns")
            if len(available_hh) < 48:
                missing_hh = [col for col in hh_cols if col not in df_hh.columns]
                print(f"   ❌ Missing: {missing_hh[:5]}...")
        else:
            for col in columns:
                if col == "...":
                    continue
                status = "✅" if col in df_hh.columns else "❌"
                print(f"   {status} {col}")
    
    print(f"\n📈 Total columns in dataframe: {len(df_hh.columns)}")
    print(f"📊 DataFrame shape: {df_hh.shape}")
    
    return df_hh

def prepare_plotting_features(df_hh):
    """
    Ensure all required features exist for plotting functions
    Creates missing columns with fallback calculations
    """
    print("🔧 Preparing features for plotting...")
    
    # Create a copy to avoid modifying original data
    df = df_hh.copy()
    
    # 1. Fix is_weekday vs is_weekend inconsistency
    if 'is_weekend' in df.columns and 'is_weekday' not in df.columns:
        df['is_weekday'] = 1 - df['is_weekend']
        print("✅ Created 'is_weekday' from 'is_weekend'")
    elif 'is_weekday' not in df.columns and 'dayofweek' in df.columns:
        df['is_weekday'] = (df['dayofweek'] < 5).astype(int)
        print("✅ Created 'is_weekday' from 'dayofweek'")
    
    # 2. Create peak_to_mean_ratio if missing
    if 'peak_to_mean_ratio' not in df.columns:
        if 'peak_kwh' in df.columns and 'mean_kwh' in df.columns:
            df['peak_to_mean_ratio'] = df['peak_kwh'] / df['mean_kwh']
            df['peak_to_mean_ratio'] = df['peak_to_mean_ratio'].replace([np.inf, -np.inf], np.nan)
            print("✅ Created 'peak_to_mean_ratio' from peak_kwh/mean_kwh")
        else:
            print("⚠️  Cannot create 'peak_to_mean_ratio' - missing peak_kwh or mean_kwh")
    
    # 3. Create day_night_ratio if missing
    if 'day_night_ratio' not in df.columns:
        if all(col in df.columns for col in ['morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh']):
            day_consumption = df['morning_kwh'] + df['afternoon_kwh'] 
            night_consumption = df['evening_kwh'] + df['night_kwh']
            df['day_night_ratio'] = day_consumption / night_consumption
            df['day_night_ratio'] = df['day_night_ratio'].replace([np.inf, -np.inf], np.nan)
            print("✅ Created 'day_night_ratio' from time-of-day columns")
        else:
            print("⚠️  Cannot create 'day_night_ratio' - missing time-of-day columns")
    
    # 4. Create daily_variability if missing
    if 'daily_variability' not in df.columns:
        df['daily_variability'] = df[hh_cols].std(axis=1)
        print("✅ Created 'daily_variability' from half-hourly readings")
    
    # 5. Create temp_impact if missing (simplified proxy)
    if 'temp_impact' not in df.columns and 'temperatureMax' in df.columns:
        # Simple temperature impact approximation
        temp_corr = df.groupby('LCLid').apply(
            lambda x: x['total_kwh'].corr(x['temperatureMax']) if len(x) > 1 else 0
        )
        df['temp_impact'] = df['LCLid'].map(temp_corr)
        print("✅ Created 'temp_impact' as temperature-consumption correlation")
    
    # 6. Ensure other time-of-day features exist
    if not all(col in df.columns for col in ['morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh']):
        print("⚠️  Creating basic time-of-day features from half-hourly data...")
        df['morning_kwh'] = df[[f"hh_{i}" for i in range(6, 12)]].sum(axis=1)  # 3-6 AM
        df['afternoon_kwh'] = df[[f"hh_{i}" for i in range(24, 36)]].sum(axis=1)  # 12-6 PM
        df['evening_kwh'] = df[[f"hh_{i}" for i in range(36, 48)]].sum(axis=1)  # 6 PM-12 AM
        df['night_kwh'] = df[[f"hh_{i}" for i in range(0, 6)]].sum(axis=1)  # 12-3 AM
        print("✅ Created time-of-day features")
    
    # 7. Ensure basic stats exist
    required_basic_features = ['total_kwh', 'mean_kwh', 'std_kwh', 'peak_kwh']
    for feature in required_basic_features:
        if feature not in df.columns:
            if feature == 'total_kwh':
                df['total_kwh'] = df[hh_cols].sum(axis=1)
            elif feature == 'mean_kwh':
                df['mean_kwh'] = df[hh_cols].mean(axis=1)
            elif feature == 'std_kwh':
                df['std_kwh'] = df[hh_cols].std(axis=1)
            elif feature == 'peak_kwh':
                df['peak_kwh'] = df[hh_cols].max(axis=1)
            print(f"✅ Created '{feature}'")
    
    print(f"🎯 Feature preparation complete! DataFrame shape: {df.shape}")
    return df
