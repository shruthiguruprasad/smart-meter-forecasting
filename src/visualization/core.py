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
    if 'daily_variability' not in df.columns and 'std_kwh' in df.columns and 'mean_kwh' in df.columns:
        df['daily_variability'] = df['std_kwh'] / df['mean_kwh']
        df['daily_variability'] = df['daily_variability'].replace([np.inf, -np.inf], np.nan)
        print("✅ Created 'daily_variability' from std_kwh/mean_kwh")
    
    # 5. Create heating_degree_days if not present
    if 'heating_degree_days' not in df.columns and 'temperatureMax' in df.columns and 'temperatureMin' in df.columns:
        base_temp = 18.0  # Standard UK base temperature for HDD
        df['avg_temp'] = (df['temperatureMax'] + df['temperatureMin']) / 2
        df['heating_degree_days'] = np.maximum(0, base_temp - df['avg_temp'])
        print("✅ Created 'heating_degree_days' from temperature data")
    
    # 6. Create cooling_degree_days if not present
    if 'cooling_degree_days' not in df.columns and 'temperatureMax' in df.columns and 'temperatureMin' in df.columns:
        base_temp = 18.0  # Standard UK base temperature for CDD
        df['avg_temp'] = (df['temperatureMax'] + df['temperatureMin']) / 2
        df['cooling_degree_days'] = np.maximum(0, df['avg_temp'] - base_temp)
        print("✅ Created 'cooling_degree_days' from temperature data")
    
    # 7. Create time-of-day consumption columns if missing but have half-hourly data
    tod_cols = ['morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh']
    if any(col not in df.columns for col in tod_cols) and all(f'hh_{i}' in df.columns for i in range(48)):
        # Define time-of-day ranges (indices in half-hourly data)
        # Morning: 7:00-12:00 (indices 14-23)
        # Afternoon: 12:00-17:00 (indices 24-33)
        # Evening: 17:00-22:00 (indices 34-43)
        # Night: 22:00-7:00 (indices 44-47, 0-13)
        
        morning_cols = [f'hh_{i}' for i in range(14, 24)]
        afternoon_cols = [f'hh_{i}' for i in range(24, 34)]
        evening_cols = [f'hh_{i}' for i in range(34, 44)]
        night_cols = [f'hh_{i}' for i in range(44, 48)] + [f'hh_{i}' for i in range(0, 14)]
        
        if 'morning_kwh' not in df.columns:
            df['morning_kwh'] = df[morning_cols].sum(axis=1)
            print("✅ Created 'morning_kwh' from half-hourly data")
        
        if 'afternoon_kwh' not in df.columns:
            df['afternoon_kwh'] = df[afternoon_cols].sum(axis=1)
            print("✅ Created 'afternoon_kwh' from half-hourly data")
        
        if 'evening_kwh' not in df.columns:
            df['evening_kwh'] = df[evening_cols].sum(axis=1)
            print("✅ Created 'evening_kwh' from half-hourly data")
        
        if 'night_kwh' not in df.columns:
            df['night_kwh'] = df[night_cols].sum(axis=1)
            print("✅ Created 'night_kwh' from half-hourly data")
    
    # 8. Create total_kwh if not present
    if 'total_kwh' not in df.columns:
        if all(f'hh_{i}' in df.columns for i in range(48)):
            hh_cols = [f'hh_{i}' for i in range(48)]
            df['total_kwh'] = df[hh_cols].sum(axis=1)
            print("✅ Created 'total_kwh' from half-hourly data")
        elif all(col in df.columns for col in tod_cols):
            df['total_kwh'] = df[tod_cols].sum(axis=1)
            print("✅ Created 'total_kwh' from time-of-day columns")
    
    # 9. Create peak_hour if not present but have half-hourly data
    if 'peak_hour' not in df.columns and all(f'hh_{i}' in df.columns for i in range(48)):
        # Find the half-hour with maximum consumption
        hh_cols = [f'hh_{i}' for i in range(48)]
        df['peak_hour'] = df[hh_cols].idxmax(axis=1).str.extract('(\d+)').astype(float) / 2
        print("✅ Created 'peak_hour' from half-hourly data")
    
    print("✅ Feature preparation complete!")
    return df
