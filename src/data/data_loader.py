"""
📂 DATA LOADER - Pure Data Loading Functions
============================================

Pure loading functions for smart meter datasets (no cleaning).

Author: Shruthi Simha Chippagiri  
Date: 2025
"""

import pandas as pd
import numpy as np
import glob
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def load_half_hourly_data(path_glob: str) -> pd.DataFrame:
    """
    Load half-hourly consumption data from multiple CSV files
    
    Args:
        path_glob: Glob pattern for CSV files (e.g., "block_*.csv")
        
    Returns:
        Raw combined dataframe with all consumption data
    """
    files = glob.glob(path_glob)
    print(f"📂 Found {len(files)} consumption files")
    
    df_list = []
    for f in files:
        df_part = pd.read_csv(f, parse_dates=["day"])
        df_list.append(df_part)
    
    df_hh = pd.concat(df_list, ignore_index=True)
    print(f"✅ Loaded {df_hh.shape[0]:,} consumption records")
    return df_hh

def load_household_data(filepath: str) -> pd.DataFrame:
    """
    Load household information (ACORN, tariff)
    
    Args:
        filepath: Path to household CSV file
        
    Returns:
        Raw household dataframe
    """
    print("📂 Loading household data...")
    household_data = pd.read_csv(filepath)
    print(f"✅ Loaded {household_data.shape[0]:,} households")
    return household_data

def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load daily weather data
    
    Args:
        filepath: Path to weather CSV file
        
    Returns:
        Raw weather dataframe
    """
    print("📂 Loading weather data...")
    weather_data = pd.read_csv(filepath, parse_dates=["time"])
    print(f"✅ Loaded {weather_data.shape[0]:,} weather records")
    return weather_data

def load_holiday_data(filepath: str) -> pd.DataFrame:
    """
    Load UK bank holiday data
    
    Args:
        filepath: Path to holiday CSV file
        
    Returns:
        Raw holiday dataframe
    """
    print("📂 Loading holiday data...")
    holiday_data = pd.read_csv(filepath, parse_dates=["Bank holidays"])
    print(f"✅ Loaded {holiday_data.shape[0]:,} holiday records")
    return holiday_data

def load_all_raw_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets without any processing
    
    Args:
        data_path: Base path to data directory
        
    Returns:
        Dictionary containing all raw dataframes
    """
    print("🚀 LOADING ALL RAW DATA")
    print("=" * 30)
    
    raw_data = {}
    
    # Load all datasets
    raw_data['consumption'] = load_half_hourly_data(f"{data_path}/hhblock_dataset/hhblock_dataset/block_*.csv")
    raw_data['household'] = load_household_data(f"{data_path}/informations_households.csv")
    raw_data['weather'] = load_weather_data(f"{data_path}/weather_daily_darksky.csv")
    raw_data['holiday'] = load_holiday_data(f"{data_path}/uk_bank_holidays.csv")
    
    print("🎉 ALL RAW DATA LOADED")
    return raw_data

if __name__ == "__main__":
    print("📂 Data Loader Module")
    print("Usage: from src.data.data_loader import load_all_raw_data") 