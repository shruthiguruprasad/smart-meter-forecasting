"""
🌤️ WEATHER FEATURES - Enhanced Weather Features for Forecasting
==============================================================

Enhanced weather features for electricity consumption forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temperature features for forecasting
    
    Args:
        df: Dataframe with temperature columns
        
    Returns:
        Dataframe with temperature features added
    """
    print("🌡️ Creating temperature features...")
    
    temp_cols = ["temperatureMax", "temperatureMin"]
    missing_cols = [col for col in temp_cols if col not in df.columns]
    
    if missing_cols:
        print(f"   ⚠️ Missing columns: {missing_cols}")
        return df
    
    # Basic temperature features
    df["temp_range"] = df["temperatureMax"] - df["temperatureMin"]
    df["temp_avg"] = (df["temperatureMax"] + df["temperatureMin"]) / 2
    
    # Degree days (critical for energy consumption)
    df["heating_degree_days"] = np.maximum(15 - df["temp_avg"], 0)  # base 15°C
    df["cooling_degree_days"] = np.maximum(df["temp_avg"] - 22, 0)  # base 22°C
    
    # Temperature extremes
    df["is_very_cold"] = (df["temp_avg"] < 5).astype(int)
    df["is_very_hot"] = (df["temp_avg"] > 25).astype(int)
    
    print("   ✅ Created temperature features")
    return df

def create_weather_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather condition features from available data
    
    Args:
        df: Dataframe with weather columns
        
    Returns:
        Dataframe with weather condition features
    """
    print("🌦️ Creating weather condition features...")
    
    weather_cols = ["humidity", "windSpeed", "cloudCover"]
    available_cols = [col for col in weather_cols if col in df.columns]
    
    if not available_cols:
        print("   ⚠️ No weather condition columns found")
        return df
    
    # Humidity features
    if "humidity" in df.columns:
        df["is_high_humidity"] = (df["humidity"] > 0.8).astype(int)
        df["is_low_humidity"] = (df["humidity"] < 0.3).astype(int)
    
    # Wind features
    if "windSpeed" in df.columns:
        df["is_windy"] = (df["windSpeed"] > df["windSpeed"].quantile(0.75)).astype(int)
        df["wind_calm"] = (df["windSpeed"] < df["windSpeed"].quantile(0.25)).astype(int)
    
    # Cloud cover features
    if "cloudCover" in df.columns:
        df["is_cloudy"] = (df["cloudCover"] > 0.7).astype(int)
        df["is_clear"] = (df["cloudCover"] < 0.3).astype(int)
    
    print(f"   ✅ Created weather condition features from {available_cols}")
    return df

def create_temperature_impact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temperature impact features per household
    
    Args:
        df: Dataframe with temperature and consumption data
        
    Returns:
        Dataframe with temperature impact features
    """
    print("🌡️ Creating temperature impact features...")
    
    if all(col in df.columns for col in ["total_kwh", "temp_avg", "LCLid"]):
        # Temperature sensitivity per household
        temp_corr = (
            df.groupby("LCLid")[["total_kwh", "temp_avg"]]
            .apply(lambda g: g["total_kwh"].corr(g["temp_avg"]) if len(g) > 1 else np.nan)
        )
        df["temp_sensitivity"] = df["LCLid"].map(temp_corr)
        
        print("   ✅ Created temperature impact features")
    else:
        print("   ⚠️ Required columns not found for temperature impact")
    
    return df

def create_seasonal_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create seasonal weather interaction features
    
    Args:
        df: Dataframe with weather and season data
        
    Returns:
        Dataframe with seasonal weather features
    """
    print("🌨️ Creating seasonal weather features...")
    
    if all(col in df.columns for col in ["season", "temp_avg"]):
        # Season-temperature interactions
        df["winter_heating_need"] = (
            (df["season"] == "Winter") * df["heating_degree_days"]
        )
        df["summer_cooling_need"] = (
            (df["season"] == "Summer") * df["cooling_degree_days"]
        )
        
        print("   ✅ Created seasonal weather features")
    else:
        print("   ⚠️ Season or temperature columns not found")
    
    return df

def create_all_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all weather features for forecasting
    
    Args:
        df: Input dataframe with weather columns
        
    Returns:
        Dataframe with all weather features
    """
    print("🚀 Creating All Weather Features")
    print("=" * 35)
    
    # Create all weather feature types
    df = create_temperature_features(df)
    df = create_weather_condition_features(df)
    df = create_temperature_impact_features(df)
    df = create_seasonal_weather_features(df)
    
    print("✅ All weather features created")
    return df

if __name__ == "__main__":
    print("🌤️ Weather Features Module")
    print("Usage: from src.features.weather_features import create_all_weather_features") 