"""
📋 CONSTANTS - Project Constants
===============================

Central place for all project constants and configuration values.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import os
from typing import List, Dict

# ===========================
# DATA CONSTANTS
# ===========================

# Half-hourly columns (48 readings per day)
HH_COLS = [f"hh_{i}" for i in range(48)]

# Data file patterns
DATA_PATTERNS = {
    'consumption': "hhblock_dataset/hhblock_dataset/block_*.csv",
    'weather': "weather_daily_darksky.csv", 
    'household': "informations_households.csv",
    'holiday': "uk_bank_holidays.csv"
}

# Data quality thresholds
DATA_QUALITY = {
    'missing_threshold': 0.2,  # Max 20% missing per day
    'min_days_per_household': 30,  # Min days of data per household
    'outlier_z_score': 3.0,  # Z-score threshold for outliers
    'max_daily_consumption': 100.0  # Max reasonable daily consumption (kWh)
}

# ===========================
# FEATURE ENGINEERING
# ===========================

# Time periods for features
TIME_PERIODS = {
    'morning_hours': list(range(6, 12)),  # 6 AM - 12 PM
    'afternoon_hours': list(range(12, 18)),  # 12 PM - 6 PM  
    'evening_hours': list(range(18, 24)),  # 6 PM - 12 AM
    'night_hours': list(range(0, 6)),  # 12 AM - 6 AM
    'peak_hours': list(range(17, 21)),  # 5 PM - 9 PM (UK peak)
    'off_peak_hours': list(range(23, 6)) + list(range(7, 16))  # Off-peak
}

# Half-hourly indices for time periods
HH_TIME_PERIODS = {
    'morning': list(range(12, 24)),  # 6 AM - 12 PM (hh_12 to hh_23)
    'afternoon': list(range(24, 36)),  # 12 PM - 6 PM (hh_24 to hh_35)
    'evening': list(range(36, 48)),  # 6 PM - 12 AM (hh_36 to hh_47)
    'night': list(range(0, 12)),  # 12 AM - 6 AM (hh_0 to hh_11)
    'peak': list(range(34, 42)),  # 5 PM - 9 PM (hh_34 to hh_41)
    'off_peak': list(range(46, 48)) + list(range(0, 12)) + list(range(14, 32))
}

# Lag periods for time series features
LAG_PERIODS = {
    'short': [1, 2, 3],  # 1-3 days
    'medium': [7, 14],  # 1-2 weeks
    'long': [30, 60, 90]  # 1-3 months
}

# Rolling window sizes
ROLLING_WINDOWS = {
    'short': [3, 7],  # 3-7 days
    'medium': [14, 30],  # 2 weeks - 1 month
    'long': [60, 90]  # 2-3 months
}

# ===========================
# MODEL PARAMETERS
# ===========================

# Forecast horizons
FORECAST_HORIZONS = [1, 7, 30]  # 1 day, 1 week, 1 month

# Model types
MODEL_TYPES = {
    'statistical': ['prophet', 'arima'],
    'machine_learning': ['xgboost', 'lightgbm', 'random_forest'],
    'deep_learning': ['lstm', 'transformer'],
    'ensemble': ['voting', 'stacking']
}

# Default model parameters
DEFAULT_PARAMS = {
    'prophet': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
}

# ===========================
# EVALUATION METRICS
# ===========================

# Metrics to compute
EVALUATION_METRICS = [
    'mae',  # Mean Absolute Error
    'rmse',  # Root Mean Square Error
    'mape',  # Mean Absolute Percentage Error
    'r2',  # R-squared
    'smape'  # Symmetric Mean Absolute Percentage Error
]

# Cross-validation parameters
CV_PARAMS = {
    'n_splits': 5,
    'test_size': 0.2,
    'gap': 7,  # Days gap between train/test
    'strategy': 'time_series'  # Respect temporal order
}

# ===========================
# WEATHER CONSTANTS
# ===========================

# UK-specific parameters
UK_WEATHER = {
    'base_temperature': 15.0,  # Base temp for degree days (°C)
    'cold_threshold': 0.0,  # Very cold threshold (°C)
    'hot_threshold': 25.0,  # Very hot threshold (°C)
    'comfortable_temp_range': [15, 22],  # Comfortable range (°C)
    'comfortable_humidity_range': [0.4, 0.6]  # Comfortable humidity range
}

# Season definitions for UK
UK_SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

# ===========================
# FILE PATHS
# ===========================

# Directory structure
DIRS = {
    'data': 'data',
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'models': 'models',
    'results': 'results',
    'plots': 'results/plots',
    'logs': 'logs'
}

# Default file names
DEFAULT_FILES = {
    'processed_data': 'processed_smart_meter_data.csv',
    'features': 'features.csv',
    'model_results': 'model_results.json',
    'feature_importance': 'feature_importance.csv'
}

# ===========================
# VISUALIZATION
# ===========================

# Plot settings
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'save_format': 'png'
}

# Colors for different model types
MODEL_COLORS = {
    'prophet': '#1f77b4',
    'xgboost': '#ff7f0e',
    'lightgbm': '#2ca02c',
    'lstm': '#d62728',
    'ensemble': '#9467bd'
}

# ===========================
# LOGGING
# ===========================

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(DIRS['logs'], 'smart_meter_forecasting.log')
}

# ===========================
# RANDOM SEEDS
# ===========================

# Seeds for reproducibility
RANDOM_SEEDS = {
    'data_split': 42,
    'model_training': 42,
    'cross_validation': 42,
    'feature_selection': 42
}

# ===========================
# HELPER FUNCTIONS
# ===========================

def get_hh_indices_for_hours(hours: List[int]) -> List[int]:
    """
    Convert hour list to half-hourly indices
    
    Args:
        hours: List of hours (0-23)
        
    Returns:
        List of half-hourly indices (0-47)
    """
    indices = []
    for hour in hours:
        indices.extend([hour * 2, hour * 2 + 1])
    return [i for i in indices if 0 <= i <= 47]

def get_season_for_month(month: int) -> str:
    """
    Get season name for given month
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name
    """
    for season, months in UK_SEASONS.items():
        if month in months:
            return season
    return "Unknown"

if __name__ == "__main__":
    print("📋 Constants Module")
    print(f"HH Columns: {len(HH_COLS)}")
    print(f"Time Periods: {list(TIME_PERIODS.keys())}")
    print(f"Model Types: {list(MODEL_TYPES.keys())}")
    print(f"Evaluation Metrics: {EVALUATION_METRICS}") 