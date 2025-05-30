"""
🔧 HELPERS - Utility Functions
=============================

Simple utility functions for common tasks.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def save_data(df: pd.DataFrame, filepath: str, format: str = "csv") -> None:
    """
    Save dataframe to file
    
    Args:
        df: Dataframe to save
        filepath: Output file path
        format: File format (csv, pickle, parquet)
    """
    print(f"💾 Saving data to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format.lower() == "csv":
        df.to_csv(filepath, index=False)
    elif format.lower() == "pickle":
        df.to_pickle(filepath)
    elif format.lower() == "parquet":
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"   ✅ Saved {df.shape[0]} rows, {df.shape[1]} columns")

def load_data(filepath: str, format: str = None) -> pd.DataFrame:
    """
    Load dataframe from file
    
    Args:
        filepath: Input file path
        format: File format (auto-detect if None)
        
    Returns:
        Loaded dataframe
    """
    print(f"📂 Loading data from {filepath}...")
    
    if format is None:
        # Auto-detect format from extension
        ext = filepath.split('.')[-1].lower()
        format = ext
    
    if format == "csv":
        df = pd.read_csv(filepath)
    elif format == "pickle":
        df = pd.read_pickle(filepath)
    elif format == "parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"   ✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model
    
    Args:
        model: Trained model object
        filepath: Output file path
    """
    print(f"💾 Saving model to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    
    print(f"   ✅ Model saved")

def load_model(filepath: str) -> Any:
    """
    Load trained model
    
    Args:
        filepath: Model file path
        
    Returns:
        Loaded model object
    """
    print(f"📂 Loading model from {filepath}...")
    
    model = joblib.load(filepath)
    
    print(f"   ✅ Model loaded")
    return model

def save_results(results: Dict, filepath: str) -> None:
    """
    Save results dictionary to JSON
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    print(f"💾 Saving results to {filepath}...")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    clean_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            clean_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            clean_results[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            clean_results[key] = float(value)
        else:
            clean_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"   ✅ Results saved")

def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file
    
    Args:
        filepath: Results file path
        
    Returns:
        Results dictionary
    """
    print(f"📂 Loading results from {filepath}...")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"   ✅ Results loaded")
    return results

def split_data_by_date(df: pd.DataFrame, 
                      date_col: str = "day",
                      train_end: str = "2013-12-31",
                      test_start: str = "2014-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by date
    
    Args:
        df: Input dataframe
        date_col: Date column name
        train_end: End date for training (inclusive)
        test_start: Start date for testing (inclusive)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"✂️ Splitting data by date...")
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    
    # Split data
    train_df = df[df[date_col] <= train_end].copy()
    test_df = df[df[date_col] >= test_start].copy()
    
    print(f"   📊 Train: {len(train_df):,} rows ({train_df[date_col].min()} to {train_df[date_col].max()})")
    print(f"   📊 Test: {len(test_df):,} rows ({test_df[date_col].min()} to {test_df[date_col].max()})")
    
    return train_df, test_df

def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """
    Get feature columns excluding ID, date, and target columns
    
    Args:
        df: Input dataframe
        exclude_patterns: Patterns to exclude from features
        
    Returns:
        List of feature column names
    """
    if exclude_patterns is None:
        exclude_patterns = ["LCLid", "day", "target_", "hh_"]
    
    feature_cols = []
    for col in df.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            feature_cols.append(col)
    
    print(f"🎯 Selected {len(feature_cols)} feature columns")
    return feature_cols

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by optimizing dtypes
    
    Args:
        df: Input dataframe
        
    Returns:
        Memory-optimized dataframe
    """
    print("🗜️ Reducing memory usage...")
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        if df[col].dtype == 'object':
            continue
            
        col_type = df[col].dtype
        
        if str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        elif str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem * 100
    
    print(f"   ✅ Memory reduced: {start_mem:.1f}MB → {end_mem:.1f}MB ({reduction:.1f}%)")
    return df

def get_household_sample(df: pd.DataFrame, 
                        n_households: int = 100,
                        group_col: str = "LCLid",
                        random_state: int = 42) -> pd.DataFrame:
    """
    Get sample of households for testing
    
    Args:
        df: Input dataframe
        n_households: Number of households to sample
        group_col: Household ID column
        random_state: Random seed
        
    Returns:
        Sampled dataframe
    """
    print(f"🎲 Sampling {n_households} households...")
    
    # Get unique households
    households = df[group_col].unique()
    
    # Sample households
    np.random.seed(random_state)
    sampled_households = np.random.choice(
        households, 
        size=min(n_households, len(households)), 
        replace=False
    )
    
    # Filter dataframe
    sample_df = df[df[group_col].isin(sampled_households)].copy()
    
    print(f"   ✅ Sampled {len(sampled_households)} households, {len(sample_df):,} rows")
    return sample_df

def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Check data quality and return summary
    
    Args:
        df: Input dataframe
        
    Returns:
        Data quality summary
    """
    print("🔍 Checking data quality...")
    
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Check for problematic values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['infinite_values'] = {}
    quality_report['zero_variance'] = {}
    
    for col in numeric_cols:
        quality_report['infinite_values'][col] = np.isinf(df[col]).sum()
        quality_report['zero_variance'][col] = df[col].var() < 1e-10
    
    print("   ✅ Data quality check complete")
    return quality_report

def print_data_quality(df: pd.DataFrame):
    """
    Print formatted data quality report
    
    Args:
        df: Input dataframe
    """
    quality = check_data_quality(df)
    
    print("\n🔍 DATA QUALITY REPORT")
    print("=" * 30)
    print(f"📊 Shape: {quality['shape']}")
    print(f"🗂️ Memory: {quality['memory_usage_mb']:.1f} MB")
    print(f"🔄 Duplicates: {quality['duplicate_rows']}")
    
    # Missing values
    missing = {k: v for k, v in quality['missing_percentage'].items() if v > 0}
    if missing:
        print("\n❌ Missing Values:")
        for col, pct in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {col}: {pct:.1f}%")
    
    # Infinite values  
    infinite = {k: v for k, v in quality['infinite_values'].items() if v > 0}
    if infinite:
        print("\n♾️ Infinite Values:")
        for col, count in infinite.items():
            print(f"   {col}: {count}")

if __name__ == "__main__":
    print("🔧 Helpers Module")
    print("Usage: from src.utils.helpers import save_data, load_data") 