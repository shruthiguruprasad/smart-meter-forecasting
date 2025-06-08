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

def reduce_memory_footprint(df, verbose=True):
    import numpy as np
    import pandas as pd

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 2**8:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 2**16:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 2**32:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            else:  # float
                if np.finfo(np.float16).min < c_min < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif np.finfo(np.float32).min < c_min < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Reduced memory usage: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df

def viz_safe_reduce_memory(df, verbose=True, preserve_float_precision=True):
    """
    Reduce memory usage of a DataFrame while preserving visualization compatibility.
    
    This is a safer version of reduce_memory_footprint that avoids using float16
    to prevent NaN issues in visualizations and calculations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to optimize
    verbose : bool, default=True
        Whether to print memory reduction information
    preserve_float_precision : bool, default=True
        If True, will use float32 instead of float16 to preserve precision
        
    Returns:
    --------
    pandas.DataFrame
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Initial memory usage: {start_mem:.2f} MB")

    # Create a copy to avoid modifying the original
    result = df.copy()
    
    for col in result.columns:
        col_type = result[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = result[col].min()
            c_max = result[col].max()

            # Handle integer columns
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 2**8:
                        result[col] = result[col].astype(np.uint8)
                    elif c_max < 2**16:
                        result[col] = result[col].astype(np.uint16)
                    elif c_max < 2**32:
                        result[col] = result[col].astype(np.uint32)
                    else:
                        result[col] = result[col].astype(np.uint64)
                else:
                    if np.iinfo(np.int8).min < c_min < np.iinfo(np.int8).max:
                        result[col] = result[col].astype(np.int8)
                    elif np.iinfo(np.int16).min < c_min < np.iinfo(np.int16).max:
                        result[col] = result[col].astype(np.int16)
                    elif np.iinfo(np.int32).min < c_min < np.iinfo(np.int32).max:
                        result[col] = result[col].astype(np.int32)
                    else:
                        result[col] = result[col].astype(np.int64)

            # Handle float columns - avoid float16 for visualization safety
            elif pd.api.types.is_float_dtype(col_type):
                if preserve_float_precision:
                    # Skip float16 and go straight to float32 for better precision
                    if np.finfo(np.float32).min < c_min < np.finfo(np.float32).max:
                        result[col] = result[col].astype(np.float32)
                    else:
                        result[col] = result[col].astype(np.float64)
                else:
                    # Original behavior but with careful handling of NaN values
                    if np.finfo(np.float16).min < c_min < np.finfo(np.float16).max:
                        # Convert but then check for NaN and repair if needed
                        temp = result[col].astype(np.float16)
                        if temp.isna().sum() > result[col].isna().sum():
                            # If NaNs were introduced, revert to float32
                            result[col] = result[col].astype(np.float32)
                        else:
                            result[col] = temp
                    elif np.finfo(np.float32).min < c_min < np.finfo(np.float32).max:
                        result[col] = result[col].astype(np.float32)
                    else:
                        result[col] = result[col].astype(np.float64)

        # Handle categorical columns
        elif col_type == object:
            num_unique = result[col].nunique()
            num_total = len(result[col])
            if num_unique / num_total < 0.5:
                result[col] = result[col].astype('category')

    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Reduced memory usage: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
        print(f"Memory saved: {start_mem - end_mem:.2f} MB")

    return result