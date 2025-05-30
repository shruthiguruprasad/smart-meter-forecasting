"""
🎯 PREDICTIVE MODEL - Stage 0: XGBoost Training & SHAP Generation
==============================================================

Stage 0: Exploratory & Predictive Modeling
- Train XGBoost regressor on daily consumption
- Generate SHAP values for consumption driver analysis
- Prepare data for feature importance analysis

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def prepare_modeling_data(df: pd.DataFrame, 
                         target_col: str = "total_kwh",
                         test_size: float = 0.2,
                         random_state: int = 42) -> dict:
    """
    Prepare clean data for Stage 0 predictive modeling
    
    Args:
        df: Input dataframe with features
        target_col: Target variable name
        test_size: Test set proportion (0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with clean train/test data and metadata
    """
    print("📊 PREPARING DATA FOR STAGE 0 PREDICTIVE MODELING")
    print("=" * 52)
    
    # Get feature columns (exclude IDs, dates, raw half-hourly data)
    exclude_patterns = ["LCLid", "day", "hh_", "holiday_type"]
    feature_cols = []
    
    for col in df.columns:
        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
        # Skip if it looks like raw half-hourly data
        if col.startswith('hh_') and col.replace('hh_', '').isdigit():
            continue
        # Skip target column
        if col == target_col:
            continue
        feature_cols.append(col)
    
    print(f"📋 Selected {len(feature_cols)} features for modeling")
    print(f"🎯 Target variable: {target_col}")
    
    # Select relevant columns
    model_data = df[feature_cols + [target_col]].copy()
    
    # Remove rows with NaN values
    initial_rows = len(model_data)
    model_data = model_data.dropna()
    final_rows = len(model_data)
    
    print(f"🧹 Cleaned data: {initial_rows:,} → {final_rows:,} rows ({initial_rows-final_rows:,} removed)")
    
    # Prepare features and target
    X = model_data[feature_cols]
    y = model_data[target_col]
    
    # Train/test split (random for Stage 0 analysis)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Train set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    print(f"📊 Features: {len(feature_cols)} columns")
    
    # Feature groups for analysis
    feature_groups = {
        'weather': [c for c in feature_cols if any(x in c for x in ['temp', 'heating', 'cooling', 'humidity', 'wind', 'cloud'])],
        'calendar': [c for c in feature_cols if any(x in c for x in ['dayofweek', 'weekend', 'month', 'season', 'holiday', 'quarter'])],
        'socio_economic': [c for c in feature_cols if any(x in c for x in ['acorn', 'Acorn'])],
        'past_use': [c for c in feature_cols if any(x in c for x in ['lag', 'roll', 'delta', 'pct_change'])],
        'consumption_patterns': [c for c in feature_cols if any(x in c for x in ['ratio', 'variability', 'concentration', 'sharpness', 'load_factor'])],
        'time_of_day': [c for c in feature_cols if any(x in c for x in ['morning', 'afternoon', 'evening', 'night', 'peak_period', 'off_peak'])],
        'interactions': [c for c in feature_cols if any(x in c for x in ['weekend_heating', 'summer_cooling', 'holiday_'])]
    }
    
    # Show feature group summary
    print(f"\n📋 FEATURE GROUP SUMMARY:")
    for group, features in feature_groups.items():
        print(f"   {group:<20}: {len(features):2d} features")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'feature_groups': feature_groups,
        'target_col': target_col
    }

def train_xgboost_model(data_dict: dict, 
                       xgb_params: dict = None) -> dict:
    """
    Train XGBoost regressor for Stage 0 predictive modeling
    
    Args:
        data_dict: Results from prepare_modeling_data
        xgb_params: XGBoost parameters (optional)
        
    Returns:
        Dictionary with trained model and performance metrics
    """
    print("🚀 TRAINING XGBOOST MODEL FOR STAGE 0")
    print("=" * 38)
    
    # Default XGBoost parameters (optimized for interpretability)
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    print(f"📊 Training on {len(X_train):,} samples with {len(data_dict['feature_cols'])} features")
    
    # Train XGBoost model
    print("🎯 Training XGBoost regressor...")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Cross-validation for model assessment
    print("📈 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    cv_metrics = {
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std()
    }
    
    print(f"✅ MODEL TRAINING COMPLETED!")
    print(f"📈 Train Performance:")
    print(f"   MAE: {train_metrics['mae']:.4f} kWh")
    print(f"   RMSE: {train_metrics['rmse']:.4f} kWh")
    print(f"   R²: {train_metrics['r2']:.4f}")
    print(f"📈 Test Performance:")
    print(f"   MAE: {test_metrics['mae']:.4f} kWh")
    print(f"   RMSE: {test_metrics['rmse']:.4f} kWh")
    print(f"   R²: {test_metrics['r2']:.4f}")
    print(f"📈 Cross-Validation:")
    print(f"   CV MAE: {cv_metrics['cv_mae_mean']:.4f} ± {cv_metrics['cv_mae_std']:.4f} kWh")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_metrics': cv_metrics,
        'xgb_params': xgb_params,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def calculate_shap_values(model_dict: dict, 
                         data_dict: dict,
                         sample_size: int = 1000) -> dict:
    """
    Calculate SHAP values for consumption driver analysis
    
    Args:
        model_dict: Results from train_xgboost_model
        data_dict: Results from prepare_modeling_data
        sample_size: Sample size for SHAP analysis (performance optimization)
        
    Returns:
        Dictionary with SHAP values and explainer
    """
    print("🔍 CALCULATING SHAP VALUES FOR CONSUMPTION DRIVER ANALYSIS")
    print("=" * 58)
    
    model = model_dict['model']
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    
    # Sample data for SHAP analysis (performance optimization)
    if len(X_test) > sample_size:
        print(f"📊 Sampling {sample_size:,} test rows for SHAP analysis...")
        X_shap = X_test.sample(n=sample_size, random_state=42)
    else:
        print(f"📊 Using all {len(X_test):,} test rows for SHAP analysis...")
        X_shap = X_test.copy()
    
    print(f"📊 SHAP analysis data: {X_shap.shape[0]:,} samples, {X_shap.shape[1]} features")
    
    # Create SHAP explainer
    print("🚀 Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("🔍 Computing SHAP values...")
    shap_values = explainer.shap_values(X_shap)
    
    print(f"✅ SHAP values calculated!")
    print(f"📊 SHAP values shape: {shap_values.shape}")
    print(f"📊 Expected value (baseline): {explainer.expected_value:.4f} kWh")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_shap': X_shap,
        'expected_value': explainer.expected_value,
        'feature_names': list(X_shap.columns)
    }

def run_stage0_predictive_modeling(df: pd.DataFrame,
                                  target_col: str = "total_kwh",
                                  test_size: float = 0.2,
                                  shap_sample_size: int = 1000,
                                  xgb_params: dict = None) -> dict:
    """
    Run complete Stage 0 predictive modeling pipeline
    
    Args:
        df: Input dataframe with comprehensive features
        target_col: Target variable name
        test_size: Test set proportion
        shap_sample_size: Sample size for SHAP analysis
        xgb_params: XGBoost parameters (optional)
        
    Returns:
        Complete Stage 0 modeling results
    """
    print("🎯 RUNNING STAGE 0: PREDICTIVE MODELING FOR CONSUMPTION DRIVERS")
    print("=" * 65)
    
    # Step 1: Prepare clean modeling data
    print("\n" + "="*20 + " STEP 1: DATA PREPARATION " + "="*20)
    data_dict = prepare_modeling_data(df, target_col, test_size)
    
    # Step 2: Train XGBoost model
    print("\n" + "="*20 + " STEP 2: MODEL TRAINING " + "="*20)
    model_dict = train_xgboost_model(data_dict, xgb_params)
    
    # Step 3: Calculate SHAP values
    print("\n" + "="*20 + " STEP 3: SHAP ANALYSIS " + "="*20)
    shap_dict = calculate_shap_values(model_dict, data_dict, shap_sample_size)
    
    print("\n🎉 STAGE 0 PREDICTIVE MODELING COMPLETED!")
    print("="*50)
    print("✅ Data prepared and cleaned")
    print("✅ XGBoost model trained and evaluated")
    print("✅ SHAP values calculated for driver analysis")
    print("✅ Ready for consumption driver analysis")
    
    # Summary statistics
    r2_score_val = model_dict['test_metrics']['r2']
    mae_score = model_dict['test_metrics']['mae']
    
    print(f"\n📊 QUICK SUMMARY:")
    print(f"   Model Performance (R²): {r2_score_val:.4f}")
    print(f"   Prediction Error (MAE): {mae_score:.4f} kWh")
    print(f"   Features analyzed: {len(data_dict['feature_cols'])}")
    print(f"   Feature groups: {len(data_dict['feature_groups'])}")
    
    return {
        'data': data_dict,
        'model': model_dict,
        'shap': shap_dict,
        'summary': {
            'r2_score': r2_score_val,
            'mae_score': mae_score,
            'feature_count': len(data_dict['feature_cols']),
            'group_count': len(data_dict['feature_groups'])
        }
    }

if __name__ == "__main__":
    print("🎯 Predictive Model - Stage 0: XGBoost Training & SHAP Generation")
    print("Usage: from src.models.predictive_model import run_stage0_predictive_modeling") 