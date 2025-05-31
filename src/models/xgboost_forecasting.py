"""
🚀 XGBOOST FORECASTING - Model Building and Prediction
====================================================

XGBoost model creation, training, and prediction functions.
Pure model building without evaluation - evaluation functions moved to evaluation folder.

🔧 ENHANCED FEATURES:
- ✅ Datetime column validation (prevents accidental leakage)
- ✅ NaN monitoring with percentage reporting
- ✅ GPU/CPU resource management optimization
- ✅ Log transform option for relative error modeling
- ✅ Comprehensive error handling and validation

⚠️ CRITICAL REQUIREMENT:
All lag/rolling features MUST be properly time-shifted upstream to prevent data leakage:
- For day-ahead: lag1_total uses t-1 to predict t
- For week-ahead: features use ≤t-1 to predict t+7

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def validate_forecasting_features(feature_cols: list, target_col: str = "total_kwh") -> None:
    """
    Validate feature list for common forecasting issues
    
    Args:
        feature_cols: List of feature column names
        target_col: Target variable name
        
    Raises:
        ValueError: If potential data leakage or issues detected
    """
    # Check for direct target leakage
    if target_col in feature_cols:
        raise ValueError(f"❌ Target variable '{target_col}' found in features!")
    
    # Check for obvious leakage patterns
    leakage_patterns = [
        'total_kwh', 'mean_kwh', 'peak_kwh', 'min_kwh',  # Direct consumption
        'morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh',  # Time-of-day consumption  
        'consumption_sharpness', 'usage_concentration'  # Consumption-derived features
    ]
    
    found_leakage = [f for f in feature_cols if any(pattern in f for pattern in leakage_patterns) 
                     and not any(lag in f for lag in ['lag', 'roll', 'delta'])]
    
    if found_leakage:
        print(f"⚠️ WARNING: Potential data leakage features found: {found_leakage[:5]}...")
        print("   Make sure these are properly time-shifted or remove them.")
    
    print(f"✅ Feature validation completed: {len(feature_cols)} features checked")

def prepare_xgboost_data(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        feature_cols: list,
                        target_col: str = "total_kwh",
                        household_id: str = None,
                        log_transform: bool = False) -> dict:
    """
    Prepare data for XGBoost forecasting
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable for forecasting
        household_id: Optional specific household ID to filter for
        log_transform: Whether to apply log transform to target variable
        
    Returns:
        Dictionary with prepared data
    """
    print("📊 Preparing XGBoost forecasting data...")
    
    # Filter for specific household if provided
    if household_id:
        train_df = train_df[train_df['LCLid'] == household_id].copy()
        val_df = val_df[val_df['LCLid'] == household_id].copy()
        test_df = test_df[test_df['LCLid'] == household_id].copy()
    
    # Prepare feature matrices
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # 🔍 SAFETY CHECK: Verify no datetime columns slip into features
    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        raise ValueError(f"❌ Datetime columns found in features: {list(datetime_cols)}. Remove these before training.")
    
    # Prepare targets
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # 📊 OPTIONAL: Apply log transform for relative errors
    if log_transform:
        print("📈 Applying log transform to target variable...")
        y_train = np.log1p(y_train)  # log(1 + x) handles zeros safely
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)
    
    # Handle categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    if len(categorical_cols) > 0:
        print(f"🔄 Encoding {len(categorical_cols)} categorical variables...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on train + val + test to handle unseen categories
            all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
            le.fit(all_values)
            
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # Store original lengths for NaN monitoring
    orig_train_len = len(X_train)
    orig_val_len = len(X_val)
    orig_test_len = len(X_test)
    
    # Remove any NaN values
    train_mask = ~(X_train.isna().any(axis=1) | pd.isna(y_train))
    val_mask = ~(X_val.isna().any(axis=1) | pd.isna(y_val))
    test_mask = ~(X_test.isna().any(axis=1) | pd.isna(y_test))
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    # 📊 NaN MONITORING: Report rows dropped
    train_dropped = orig_train_len - len(X_train)
    val_dropped = orig_val_len - len(X_val)
    test_dropped = orig_test_len - len(X_test)
    
    print(f"📊 Rows dropped due to NaNs:")
    print(f"   Train: {train_dropped} ({(train_dropped/orig_train_len)*100:.1f}%)")
    print(f"   Val: {val_dropped} ({(val_dropped/orig_val_len)*100:.1f}%)")
    print(f"   Test: {test_dropped} ({(test_dropped/orig_test_len)*100:.1f}%)")
    
    print(f"✅ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'train_dates': train_df['day'].values[train_mask],
        'val_dates': val_df['day'].values[val_mask],
        'test_dates': test_df['day'].values[test_mask],
        'log_transform': log_transform
    }

def create_xgboost_model(params: dict = None, use_gpu: bool = False) -> xgb.XGBRegressor:
    """
    Create XGBoost model with optimized parameters
    
    Args:
        params: Custom parameters dictionary
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured XGBoost model
    """
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'gpu_hist' if use_gpu else 'hist'
    }
    
    # 🔧 GPU RESOURCE MANAGEMENT: Avoid CPU+GPU conflicts
    if use_gpu:
        default_params['n_jobs'] = 1  # Use single job with GPU to avoid conflicts
        print("🚀 Using GPU acceleration with single job")
    else:
        default_params['n_jobs'] = -1  # Use all CPU cores
        print("💻 Using CPU with all available cores")
    
    if params:
        default_params.update(params)
    
    model = xgb.XGBRegressor(**default_params)
    
    print(f"✅ XGBoost model created with {default_params['n_estimators']} estimators")
    return model

def train_xgboost_forecasting(data_dict: dict,
                             params: dict = None,
                             use_gpu: bool = False,
                             early_stopping_rounds: int = 50,
                             verbose_eval: bool = False) -> dict:
    """
    Train XGBoost model for forecasting
    
    Args:
        data_dict: Dictionary with prepared data
        params: Custom model parameters
        use_gpu: Whether to use GPU acceleration
        early_stopping_rounds: Early stopping patience
        verbose_eval: Whether to print evaluation metrics during training
        
    Returns:
        Dictionary with trained model and metadata
    """
    print("🚀 Training XGBoost forecasting model...")
    
    # Create model
    model = create_xgboost_model(params, use_gpu)
    
    # 📊 EARLY STOPPING STRATEGY: Monitor validation performance
    eval_set = [
        (data_dict['X_train'], data_dict['y_train']),
        (data_dict['X_val'], data_dict['y_val'])
    ]
    eval_names = ['train', 'validation']
    
    # Train with early stopping
    model.fit(
        data_dict['X_train'], 
        data_dict['y_train'],
        eval_set=eval_set,
        eval_metric='rmse',
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose_eval
    )
    
    print(f"✅ Model trained with {model.best_iteration + 1} iterations (stopped at {model.n_estimators})")
    
    # 📈 TRAINING SUMMARY: Report final performance
    train_rmse = model.evals_result_['validation_0']['rmse'][-1]
    val_rmse = model.evals_result_['validation_1']['rmse'][-1]
    print(f"📊 Final RMSE - Train: {train_rmse:.4f}, Validation: {val_rmse:.4f}")
    
    # Check for potential overfitting
    if val_rmse > train_rmse * 1.5:
        print("⚠️ WARNING: Significant overfitting detected (val_RMSE >> train_RMSE)")
        print("   Consider: reducing max_depth, increasing regularization, or more data")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data_dict['feature_cols'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'feature_importance': feature_importance,
        'best_iteration': model.best_iteration,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'training_completed': True
    }

def predict_xgboost(model: xgb.XGBRegressor, X: pd.DataFrame, log_transform: bool = False) -> np.array:
    """
    Generate predictions with XGBoost model
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        log_transform: Whether to inverse transform log predictions
        
    Returns:
        Predictions array
    """
    predictions = model.predict(X)
    
    # 📈 INVERSE LOG TRANSFORM if applied during training
    if log_transform:
        predictions = np.expm1(predictions)  # exp(x) - 1, inverse of log1p
        
    return predictions

def xgboost_day_ahead_forecast(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               feature_cols: list,
                               target_col: str = "total_kwh",
                               household_id: str = None,
                               params: dict = None,
                               use_gpu: bool = False,
                               log_transform: bool = False) -> dict:
    """
    Complete day-ahead forecasting pipeline with XGBoost
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable for forecasting
        household_id: Optional specific household ID
        params: Custom model parameters
        use_gpu: Whether to use GPU acceleration
        log_transform: Whether to apply log transform to target
        
    Returns:
        Dictionary with model, predictions, actuals, and metadata
    """
    print("🚀 XGBOOST DAY-AHEAD FORECASTING")
    print("=" * 40)
    
    # 🔍 VALIDATE FEATURES: Check for potential data leakage
    validate_forecasting_features(feature_cols, target_col)
    
    # Prepare data
    data_dict = prepare_xgboost_data(
        train_df, val_df, test_df, feature_cols, target_col, household_id, log_transform
    )
    
    # Train model
    model_dict = train_xgboost_forecasting(data_dict, params, use_gpu)
    
    # Generate predictions
    train_pred = predict_xgboost(model_dict['model'], data_dict['X_train'], log_transform)
    val_pred = predict_xgboost(model_dict['model'], data_dict['X_val'], log_transform)
    test_pred = predict_xgboost(model_dict['model'], data_dict['X_test'], log_transform)
    
    # If log transform was used, also inverse transform actuals for comparison
    y_train_orig = data_dict['y_train']
    y_val_orig = data_dict['y_val'] 
    y_test_orig = data_dict['y_test']
    
    if log_transform:
        y_train_orig = np.expm1(y_train_orig)
        y_val_orig = np.expm1(y_val_orig)
        y_test_orig = np.expm1(y_test_orig)
    
    # Prepare results
    results = {
        'model': model_dict['model'],
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'actuals': {
            'train': y_train_orig,
            'val': y_val_orig,
            'test': y_test_orig
        },
        'dates': {
            'train': data_dict['train_dates'],
            'val': data_dict['val_dates'],
            'test': data_dict['test_dates']
        },
        'feature_importance': model_dict['feature_importance'],
        'feature_cols': feature_cols,
        'target_col': target_col,
        'y_true': y_test_orig,
        'y_pred': test_pred,
        'log_transform': log_transform
    }
    
    print("✅ XGBoost forecasting completed")
    return results

def xgboost_multi_household_forecast(train_df: pd.DataFrame,
                                    val_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    feature_cols: list,
                                    target_col: str = "total_kwh",
                                    n_households: int = 5,
                                    params: dict = None,
                                    log_transform: bool = False) -> dict:
    """
    Run forecasting for multiple households
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable
        n_households: Number of households to forecast
        params: Model parameters
        log_transform: Whether to apply log transform to target
        
    Returns:
        Dictionary with results for all households
    """
    print(f"🚀 Multi-Household XGBoost Forecasting ({n_households} households)")
    print("=" * 60)
    
    # Get available households
    households = train_df['LCLid'].unique()[:n_households]
    
    results = {}
    
    for i, household_id in enumerate(households):
        print(f"\n📊 Forecasting household {i+1}/{n_households}: {household_id}")
        
        # Run forecasting for this household
        household_results = xgboost_day_ahead_forecast(
            train_df, val_df, test_df, feature_cols, target_col, 
            household_id, params, use_gpu=False, log_transform=log_transform
        )
        
        results[household_id] = household_results
    
    results['households'] = households
    results['log_transform'] = log_transform
    
    return results

def get_top_features(feature_importance: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Get top K most important features
    
    Args:
        feature_importance: Feature importance dataframe
        top_k: Number of top features to return
        
    Returns:
        Top features dataframe
    """
    top_features = feature_importance.head(top_k)
    
    print(f"\n📊 TOP {top_k} MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<30} : {row['importance']:.4f}")
    
    return top_features

if __name__ == "__main__":
    print("🚀 XGBoost Forecasting Module - Enhanced Edition")
    print("=" * 50)
    print("✅ NEW FEATURES:")
    print("   🔍 Automatic data leakage validation")
    print("   📊 NaN monitoring with percentage reporting") 
    print("   🔧 GPU/CPU resource optimization")
    print("   📈 Log transform for relative error modeling")
    print("   ⚠️ Overfitting detection and warnings")
    print("   🎯 Enhanced early stopping monitoring")
    print("=" * 50)
    print("Usage: from src.models.xgboost_forecasting import xgboost_day_ahead_forecast")
    print("Example: results = xgboost_day_ahead_forecast(train, val, test, features, log_transform=True)") 