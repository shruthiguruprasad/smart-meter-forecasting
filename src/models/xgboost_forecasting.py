"""
🚀 XGBOOST FORECASTING - Model Building and Prediction
====================================================

XGBoost model creation, training, and prediction functions.
Pure model building without evaluation - evaluation functions moved to evaluation folder.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def prepare_xgboost_data(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        feature_cols: list,
                        target_col: str = "total_kwh",
                        household_id: str = None) -> dict:
    """
    Prepare data for XGBoost forecasting
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable for forecasting
        household_id: Optional specific household ID to filter for
        
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
    
    # Prepare targets
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
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
    
    # Remove any NaN values
    train_mask = ~(X_train.isna().any(axis=1) | pd.isna(y_train))
    val_mask = ~(X_val.isna().any(axis=1) | pd.isna(y_val))
    test_mask = ~(X_test.isna().any(axis=1) | pd.isna(y_test))
    
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"✅ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'train_dates': train_df['day'].values[train_mask],
        'val_dates': val_df['day'].values[val_mask],
        'test_dates': test_df['day'].values[test_mask]
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
        'n_jobs': -1,
        'tree_method': 'gpu_hist' if use_gpu else 'hist'
    }
    
    if params:
        default_params.update(params)
    
    model = xgb.XGBRegressor(**default_params)
    
    print(f"✅ XGBoost model created with {default_params['n_estimators']} estimators")
    return model

def train_xgboost_forecasting(data_dict: dict,
                             params: dict = None,
                             use_gpu: bool = False,
                             early_stopping_rounds: int = 50) -> dict:
    """
    Train XGBoost model for forecasting
    
    Args:
        data_dict: Dictionary with prepared data
        params: Custom model parameters
        use_gpu: Whether to use GPU acceleration
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Dictionary with trained model and metadata
    """
    print("🚀 Training XGBoost forecasting model...")
    
    # Create model
    model = create_xgboost_model(params, use_gpu)
    
    # Train with early stopping
    model.fit(
        data_dict['X_train'], 
        data_dict['y_train'],
        eval_set=[(data_dict['X_train'], data_dict['y_train']),
                  (data_dict['X_val'], data_dict['y_val'])],
        eval_metric='rmse',
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
    )
    
    print(f"✅ Model trained with {model.best_iteration + 1} iterations")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': data_dict['feature_cols'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'feature_importance': feature_importance,
        'best_iteration': model.best_iteration,
        'training_completed': True
    }

def predict_xgboost(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.array:
    """
    Generate predictions with XGBoost model
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        
    Returns:
        Predictions array
    """
    predictions = model.predict(X)
    return predictions

def xgboost_day_ahead_forecast(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               feature_cols: list,
                               target_col: str = "total_kwh",
                               household_id: str = None,
                               params: dict = None,
                               use_gpu: bool = False) -> dict:
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
        
    Returns:
        Dictionary with model, predictions, actuals, and metadata
    """
    print("🚀 XGBOOST DAY-AHEAD FORECASTING")
    print("=" * 40)
    
    # Prepare data
    data_dict = prepare_xgboost_data(
        train_df, val_df, test_df, feature_cols, target_col, household_id
    )
    
    # Train model
    model_dict = train_xgboost_forecasting(data_dict, params, use_gpu)
    
    # Generate predictions
    train_pred = predict_xgboost(model_dict['model'], data_dict['X_train'])
    val_pred = predict_xgboost(model_dict['model'], data_dict['X_val'])
    test_pred = predict_xgboost(model_dict['model'], data_dict['X_test'])
    
    # Prepare results
    results = {
        'model': model_dict['model'],
        'predictions': {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        },
        'actuals': {
            'train': data_dict['y_train'],
            'val': data_dict['y_val'],
            'test': data_dict['y_test']
        },
        'dates': {
            'train': data_dict['train_dates'],
            'val': data_dict['val_dates'],
            'test': data_dict['test_dates']
        },
        'feature_importance': model_dict['feature_importance'],
        'feature_cols': feature_cols,
        'target_col': target_col,
        'y_true': data_dict['y_test'],
        'y_pred': test_pred
    }
    
    print("✅ XGBoost forecasting completed")
    return results

def xgboost_multi_household_forecast(train_df: pd.DataFrame,
                                    val_df: pd.DataFrame,
                                    test_df: pd.DataFrame,
                                    feature_cols: list,
                                    target_col: str = "total_kwh",
                                    n_households: int = 5,
                                    params: dict = None) -> dict:
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
            household_id, params
        )
        
        results[household_id] = household_results
    
    results['households'] = households
    
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
    print("🚀 XGBoost Forecasting Module")
    print("Usage: from src.models.xgboost_forecasting import xgboost_day_ahead_forecast") 