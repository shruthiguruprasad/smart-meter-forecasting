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
from sklearn.preprocessing import LabelEncoder
import warnings
import optuna
from optuna.integration import XGBoostPruningCallback

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
    
    # Handle categorical variables - encode to numerical
    categorical_encodings = {}
    
    for col in feature_cols:
        if col in model_data.columns and model_data[col].dtype == 'object':
            print(f"🔄 Encoding categorical variable: {col}")
            # Use label encoding for categorical variables
            encoder = LabelEncoder()
            model_data[col] = encoder.fit_transform(model_data[col].astype(str))
            categorical_encodings[col] = encoder
    
    print(f"🔢 Encoded {len(categorical_encodings)} categorical variables")
    
    # Prepare features and target
    X = model_data[feature_cols]
    y = model_data[target_col]
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"⚠️  Converting remaining object column to numeric: {col}")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Remove any remaining NaN values after conversion
    initial_samples = len(X)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    final_samples = len(X)
    
    if final_samples < initial_samples:
        print(f"🧹 Removed {initial_samples - final_samples} samples with NaN after encoding")
    
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
        'target_col': target_col,
        'categorical_encodings': categorical_encodings
    }

def tune_hyperparameters(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        feature_cols: list,
                        target_col: str = "total_kwh",
                        n_trials: int = 50) -> dict:
    """
    Tune XGBoost hyperparameters using Optuna with early stopping and validation set
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        feature_cols: List of feature columns
        target_col: Target variable name
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of best hyperparameters
    """
    print("🎯 Tuning XGBoost Hyperparameters")
    print("=" * 40)
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'tree_method': 'hist',  # Use histogram-based algorithm for faster training
            'random_state': 42
        }
        
        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get best validation score
        return model.best_score
    
    # Create study
    study = optuna.create_study(
        direction='minimize',  # Minimize RMSE
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'random_state': 42
    })
    
    print("\n📊 Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Best Validation RMSE: {study.best_value:.4f}")
    
    return best_params

def train_xgboost_model(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols: list,
                       target_col: str = "total_kwh",
                       params: dict = None) -> tuple:
    """
    Train XGBoost model with best hyperparameters and evaluate on validation/test sets
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable name
        params: Optional hyperparameters (if None, will use default tuned parameters)
        
    Returns:
        Tuple of (trained model, validation metrics, test metrics)
    """
    print("🚀 Training XGBoost Model")
    print("=" * 35)
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Use default parameters if none provided
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'tree_method': 'hist',
            'random_state': 42
        }
    
    # Train model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Make predictions
    val_preds = model.predict(dval)
    test_preds = model.predict(dtest)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    val_metrics = calculate_metrics(y_val, val_preds)
    test_metrics = calculate_metrics(y_test, test_preds)
    
    print("\n📊 Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    print("\n📊 Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    return model, val_metrics, test_metrics

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
    model_dict = train_xgboost_model(data_dict['X_train'], data_dict['X_test'], data_dict['X_test'], data_dict['feature_cols'], target_col)
    
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
    r2_score_val = model_dict[1]['r2']
    mae_score = model_dict[1]['mae']
    
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