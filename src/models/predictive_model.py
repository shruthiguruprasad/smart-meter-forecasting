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
import time

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
                        n_trials: int = 50,
                        use_gpu: bool = True) -> dict:
    """
    Tune XGBoost hyperparameters using Optuna with early stopping and validation set
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        feature_cols: List of feature columns
        target_col: Target variable name
        n_trials: Number of optimization trials
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary of best hyperparameters
    """
    print("🎯 Tuning XGBoost Hyperparameters")
    print("=" * 40)
    
    try:
        # Prepare data
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col]
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col]
        
        # Handle categorical variables
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔄 Encoding {len(categorical_cols)} categorical variables...")
            from sklearn.preprocessing import LabelEncoder
            
            # Create and fit label encoders
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit on train data only
                le.fit(X_train[col].astype(str))
                # Transform both train and validation
                X_train[col] = le.transform(X_train[col].astype(str))
                X_val[col] = le.transform(X_val[col].astype(str))
                label_encoders[col] = le
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                # Tree structure
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                
                # Learning process
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                
                # Sampling and regularization
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                
                # Performance settings
                'tree_method': 'gpu_hist' if use_gpu else 'hist',
                'random_state': 42,
                'nthread': 4,
                'max_bin': 256
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
        
        # Create study with memory optimization
        study = optuna.create_study(
            direction='minimize',  # Minimize RMSE
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization with memory management
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True  # Garbage collection after each trial
        )
        
        # Get best parameters
        best_params = study.best_params
        best_params.update({
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'random_state': 42,
            'nthread': 4,
            'max_bin': 256
        })
        
        print("\n📊 Best Hyperparameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"\n🎯 Best Validation RMSE: {study.best_value:.4f}")
        
        return best_params
        
    except Exception as e:
        print(f"❌ Error during hyperparameter tuning: {str(e)}")
        # Return default parameters if tuning fails
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'random_state': 42,
            'nthread': 4,
            'max_bin': 256
        }

def train_xgboost_model(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    feature_cols: list,
                    target_col: str = "total_kwh",
                    params: dict = None,
                    use_gpu: bool = True) -> tuple:
    """
    Train XGBoost model with best hyperparameters and evaluate on validation/test sets
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        feature_cols: List of feature columns
        target_col: Target variable name
        params: Optional hyperparameters (if None, will use default tuned parameters)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (trained model, validation metrics, test metrics)
    """
    print("🚀 Training XGBoost Model")
    print("=" * 35)
    
    try:
        # Prepare data
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col]
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col]
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col]
        
        # Handle categorical variables
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔄 Encoding {len(categorical_cols)} categorical variables...")
            from sklearn.preprocessing import LabelEncoder
            
            # Create and fit label encoders
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit on train data only
                le.fit(X_train[col].astype(str))
                # Transform train, validation, and test
                X_train[col] = le.transform(X_train[col].astype(str))
                X_val[col] = le.transform(X_val[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                label_encoders[col] = le
        
        # Create DMatrix for XGBoost with memory optimization
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
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
                'tree_method': 'gpu_hist' if use_gpu else 'hist',
                'random_state': 42,
                'nthread': 4,
                'max_bin': 256
            }
        
        # Train model with early stopping and memory optimization
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
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling code

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
    
    # Handle case where only test data is provided
    if 'X_test' in data_dict:
        print("📊 Using test data for SHAP analysis...")
        X_shap = data_dict['X_test'].copy()
    else:
        raise ValueError("X_test must be provided in data_dict")
    
    # Handle categorical variables
    categorical_cols = X_shap.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"🔄 Encoding {len(categorical_cols)} categorical variables...")
        from sklearn.preprocessing import LabelEncoder
        
        # Create and fit label encoders
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit and transform
            X_shap[col] = le.fit_transform(X_shap[col].astype(str))
            label_encoders[col] = le
    
    # Sample data for SHAP analysis (performance optimization)
    if len(X_shap) > sample_size:
        print(f"📊 Sampling {sample_size:,} test rows for SHAP analysis...")
        X_shap = X_shap.sample(n=sample_size, random_state=42)
    else:
        print(f"📊 Using all {len(X_shap):,} test rows for SHAP analysis...")
    
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
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X_shap.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Most Important Features:")
    for _, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_shap': X_shap,
        'expected_value': explainer.expected_value,
        'feature_names': list(X_shap.columns),
        'feature_importance': feature_importance,
        'label_encoders': label_encoders if len(categorical_cols) > 0 else None
    }

def prepare_driver_analysis_data(df: pd.DataFrame, 
                               target_col: str = "total_kwh",
                               test_days: int = 90,
                               val_days: int = 30,
                               exclude_features: list = None) -> tuple:
    """
    Prepare data for consumption driver analysis (includes half-hourly data)
    
    Args:
        df: Input dataframe
        target_col: Target variable name
        test_days: Number of days for test set
        val_days: Number of days for validation set
        exclude_features: List of additional features to exclude (optional)
        
    Returns:
        Tuple of (train_df, val_df, test_df, feature_cols, target_col, feature_groups)
    """
    print("📊 PREPARING DATA FOR CONSUMPTION DRIVER ANALYSIS")
    print("=" * 50)
    
    # Get feature columns (exclude IDs, dates, and ALL target-related columns)
    exclude_patterns = [
        "LCLid", "day", "holiday_type",  # Basic exclusions
        target_col,  # Target variable itself
        # ALL variations of target column to prevent data leakage
        "total_kwh", "mean_kwh", "std_kwh", "peak_kwh", "min_kwh",  # Direct consumption stats
        "morning_kwh", "afternoon_kwh", "evening_kwh", "night_kwh",  # Time-of-day consumption
        "peak_period_kwh", "off_peak_kwh",  # Peak period consumption
        "base_load", "load_factor",  # Load metrics derived from consumption
        # Time series features derived from target
        f"lag1_{target_col}", f"lag7_{target_col}", f"lag14_{target_col}",  # Target lags
        "lag1_total", "lag7_total", "lag14_total",  # Total lags
        f"roll7_{target_col}_mean", f"roll7_{target_col}_std",  # Target rolling stats
        "roll7_total_mean", "roll7_total_std", "roll14_total_mean", "roll14_total_std",  # Total rolling stats
        f"delta1_{target_col}", f"delta7_{target_col}",  # Target deltas
        "delta1_total", "delta7_total", "delta14_total",  # Total deltas
        f"pct_change_{target_col}", "pct_change1_total", "pct_change7_total",  # Target percentage changes
        # Consumption patterns derived from target
        "daily_variability", "coefficient_of_variation", "sharpness", "concentration"
    ]
    
    # Add user-specified features to exclude
    if exclude_features:
        exclude_patterns.extend(exclude_features)
        print(f"📋 Excluding {len(exclude_features)} additional features specified by user")
    
    # Get all feature columns
    feature_cols = []
    excluded_count = 0
    
    for col in df.columns:
        # Skip excluded patterns
        exclude_this = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude_this = True
                excluded_count += 1
                break
        
        if exclude_this:
            continue
            
        # Skip target column and its derivatives (additional check)
        if col == target_col or col.startswith(f"{target_col}_") or "kwh" in col.lower():
            excluded_count += 1
            continue
            
        feature_cols.append(col)
    
    print(f"📋 Selected {len(feature_cols)} features for modeling")
    print(f"🚫 Excluded {excluded_count} target-related features to prevent data leakage")
    print(f"🎯 Target variable: {target_col}")
    
    # Sort by date for chronological split
    df = df.sort_values('day')
    
    # Calculate split points
    total_days = len(df['day'].unique())
    test_start = total_days - test_days
    val_start = test_start - val_days
    
    # Get split dates
    dates = sorted(df['day'].unique())
    test_start_date = dates[test_start]
    val_start_date = dates[val_start]
    
    # Create splits
    train_df = df[df['day'] < val_start_date].copy()
    val_df = df[(df['day'] >= val_start_date) & (df['day'] < test_start_date)].copy()
    test_df = df[df['day'] >= test_start_date].copy()
    
    print(f"\n📅 Chronological Split:")
    print(f"   Training: {train_df['day'].min()} to {train_df['day'].max()} ({len(train_df):,} samples)")
    print(f"   Validation: {val_df['day'].min()} to {val_df['day'].max()} ({len(val_df):,} samples)")
    print(f"   Test: {test_df['day'].min()} to {test_df['day'].max()} ({len(test_df):,} samples)")
    
    # Define feature groups for consumption driver analysis (no consumption features)
    half_hourly = [c for c in feature_cols if c.startswith('hh_')]
    temporal = [c for c in feature_cols if any(x in c for x in ['dayofweek', 'month', 'season', 'quarter', 'holiday', 'weekend'])]
    weather = [c for c in feature_cols if any(x in c for x in ['temp', 'heating', 'cooling', 'humidity', 'wind', 'cloud'])]
    socio_economic = [c for c in feature_cols if any(x in c for x in ['acorn', 'Acorn'])]
    external = [c for c in feature_cols if any(x in c for x in ['tariff', 'ToU', 'stdor'])]
    
    # Get features that don't belong to any other group
    assigned_features = set(half_hourly + temporal + weather + socio_economic + external)
    other = [c for c in feature_cols if c not in assigned_features]
    
    feature_groups = {
        'half_hourly': half_hourly,
        'temporal': temporal,
        'weather': weather,
        'socio_economic': socio_economic,
        'external': external,
        'other': other
    }
    
    # Show feature group summary
    print(f"\n📋 FEATURE GROUP SUMMARY (LEAKAGE-FREE):")
    for group, features in feature_groups.items():
        if features:  # Only show groups with features
            print(f"   {group:<20}: {len(features):2d} features")
            if group == 'half_hourly' and len(features) > 0:
                print(f"      Sample: {features[:3]}")
    
    return train_df, val_df, test_df, feature_cols, target_col, feature_groups

if __name__ == "__main__":
    print("🎯 Predictive Model - Stage 0: XGBoost Training & SHAP Generation")
    print("Usage: from src.models.predictive_model import tune_hyperparameters, train_xgboost_model, prepare_driver_analysis_data") 