"""
MODEL BUILDING - Enhanced XGBoost with Optuna Optimization
==========================================================

Advanced model building pipeline for electricity consumption forecasting.
Features Optuna hyperparameter optimization and clean separation of concerns.

Key Features:
- Optuna-based hyperparameter tuning
- Support for both day-ahead (h=1) and week-ahead (h=7) forecasting
- Log transform option for relative error modeling
- Comprehensive data preparation and validation
- GPU/CPU optimization

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def prepare_xgboost_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "total_kwh",
    categorical_cols: list = None,
    log_transform: bool = False
) -> dict:
    """
    Prepare XGBoost-ready datasets: feature matrices and target arrays.
    - Optionally applies log1p transform on the target.
    - Label-encodes any categorical columns (if specified); fits encoders on combined train+val+test.
    - Drops rows where X or y is NaN (after encoding and optional transform).
    
    Args:
        train_df, val_df, test_df: DataFrames with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        categorical_cols: List of categorical columns to encode (auto-detected if None)
        log_transform: Whether to apply log1p transform to target
        
    Returns:
        Dictionary with X_train, y_train, etc. and metadata
    """
    print(f"📊 Preparing XGBoost data for target: {target_col}")
    
    # 1) Subset feature matrices
    X_train = train_df[feature_cols].copy()
    X_val   = val_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    # 2) Prepare target arrays
    y_train = train_df[target_col].values
    y_val   = val_df[target_col].values
    y_test  = test_df[target_col].values

    # 3) Log-transform target if requested
    if log_transform:
        print("   📈 Applying log1p transform to target variable...")
        eps = 1e-6
        y_train = np.log1p(np.clip(y_train, a_min=0, a_max=None) + eps)
        y_val   = np.log1p(np.clip(y_val,   a_min=0, a_max=None) + eps)
        y_test  = np.log1p(np.clip(y_test,  a_min=0, a_max=None) + eps)

    # 4) Identify categorical columns
    if categorical_cols is None:
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    label_encoders = {}
    if categorical_cols:
        print(f"   🔄 Encoding {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col]   = le.transform(X_val[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))
            label_encoders[col] = le

    # 5) Drop rows with NaN in X or y
    def drop_na(X: pd.DataFrame, y: np.ndarray):
        mask_X = ~X.isna().any(axis=1)
        mask_y = ~np.isnan(y)
        mask = mask_X & mask_y
        return X[mask].reset_index(drop=True), y[mask]

    orig_train_len, orig_val_len, orig_test_len = len(X_train), len(X_val), len(X_test)
    
    X_train, y_train = drop_na(X_train, y_train)
    X_val,   y_val   = drop_na(X_val,   y_val)
    X_test,  y_test  = drop_na(X_test,  y_test)

    # Report NaN drops
    train_dropped = orig_train_len - len(X_train)
    val_dropped = orig_val_len - len(X_val)
    test_dropped = orig_test_len - len(X_test)
    
    if train_dropped + val_dropped + test_dropped > 0:
        print(f"   🧹 Dropped rows due to NaNs:")
        print(f"      Train: {train_dropped} ({(train_dropped/orig_train_len)*100:.1f}%)")
        print(f"      Val: {val_dropped} ({(val_dropped/orig_val_len)*100:.1f}%)")
        print(f"      Test: {test_dropped} ({(test_dropped/orig_test_len)*100:.1f}%)")

    print(f"   ✅ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "label_encoders": label_encoders,
        "log_transform": log_transform
    }


def create_xgboost_model(
    params: dict = None,
    use_gpu: bool = False
) -> xgb.XGBRegressor:
    """
    Instantiate an XGBoost regressor with default or custom parameters.
    
    Args:
        params: Dictionary of custom parameters
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured XGBoost regressor
    """
    default_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "gpu_hist" if use_gpu else "hist"
    }
    
    if use_gpu:
        default_params["n_jobs"] = 1  # Avoid CPU+GPU conflicts
        
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    print(f"   ✅ XGBoost model created with {default_params['n_estimators']} estimators "
          f"and tree_method='{default_params['tree_method']}'")
    return model


def train_xgboost_model(
    data_dict: dict,
    params: dict = None,
    use_gpu: bool = False,
    early_stopping_rounds: int = 50,
    verbose: bool = False
) -> dict:
    """
    Train an XGBoost regressor using provided datasets and return metadata.
    
    Args:
        data_dict: Dictionary with prepared data
        params: Model parameters
        use_gpu: Whether to use GPU
        early_stopping_rounds: Early stopping patience
        verbose: Whether to print training logs
        
    Returns:
        Dictionary with trained model and metadata
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val   = data_dict["X_val"]
    y_val   = data_dict["y_val"]

    model = create_xgboost_model(params=params, use_gpu=use_gpu)

    print("   🚀 Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )

    best_iter = model.best_iteration
    fi_df = pd.DataFrame({
        "feature": data_dict["X_train"].columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"   ✅ Model trained with best_iteration = {best_iter + 1}")

    return {
        "model": model,
        "best_iteration": best_iter,
        "feature_importance": fi_df
    }


def predict_xgboost(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    log_transform: bool = False
) -> np.ndarray:
    """
    Generate predictions using a trained XGBoost model.
    If log_transform=True, apply inverse transform expm1 to the predictions.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        log_transform: Whether to inverse transform predictions
        
    Returns:
        Prediction array
    """
    preds = model.predict(X)
    if log_transform:
        preds = np.expm1(np.clip(preds, a_min=0, a_max=None))
    return preds


def objective_optuna(trial: optuna.Trial, data_dict: dict) -> float:
    """
    Optuna objective to minimize validation RMSE.
    
    Args:
        trial: Optuna trial object
        data_dict: Dictionary with training data
        
    Returns:
        Validation RMSE to minimize
    """
    param = {
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "n_estimators": 1000,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "gpu_hist" if data_dict.get("use_gpu", False) else "hist"
    }

    model = xgb.XGBRegressor(**param)
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val   = data_dict["X_val"]
    y_val   = data_dict["y_val"]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False
    )

    y_val_pred = model.predict(X_val)
    rmse_val = np.sqrt(((y_val - y_val_pred) ** 2).mean())
    return rmse_val


def optimize_xgboost_hyperparams(
    data_dict: dict,
    n_trials: int = 50,
    use_gpu: bool = False,
    seed: int = 42
) -> dict:
    """
    Run an Optuna study to find best hyperparameters minimizing validation RMSE.
    
    Args:
        data_dict: Dictionary with training data
        n_trials: Number of optimization trials
        use_gpu: Whether to use GPU
        seed: Random seed
        
    Returns:
        Dictionary with best parameters
    """
    print(f"🔍 Starting Optuna optimization with {n_trials} trials...")
    
    data_dict["use_gpu"] = use_gpu

    def objective(trial):
        return objective_optuna(trial, data_dict)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    
    # Suppress Optuna logs during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"   🎯 Best trial RMSE = {study.best_value:.4f}")
    print(f"   📋 Best params: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({
        "n_estimators": 1000,
        "random_state": 42,
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "n_jobs": 1 if use_gpu else -1
    })
    return best_params


def train_and_evaluate_dayahead(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "total_kwh",
    categorical_cols: list = None,
    use_gpu: bool = False,
    n_trials: int = 50,
    log_transform: bool = False
) -> dict:
    """
    Day‐ahead (h=1) pipeline with Optuna tuning.
    Returns dict with best_params, model, feature_importance, metrics, predictions, actuals.
    
    Args:
        train_df, val_df, test_df: DataFrames with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        categorical_cols: List of categorical columns
        use_gpu: Whether to use GPU
        n_trials: Number of Optuna trials
        log_transform: Whether to use log transform
        
    Returns:
        Dictionary with complete results
    """
    print("🚀 DAY-AHEAD FORECASTING PIPELINE")
    print("=" * 40)
    
    # Try importing evaluation functions
    try:
        from ..evaluation.forecast_evaluation import compute_regression_metrics, print_regression_results
    except ImportError:
        print("⚠️ Warning: Could not import evaluation functions. Will return raw predictions.")
        compute_regression_metrics = None
        print_regression_results = None

    data_dict = prepare_xgboost_data(
        train_df, val_df, test_df,
        feature_cols, target_col,
        categorical_cols=categorical_cols,
        log_transform=log_transform
    )

    print("\n🔍 Starting Optuna hyperparameter optimization...")
    best_params = optimize_xgboost_hyperparams(
        data_dict, n_trials=n_trials, use_gpu=use_gpu, seed=42
    )

    print("\n🚀 Training final XGBoost model with best hyperparameters...")
    model_dict = train_xgboost_model(
        data_dict,
        params=best_params,
        use_gpu=use_gpu,
        early_stopping_rounds=50,
        verbose=False
    )
    model = model_dict["model"]

    # Get true values (inverse transform if needed)
    y_train_true = data_dict["y_train"]
    y_val_true   = data_dict["y_val"]
    y_test_true  = data_dict["y_test"]
    
    if log_transform:
        y_train_true = np.expm1(np.clip(y_train_true, a_min=0, a_max=None))
        y_val_true   = np.expm1(np.clip(y_val_true,   a_min=0, a_max=None))
        y_test_true  = np.expm1(np.clip(y_test_true,  a_min=0, a_max=None))

    # Generate predictions
    y_train_pred = predict_xgboost(model, data_dict["X_train"], log_transform=log_transform)
    y_val_pred   = predict_xgboost(model, data_dict["X_val"],   log_transform=log_transform)
    y_test_pred  = predict_xgboost(model, data_dict["X_test"],  log_transform=log_transform)

    # Compute metrics if evaluation functions available
    metrics = {}
    if compute_regression_metrics is not None:
        metrics = {
            "train": compute_regression_metrics(y_train_true, y_train_pred),
            "val":   compute_regression_metrics(y_val_true,   y_val_pred),
            "test":  compute_regression_metrics(y_test_true,  y_test_pred)
        }

        print("\n📈 DAY‐AHEAD PERFORMANCE (Final Model):")
        print("   Train Metrics:")
        print_regression_results(metrics["train"], prefix="Train")
        print("   Validation Metrics:")
        print_regression_results(metrics["val"], prefix="Val")
        print("   Test Metrics:")
        print_regression_results(metrics["test"], prefix="Test")

    return {
        "best_params": best_params,
        "model": model,
        "feature_importance": model_dict["feature_importance"],
        "metrics": metrics,
        "predictions": {
            "train": y_train_pred,
            "val":   y_val_pred,
            "test":  y_test_pred
        },
        "actuals": {
            "train": y_train_true,
            "val":   y_val_true,
            "test":  y_test_true
        }
    }


def train_and_evaluate_weekahead(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "label_7",
    categorical_cols: list = None,
    use_gpu: bool = False,
    n_trials: int = 50,
    log_transform: bool = False
) -> dict:
    """
    Week‐ahead (h=7) pipeline with Optuna tuning.
    
    Args:
        train_df, val_df, test_df: DataFrames with features and target
        feature_cols: List of feature column names
        target_col: Target column name (should be 'label_7')
        categorical_cols: List of categorical columns
        use_gpu: Whether to use GPU
        n_trials: Number of Optuna trials
        log_transform: Whether to use log transform
        
    Returns:
        Dictionary with complete results
    """
    print("🚀 WEEK-AHEAD FORECASTING PIPELINE")
    print("=" * 40)
    
    # Try importing evaluation functions
    try:
        from ..evaluation.forecast_evaluation import compute_regression_metrics, print_regression_results
    except ImportError:
        print("⚠️ Warning: Could not import evaluation functions. Will return raw predictions.")
        compute_regression_metrics = None
        print_regression_results = None

    data_dict = prepare_xgboost_data(
        train_df, val_df, test_df,
        feature_cols, target_col,
        categorical_cols=categorical_cols,
        log_transform=log_transform
    )

    print("\n🔍 Starting Optuna hyperparameter optimization (Week‐Ahead)...")
    best_params = optimize_xgboost_hyperparams(
        data_dict, n_trials=n_trials, use_gpu=use_gpu, seed=42
    )

    print("\n🚀 Training final XGBoost model with best hyperparameters (Week‐Ahead)...")
    model_dict = train_xgboost_model(
        data_dict,
        params=best_params,
        use_gpu=use_gpu,
        early_stopping_rounds=50,
        verbose=False
    )
    model = model_dict["model"]

    # Get true values (inverse transform if needed)
    y_train_true = data_dict["y_train"]
    y_val_true   = data_dict["y_val"]
    y_test_true  = data_dict["y_test"]
    
    if log_transform:
        y_train_true = np.expm1(np.clip(y_train_true, a_min=0, a_max=None))
        y_val_true   = np.expm1(np.clip(y_val_true,   a_min=0, a_max=None))
        y_test_true  = np.expm1(np.clip(y_test_true,  a_min=0, a_max=None))

    # Generate predictions
    y_train_pred = predict_xgboost(model, data_dict["X_train"], log_transform=log_transform)
    y_val_pred   = predict_xgboost(model, data_dict["X_val"],   log_transform=log_transform)
    y_test_pred  = predict_xgboost(model, data_dict["X_test"],  log_transform=log_transform)

    # Compute metrics if evaluation functions available
    metrics = {}
    if compute_regression_metrics is not None:
        metrics = {
            "train": compute_regression_metrics(y_train_true, y_train_pred),
            "val":   compute_regression_metrics(y_val_true,   y_val_pred),
            "test":  compute_regression_metrics(y_test_true,  y_test_pred)
        }

        print("\n📈 WEEK‐AHEAD PERFORMANCE (Final Model):")
        print("   Train Metrics:")
        print_regression_results(metrics["train"], prefix="Train")
        print("   Validation Metrics:")
        print_regression_results(metrics["val"], prefix="Val")
        print("   Test Metrics:")
        print_regression_results(metrics["test"], prefix="Test")

    return {
        "best_params": best_params,
        "model": model,
        "feature_importance": model_dict["feature_importance"],
        "metrics": metrics,
        "predictions": {
            "train": y_train_pred,
            "val":   y_val_pred,
            "test":  y_test_pred
        },
        "actuals": {
            "train": y_train_true,
            "val":   y_val_true,
            "test":  y_test_true
        }
    }


if __name__ == "__main__":
    print("🚀 Enhanced Model Building with Optuna Optimization")
    print("=" * 50)
    print("✅ FEATURES:")
    print("   🔍 Optuna hyperparameter optimization")
    print("   📊 Support for day-ahead and week-ahead forecasting")
    print("   🔧 GPU/CPU optimization")
    print("   📈 Log transform option for relative errors")
    print("   🧹 Comprehensive data cleaning and validation")
    print("=" * 50)
    print("Usage:")
    print("  from src.models.model_building import train_and_evaluate_dayahead")
    print("  results = train_and_evaluate_dayahead(train, val, test, features)") 