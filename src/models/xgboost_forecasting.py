"""
xgboost_forecasting.py

Enhanced XGBoost training pipeline with Optuna optimization, but without any
printing or plotting of metrics. All evaluation logic has been moved
to forecasting_evaluation.py.

Usage:
    from xgboost_forecasting import run_xgb_with_optuna
    results = run_xgb_with_optuna(train_df, val_df, test_df, feature_cols, target_col, …)

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna import logging as optuna_logging
from sklearn.metrics import mean_squared_error
from pandas.api.types import is_numeric_dtype

# Suppress Optuna info logs (keep only progress bar)
optuna_logging.set_verbosity(optuna_logging.WARNING)


def run_xgb_with_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    use_gpu: bool = False,
    log_transform: bool = False,
    n_trials: int = 5,
    seed: int = 42
) -> dict:
    """
    Train an XGBoost regressor with Optuna hyperparameter tuning.

    Returns a dictionary with:
      - "best_params": dict of best hyperparameters
      - "best_iteration": int
      - "model": trained xgboost.Booster
      - "predictions": {"train": array, "val": array, "test": array}
      - "actuals":     {"train": array, "val": array, "test": array}
    """
    np.random.seed(seed)

    # 1) Build X/y arrays for train, val, test
    df_train = train_df[feature_cols + [target_col]].copy()
    df_val   = val_df[feature_cols + [target_col]].copy()
    df_test  = test_df[feature_cols + [target_col]].copy()

    # 2) Drop rows with NaNs in features or target
    df_train.dropna(subset=feature_cols + [target_col], inplace=True)
    df_val.dropna(subset=feature_cols + [target_col], inplace=True)
    df_test.dropna(subset=feature_cols + [target_col], inplace=True)

    # 3) Extract X/y arrays
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy().values

    X_val = df_val[feature_cols].copy()
    y_val = df_val[target_col].copy().values

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy().values

    # 4) Apply log1p transform if requested
    if log_transform:
        y_train = np.log1p(y_train)
        y_val   = np.log1p(y_val)
        y_test  = np.log1p(y_test)

    # 5) Encode pandas Categorical columns to integer codes
    for df_ in (X_train, X_val, X_test):
        for col in df_.select_dtypes(include="category").columns:
            df_[col] = df_[col].cat.codes

    # 6) Create DMatrix for train & val
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    # 7) Optuna objective
    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "random_state": seed,
            "eta": trial.suggest_loguniform("eta", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_uniform("gamma", 0.0, 5.0),
        }
        bst = xgb.train(
            param,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "validation")],
            early_stopping_rounds=25,
            verbose_eval=False
        )
        best_iter = bst.best_iteration
        trial.set_user_attr("best_iteration", best_iter)

        # Predict on validation up to best_iter
        preds_val = bst.predict(dval, iteration_range=(0, best_iter))
        rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))
        return rmse_val

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_iter = study.best_trial.user_attrs["best_iteration"]

    # 8) Retrain on train+val using best_params and best_iter
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    dall  = xgb.DMatrix(X_all, label=y_all)

    final_params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "random_state": seed,
        **best_params
    }
    final_model = xgb.train(
        final_params,
        dall,
        num_boost_round=best_iter
    )

    # 9) Build DMatrix for train, val, test (to get raw predictions)
    dtrain_all = xgb.DMatrix(X_train, label=y_train)
    dval_all   = xgb.DMatrix(X_val,   label=y_val)
    dtest_all  = xgb.DMatrix(X_test,  label=y_test)

    preds_train = final_model.predict(dtrain_all, iteration_range=(0, best_iter))
    preds_val   = final_model.predict(dval_all,   iteration_range=(0, best_iter))
    preds_test  = final_model.predict(dtest_all,  iteration_range=(0, best_iter))

    # 10) Invert log1p if needed
    if log_transform:
        preds_train = np.expm1(preds_train)
        preds_val   = np.expm1(preds_val)
        preds_test  = np.expm1(preds_test)
        y_train_true = np.expm1(y_train)
        y_val_true   = np.expm1(y_val)
        y_test_true  = np.expm1(y_test)
    else:
        y_train_true = y_train.copy()
        y_val_true   = y_val.copy()
        y_test_true  = y_test.copy()

    return {
        "best_params": best_params,
        "best_iteration": best_iter,
        "model": final_model,
        "predictions": {
            "train": preds_train,
            "val":   preds_val,
            "test":  preds_test
        },
        "actuals": {
            "train": y_train_true,
            "val":   y_val_true,
            "test":  y_test_true
        }
    }


if __name__ == "__main__":
    print("✔️ xgboost_forecasting.py loaded.")
    print("   Use run_xgb_with_optuna() to train and return results.")
