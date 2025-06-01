"""
forecast_evaluation.py

Evaluation utilities for day-ahead and week-ahead electricity forecasts.
This module reuses existing metric computations from the XGBoost forecasting pipeline
and provides clean functions strictly for evaluating trained models, including:
- compute_forecast_metrics: calculate MAE, RMSE, MAPE, R², and bias
- print_split_summary: print a formatted summary for a given split (Train, Val, Test)
- evaluate_model: take the results dict returned by run_xgb_with_optuna and print all splits
- evaluate_peak_performance: compute metrics specifically on the highest‐consumption days/weeks
- evaluate_forecast_residuals: analyze residual distribution (mean, std, percentiles) for any split

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics for forecasting:
      - MAE
      - RMSE
      - MAPE (zero-safe: excludes true==0)
      - R²
      - Bias (mean error)

    Parameters
    ----------
    y_true : np.ndarray
        Array of true target values.
    y_pred : np.ndarray
        Array of predicted target values.

    Returns
    -------
    dict with keys:
      - "mae": float
      - "rmse": float
      - "mape": float
      - "r2": float
      - "bias": float
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "bias": bias}


def print_split_summary(split_name: str, y_true: np.ndarray, y_pred: np.ndarray, units: str = "kWh"):
    """
    Print a formatted summary of forecast metrics for one data split.

    Example:
        📊 Train Performance:
           MAE:  2.3014 kWh
           RMSE: 4.1256 kWh
           MAPE: 19.94%
           R²:   0.8154
           Bias: -0.8181 (positive=overestimate, negative=underestimate)

    Parameters
    ----------
    split_name : str
        Name of the split (e.g., "Train", "Val", "Test").
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    units : str, default "kWh"
        Units to display for MAE/RMSE/Bias.
    """
    metrics = compute_forecast_metrics(y_true, y_pred)
    mae, rmse, mape, r2, bias = (
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
        metrics["r2"],
        metrics["bias"],
    )

    print(f"   📊 {split_name} Performance:")
    print(f"      MAE:  {mae:.4f} {units}")
    print(f"      RMSE: {rmse:.4f} {units}")
    print(f"      MAPE: {mape:.2f}%")
    print(f"      R²:   {r2:.4f}")
    print(f"      Bias: {bias:.4f}  (positive=overestimate, negative=underestimate)\n")


def evaluate_model(results: dict, units: str = "kWh"):
    """
    Given the dictionary returned by run_xgb_with_optuna, print evaluation
    summaries for Train, Val, and Test splits.

    Parameters
    ----------
    results : dict
        Output of run_xgb_with_optuna(). Expected keys:
          - "actuals": {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}
          - "predictions": {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}
    units : str, default "kWh"
        Units for MAE/RMSE/Bias.

    Usage
    -----
    results = run_xgb_with_optuna(...)
    evaluate_model(results)
    """
    print("\n📊 OVERALL FORECAST EVALUATION")
    print("----------------------------------------")
    # Train
    y_train = results["actuals"]["train"]
    p_train = results["predictions"]["train"]
    print_split_summary("Train", y_train, p_train, units=units)

    # Validation
    y_val = results["actuals"]["val"]
    p_val = results["predictions"]["val"]
    print_split_summary("Val", y_val, p_val, units=units)

    # Test
    y_test = results["actuals"]["test"]
    p_test = results["predictions"]["test"]
    print_split_summary("Test", y_test, p_test, units=units)


def evaluate_peak_performance(
    df_test: pd.DataFrame,
    target_col: str,
    y_pred: np.ndarray,
    percentile: float = 90.0,
    units: str = "kWh"
) -> dict:
    """
    Compute forecast metrics on the highest-consumption days/weeks ("peak" periods).

    - Select rows from df_test where target_col >= the given percentile threshold.
    - Compute MAE, RMSE, MAPE, R², and bias on that subset.

    Parameters
    ----------
    df_test : pd.DataFrame
        The clean test DataFrame used for prediction. Must contain `target_col`.
    target_col : str
        Column name for actual consumption (e.g., "label_1").
    y_pred : np.ndarray
        Predictions corresponding row-for-row with df_test.
    percentile : float, default 90.0
        Percentile threshold (0–100). Only rows with actual >= this percentile are evaluated.
    units : str, default "kWh"
        Units for MAE/RMSE/Bias.

    Returns
    -------
    dict containing:
      - "threshold": actual value at the percentile cut
      - "n_samples": number of rows in peak subset
      - "metrics": dict of MAE, RMSE, MAPE, R², bias on peak subset
    """
    # Extract true values
    y_true = df_test[target_col].values
    # Determine threshold
    threshold = np.percentile(y_true, percentile)
    # Boolean mask for peak rows
    mask = y_true >= threshold

    if mask.sum() == 0:
        print(f"⚠️  No data points exceed the {percentile}th percentile threshold ({threshold:.4f}).")
        return {"threshold": threshold, "n_samples": 0, "metrics": None}

    y_true_peak = y_true[mask]
    y_pred_peak = y_pred[mask]

    peak_metrics = compute_forecast_metrics(y_true_peak, y_pred_peak)

    print(f"\n📊 Peak Performance (≥ {percentile}th percentile → ≥ {threshold:.4f} {units})")
    print(f"   Number of peak samples: {mask.sum()}")
    print(f"   MAE_peak:  {peak_metrics['mae']:.4f} {units}")
    print(f"   RMSE_peak: {peak_metrics['rmse']:.4f} {units}")
    print(f"   MAPE_peak: {peak_metrics['mape']:.2f}%")
    print(f"   R²_peak:   {peak_metrics['r2']:.4f}")
    print(f"   Bias_peak: {peak_metrics['bias']:.4f} {units}\n")

    return {"threshold": threshold, "n_samples": int(mask.sum()), "metrics": peak_metrics}


def evaluate_forecast_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_dataframe: bool = False
) -> pd.DataFrame:
    """
    Analyze forecast residuals (y_pred - y_true) and return summary statistics.

    - Returns a DataFrame with:
      - "residual": error values (y_pred - y_true)
      - "abs_residual": absolute error
      - optionally: percentile ranks or bins

    - Prints key residual statistics (mean, median, std, 5th/95th percentiles).

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    return_dataframe : bool, default False
        If True, return the full DataFrame of residuals. Otherwise, only print stats.

    Returns
    -------
    pd.DataFrame (if return_dataframe=True) with columns:
      - "residual"
      - "abs_residual"
      - "pct_error" (absolute percentage error for nonzero y_true)
      - "pct_true_rank" (percentile rank of true value within test set)
    """
    # Ensure arrays
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    # Compute absolute percentage error for nonzero true
    pct_error = np.zeros_like(y_true, dtype=float)
    mask = y_true != 0
    pct_error[:] = np.nan
    pct_error[mask] = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

    # Compute percentile rank of each true value (to see where residuals occur)
    ranks = pd.Series(y_true).rank(pct=True).values * 100

    # Summary statistics
    print("\n📊 Residuals Summary")
    print("-----------------------")
    print(f"Mean Residual:        {residuals.mean():.4f}")
    print(f"Median Residual:      {np.median(residuals):.4f}")
    print(f"Std of Residuals:     {residuals.std():.4f}")
    print(f"Residual 5th pct:     {np.percentile(residuals, 5):.4f}")
    print(f"Residual 95th pct:    {np.percentile(residuals, 95):.4f}")
    print(f"Mean Absolute Residual:  {abs_residuals.mean():.4f}")
    print(f"Median Absolute Residual: {np.median(abs_residuals):.4f}")
    print(f"Mean % Error (nonzero):   {np.nanmean(pct_error):.2f}%\n")

    if return_dataframe:
        df_res = pd.DataFrame({
            "residual": residuals,
            "abs_residual": abs_residuals,
            "pct_error": pct_error,
            "pct_true_rank": ranks
        })
        return df_res

    return None


if __name__ == "__main__":
    print("✔️ forecasting_evaluation.py loaded.")
    print("   Use compute_forecast_metrics, print_split_summary, evaluate_model,")
    print("   evaluate_peak_performance, evaluate_forecast_residuals.")
