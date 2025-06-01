"""
forecast_plots.py

Key plotting routines for electricity‐consumption forecasting:
  1. Feature importance (gain‐based)
  2. Actual vs. Predicted time series (sampled households)
  3. Peak‐period diagnostics (scatter of actual vs. predicted for top percentiles)

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(
    xgb_model,
    feature_names,
    top_n: int = 15,
    importance_type: str = "gain",
    figsize: tuple = (8, 5),
    title: str = "Feature Importance"
):
    """
    Plot the top‐N features by XGBoost importance (default = "gain").

    This function handles two possible key formats from xgb_model.get_score():
      - If keys are "f0", "f1", ... then it maps them via feature_names list.
      - Otherwise, it treats each key as the actual feature name.

    Parameters
    ----------
    xgb_model : xgboost.Booster
        Trained XGBoost model.
    feature_names : list of str
        List of feature names in the same order they were passed to XGBoost
        (only used if get_score keys are numeric indices like "f0", "f1", ...).
    top_n : int, default 15
        Number of top features to display.
    importance_type : str, default "gain"
        One of "weight", "gain", "cover", "total_gain", "total_cover".
    figsize : tuple, default (8, 5)
        Figure size for the bar chart.
    title : str, default "Feature Importance"
        Title for the plot.
    """
    raw_imp = xgb_model.get_score(importance_type=importance_type)
    if not raw_imp:
        print("⚠️  No feature importance to display.")
        return

    mapped = {}
    for key, val in raw_imp.items():
        # Case 1: key looks like "f12" → numeric index into feature_names
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if 0 <= idx < len(feature_names):
                fname = feature_names[idx]
            else:
                # Fallback if index out of range
                fname = key
        else:
            # Case 2: key is already the actual feature name
            fname = key

        mapped[fname] = val

    # Sort by importance descending and take top_n
    sorted_items = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats, gains = zip(*sorted_items)

    plt.figure(figsize=figsize)
    plt.barh(feats[::-1], gains[::-1], color="#4C72B0")
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"📊 {title}")
    plt.tight_layout()
    plt.show()



def plot_actual_vs_predicted(
    df_test: pd.DataFrame,
    date_col: str,
    id_col: str,
    target_col: str,
    y_pred: np.ndarray,
    sample_n: int = 5,
    figsize: tuple = (12, 4),
    title_prefix: str = "Forecast vs Actual"
):
    """
    For a small random sample of households, plot actual vs. predicted time series
    over the test period.

    Parameters
    ----------
    df_test : pd.DataFrame
        Clean test DataFrame used for predictions. Must include [date_col, id_col, target_col]
        and align row‐wise with y_pred.
    date_col : str
        Column name containing the date (x‐axis).
    id_col : str
        Column name containing household identifier (e.g. "LCLid").
    target_col : str
        Column name for true consumption (e.g. "label_1").
    y_pred : np.ndarray
        Predicted values, in the same row order as df_test.
    sample_n : int, default 5
        Number of households to randomly sample.
    figsize : tuple, default (12, 4)
        Size of each subplot (width, height).
    title_prefix : str, default "Forecast vs Actual"
        Prefix inserted into each subplot title.
    """
    df_plot = df_test.reset_index(drop=True).copy()
    df_plot["predicted"] = y_pred

    unique_ids = df_plot[id_col].unique()
    if sample_n > len(unique_ids):
        sample_n = len(unique_ids)

    # Randomly choose sample_n households
    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(unique_ids, size=sample_n, replace=False)

    n_rows = len(sampled_ids)
    fig, axes = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, hid in zip(axes, sampled_ids):
        sub = df_plot[df_plot[id_col] == hid].sort_values(date_col)
        dates = sub[date_col]
        actual = sub[target_col]
        pred = sub["predicted"]

        ax.plot(dates, actual, label="Actual", color="black", linewidth=1.5)
        ax.plot(dates, pred, label="Predicted", color="#4C72B0", linestyle="--", linewidth=1.5)
        ax.set_title(f"{title_prefix} ─ Household: {hid}")
        ax.set_ylabel("kWh")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel(date_col)
    plt.tight_layout()
    plt.show()


def plot_peak_actual_vs_predicted(
    df_test: pd.DataFrame,
    target_col: str,
    y_pred: np.ndarray,
    percentile: float = 90.0,
    figsize: tuple = (6, 6),
    title: str = "Peak Period: Actual vs Predicted"
):
    """
    Plot actual vs. predicted scatter only for the highest‐consumption periods.

    Parameters
    ----------
    df_test : pd.DataFrame
        Clean test DataFrame containing [target_col], aligned with y_pred.
    target_col : str
        Column name for the true target (e.g. "label_1").
    y_pred : np.ndarray
        Predicted values, same row order as df_test.
    percentile : float, default 90.0
        Percentile threshold to define "peak" periods (e.g. top 10% of true values).
    figsize : tuple, default (6, 6)
        Size of the scatter plot.
    title : str, default "Peak Period: Actual vs Predicted"
        Title for the plot.
    """
    y_true = df_test[target_col].values
    threshold = np.percentile(y_true, percentile)

    # Filter for peak rows
    mask = y_true >= threshold
    if not np.any(mask):
        print(f"⚠️  No data points above the {percentile}th percentile ({threshold:.4f}).")
        return

    y_true_peak = y_true[mask]
    y_pred_peak = y_pred[mask]

    plt.figure(figsize=figsize)
    plt.scatter(y_true_peak, y_pred_peak, alpha=0.6, color="#55A868")
    max_val = max(y_true_peak.max(), y_pred_peak.max())
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--")
    plt.xlabel("Actual (kWh)")
    plt.ylabel("Predicted (kWh)")
    plt.title(f"📈 {title}\n(Top {100-percentile:.0f}% of consumption ≥ {threshold:.4f})")
    plt.tight_layout()
    plt.show()
