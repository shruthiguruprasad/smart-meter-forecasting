"""
📊 FORECAST VISUALIZATION - Forecasting Results Plots
====================================================

Comprehensive visualization functions for forecasting models.
Includes forecast plots, model comparison, error analysis, and residual plots.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_forecast_vs_actual(y_true: np.array, 
                           y_pred: np.array,
                           dates: np.array = None,
                           model_name: str = "Model",
                           title: str = None,
                           figsize: tuple = (14, 8),
                           save_path: str = None) -> plt.Figure:
    """
    Plot forecast vs actual values over time
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Array of dates
        model_name: Name of the model
        title: Custom title
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Use index if no dates provided
    x_axis = pd.to_datetime(dates) if dates is not None else range(len(y_true))
    
    # Main forecast plot
    ax1.plot(x_axis, y_true, label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(x_axis, y_pred, label='Predicted', color='red', alpha=0.7, linewidth=1.5)
    
    ax1.set_ylabel('Consumption (kWh)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax1.set_title(f'{model_name} - Forecast vs Actual', fontsize=14, fontweight='bold')
    
    # Error plot
    errors = y_pred - y_true
    ax2.plot(x_axis, errors, color='green', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Error (kWh)', fontsize=12)
    ax2.set_xlabel('Date' if dates is not None else 'Time Period', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    if dates is not None:
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_forecast_scatter(y_true: np.array, 
                         y_pred: np.array,
                         model_name: str = "Model",
                         figsize: tuple = (10, 8),
                         save_path: str = None) -> plt.Figure:
    """
    Create scatter plot of predicted vs actual values
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    ax.set_xlabel('Actual Consumption (kWh)', fontsize=12)
    ax.set_ylabel('Predicted Consumption (kWh)', fontsize=12)
    ax.set_title(f'{model_name} - Predicted vs Actual (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text with metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    textstr = f'MAE: {mae:.3f} kWh\nRMSE: {rmse:.3f} kWh'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_model_comparison(results_dict: dict,
                         metric: str = 'mae',
                         figsize: tuple = (12, 8),
                         save_path: str = None) -> plt.Figure:
    """
    Compare multiple models by a specific metric
    
    Args:
        results_dict: Dictionary with model results
        metric: Metric to compare ('mae', 'rmse', 'mape', 'r2')
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    models = []
    values = []
    
    for model_name, results in results_dict.items():
        # Extract test metrics
        if 'metrics' in results and 'test' in results['metrics']:
            metrics = results['metrics']['test']
        elif 'metrics' in results:
            metrics = results['metrics']
        else:
            continue
        
        if metric in metrics:
            models.append(model_name)
            values.append(metrics[metric])
    
    if not models:
        print(f"⚠️  No {metric} values found for comparison")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Formatting
    metric_names = {
        'mae': 'Mean Absolute Error (kWh)',
        'rmse': 'Root Mean Square Error (kWh)',
        'mape': 'Mean Absolute Percentage Error (%)',
        'r2': 'R-squared'
    }
    
    ax.set_ylabel(metric_names.get(metric, metric.upper()), fontsize=12)
    ax.set_title(f'Model Comparison - {metric_names.get(metric, metric.upper())}', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    if metric == 'r2':
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    
    bars[best_idx].set_color('lightgreen')
    bars[best_idx].set_edgecolor('darkgreen')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_forecast_residuals(y_true: np.array,
                           y_pred: np.array,
                           dates: np.array = None,
                           model_name: str = "Model",
                           figsize: tuple = (15, 10),
                           save_path: str = None) -> plt.Figure:
    """
    Create comprehensive residual analysis plots
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Array of dates
        model_name: Name of the model
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Residuals over time
    x_axis = pd.to_datetime(dates) if dates is not None else range(len(residuals))
    ax1.plot(x_axis, residuals, color='blue', alpha=0.7, linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('Residuals Over Time', fontweight='bold')
    ax1.set_ylabel('Residuals (kWh)')
    ax1.grid(True, alpha=0.3)
    if dates is not None:
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Residuals vs Predicted
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Residuals vs Predicted', fontweight='bold')
    ax2.set_xlabel('Predicted Values (kWh)')
    ax2.set_ylabel('Residuals (kWh)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    ax3.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Distribution of Residuals', fontweight='bold')
    ax3.set_xlabel('Residuals (kWh)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
         np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
    y = y * len(residuals) * (residuals.max() - residuals.min()) / 30  # Scale to histogram
    ax3.plot(x, y, 'r--', linewidth=2, label='Normal')
    ax3.legend()
    
    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_feature_importance(feature_importance: pd.DataFrame,
                           top_k: int = 20,
                           figsize: tuple = (10, 12),
                           save_path: str = None) -> plt.Figure:
    """
    Plot feature importance for XGBoost model
    
    Args:
        feature_importance: DataFrame with feature importance
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Get top features
    top_features = feature_importance.head(top_k)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importance - XGBoost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        ax.text(value + max(top_features['importance'])*0.01, i, 
                f'{value:.3f}', va='center', fontsize=9)
    
    # Invert y-axis to show most important at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_forecast_with_intervals(y_true: np.array,
                                y_pred: np.array,
                                lower_bound: np.array,
                                upper_bound: np.array,
                                dates: np.array = None,
                                model_name: str = "Model",
                                confidence_level: float = 0.95,
                                figsize: tuple = (14, 8),
                                save_path: str = None) -> plt.Figure:
    """
    Plot forecast with prediction intervals
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        lower_bound: Lower prediction bound
        upper_bound: Upper prediction bound
        dates: Array of dates
        model_name: Name of the model
        confidence_level: Confidence level for intervals
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_axis = pd.to_datetime(dates) if dates is not None else range(len(y_true))
    
    # Plot actual values
    ax.plot(x_axis, y_true, label='Actual', color='blue', linewidth=2, alpha=0.8)
    
    # Plot predictions
    ax.plot(x_axis, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.8)
    
    # Plot prediction intervals
    ax.fill_between(x_axis, lower_bound, upper_bound, 
                    alpha=0.3, color='red', 
                    label=f'{confidence_level*100:.0f}% Prediction Interval')
    
    ax.set_ylabel('Consumption (kWh)', fontsize=12)
    ax.set_xlabel('Date' if dates is not None else 'Time Period', fontsize=12)
    ax.set_title(f'{model_name} - Forecast with {confidence_level*100:.0f}% Prediction Intervals', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if dates is not None:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def plot_multi_household_results(results_dict: dict,
                                 metric: str = 'mae',
                                 figsize: tuple = (12, 8),
                                 save_path: str = None) -> plt.Figure:
    """
    Plot results for multiple households
    
    Args:
        results_dict: Dictionary with household results
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    households = []
    values = []
    
    for household_id, results in results_dict.items():
        if household_id in ['aggregate_metrics', 'households']:
            continue
            
        if 'metrics' in results and 'test' in results['metrics']:
            metrics = results['metrics']['test']
            if metric in metrics:
                households.append(household_id)
                values.append(metrics[metric])
    
    if not households:
        print(f"⚠️  No {metric} values found for households")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(range(len(households)), values, color='lightcoral', 
                  edgecolor='darkred', alpha=0.7)
    
    # Add aggregate line if available
    if 'aggregate_metrics' in results_dict and metric in results_dict['aggregate_metrics']:
        avg_value = results_dict['aggregate_metrics'][metric]
        ax.axhline(y=avg_value, color='blue', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_value:.3f}')
        ax.legend()
    
    # Formatting
    ax.set_xticks(range(len(households)))
    ax.set_xticklabels([f'HH{i+1}' for i in range(len(households))])
    
    metric_names = {
        'mae': 'Mean Absolute Error (kWh)',
        'rmse': 'Root Mean Square Error (kWh)',
        'mape': 'Mean Absolute Percentage Error (%)',
        'r2': 'R-squared'
    }
    
    ax.set_ylabel(metric_names.get(metric, metric.upper()), fontsize=12)
    ax.set_xlabel('Household', fontsize=12)
    ax.set_title(f'Multi-Household Forecast Performance - {metric_names.get(metric, metric.upper())}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved: {save_path}")
    
    return fig

def create_forecast_dashboard(results_dict: dict,
                             y_true: np.array,
                             y_pred: np.array,
                             dates: np.array = None,
                             model_name: str = "Model",
                             save_path: str = None) -> plt.Figure:
    """
    Create comprehensive forecast dashboard
    
    Args:
        results_dict: Dictionary with model results
        y_true: Actual values
        y_pred: Predicted values
        dates: Array of dates
        model_name: Name of the model
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Time series plot
    ax1 = fig.add_subplot(gs[0, :])
    x_axis = pd.to_datetime(dates) if dates is not None else range(len(y_true))
    ax1.plot(x_axis, y_true, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
    ax1.plot(x_axis, y_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
    ax1.set_title(f'{model_name} - Forecast vs Actual', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val, max_val = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Predicted vs Actual', fontweight='bold', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Predicted', fontweight='bold', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual histogram
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution', fontweight='bold', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature importance (if available)
    if 'feature_importance' in results_dict:
        ax5 = fig.add_subplot(gs[2, :])
        top_features = results_dict['feature_importance'].head(15)
        bars = ax5.barh(range(len(top_features)), top_features['importance'])
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['feature'], fontsize=8)
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 15 Feature Importance', fontweight='bold', fontsize=10)
        ax5.invert_yaxis()
    
    # Overall title
    fig.suptitle(f'{model_name} - Comprehensive Forecast Analysis', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Dashboard saved: {save_path}")
    
    return fig

if __name__ == "__main__":
    print("📊 Forecast Visualization Module")
    print("Usage: from src.visualization.forecast_plots import plot_forecast_vs_actual") 