"""
📊 FORECAST EVALUATION - Model Performance Assessment
===================================================

Comprehensive evaluation functions for forecasting models.
Provides the core evaluation functions needed by the model building pipeline.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_regression_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    Compute comprehensive regression metrics for forecasting evaluation.
    This is the main function expected by the model building pipeline.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'error': 'Empty arrays provided'}
    
    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE with handling for zero values
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Normalized metrics
    mean_actual = np.mean(y_true)
    nmae = (mae / mean_actual * 100) if mean_actual != 0 else float('inf')
    nrmse = (rmse / mean_actual * 100) if mean_actual != 0 else float('inf')
    
    # Error statistics
    errors = y_pred - y_true
    mean_error = np.mean(errors)  # Bias
    std_error = np.std(errors)
    
    # Directional accuracy (for time series)
    if len(y_true) > 1:
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    else:
        directional_accuracy = np.nan
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'NMAE': nmae,
        'NRMSE': nrmse,
        'mean_error': mean_error,
        'std_error': std_error,
        'directional_accuracy': directional_accuracy,
        'n_samples': len(y_true)
    }


def print_regression_results(metrics: dict, prefix: str = "Model") -> None:
    """
    Print formatted regression results.
    This function is expected by the model building pipeline.
    
    Args:
        metrics: Dictionary of metrics from compute_regression_metrics
        prefix: Prefix for the output (e.g., "Train", "Val", "Test")
    """
    if 'error' in metrics:
        print(f"   ❌ {prefix}: {metrics['error']}")
        return
    
    print(f"   📊 {prefix} Performance:")
    print(f"      MAE:  {metrics['MAE']:.4f} kWh")
    print(f"      RMSE: {metrics['RMSE']:.4f} kWh")
    print(f"      MAPE: {metrics['MAPE']:.2f}%")
    print(f"      R²:   {metrics['R2']:.4f}")
    print(f"      Samples: {metrics['n_samples']:,}")


def evaluate_forecast_model(y_true: np.array, 
                           y_pred: np.array,
                           model_name: str = "Model",
                           detailed: bool = True) -> dict:
    """
    Comprehensive forecast model evaluation with optional detailed output
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for reporting
        detailed: Whether to print detailed results
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = compute_regression_metrics(y_true, y_pred)
    
    if detailed:
        print(f"\n🎯 {model_name} FORECAST EVALUATION")
        print("=" * 40)
        print_regression_results(metrics, "Overall")
        
        # Additional insights
        if metrics.get('mean_error', 0) > 0.1:
            print(f"   ⚠️  Positive bias detected: {metrics['mean_error']:.3f} kWh")
        elif metrics.get('mean_error', 0) < -0.1:
            print(f"   ⚠️  Negative bias detected: {metrics['mean_error']:.3f} kWh")
        
        if metrics.get('MAPE', 0) > 20:
            print(f"   ⚠️  High MAPE: {metrics['MAPE']:.1f}% - consider model improvements")
        
        if metrics.get('R2', 0) < 0.7:
            print(f"   ⚠️  Low R²: {metrics['R2']:.3f} - model may need enhancement")
    
    return metrics


def compare_forecast_models(results_dict: dict) -> pd.DataFrame:
    """
    Compare multiple forecasting models
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        
    Returns:
        Comparison dataframe sorted by performance
    """
    print("\n📊 FORECAST MODEL COMPARISON")
    print("=" * 40)
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        # Extract metrics from different possible structures
        metrics = None
        
        if 'actuals' in results and 'predictions' in results:
            # Structure from train_and_evaluate functions
            y_true = results['actuals'].get('test', results['actuals'].get('val'))
            y_pred = results['predictions'].get('test', results['predictions'].get('val'))
            if y_true is not None and y_pred is not None:
                metrics = compute_regression_metrics(y_true, y_pred)
        elif 'metrics' in results and 'test' in results['metrics']:
            metrics = results['metrics']['test']
        elif 'y_true' in results and 'y_pred' in results:
            metrics = compute_regression_metrics(results['y_true'], results['y_pred'])
        elif isinstance(results, dict) and 'MAE' in results:
            metrics = results
        
        if metrics is None or 'error' in metrics:
            print(f"   ⚠️ No valid metrics found for {model_name}")
            continue
        
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics.get('MAE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAPE (%)': metrics.get('MAPE', np.nan),
            'R²': metrics.get('R2', np.nan),
            'NMAE (%)': metrics.get('NMAE', np.nan),
            'Samples': metrics.get('n_samples', 0)
        })
    
    if not comparison_data:
        print("   ❌ No valid models to compare")
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank models by primary metrics (lower is better for error metrics)
    comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
    comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
    comparison_df['MAPE_Rank'] = comparison_df['MAPE (%)'].rank()
    comparison_df['R²_Rank'] = comparison_df['R²'].rank(ascending=False)
    
    # Calculate overall ranking
    rank_cols = ['MAE_Rank', 'RMSE_Rank', 'MAPE_Rank', 'R²_Rank']
    comparison_df['Overall_Rank'] = comparison_df[rank_cols].mean(axis=1)
    comparison_df = comparison_df.sort_values('Overall_Rank')
    
    # Display results
    display_cols = ['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²', 'Overall_Rank']
    print(comparison_df[display_cols].round(4).to_string(index=False))
    
    # Identify best model
    best_model = comparison_df.iloc[0]['Model']
    print(f"\n🏆 Best performing model: {best_model}")
    
    return comparison_df


def evaluate_forecast_residuals(y_true: np.array, y_pred: np.array, model_name: str = "Model") -> dict:
    """
    Analyze forecast residuals for model diagnostics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with residual analysis results
    """
    print(f"\n🔍 RESIDUAL ANALYSIS - {model_name}")
    print("=" * 40)
    
    residuals = y_true - y_pred
    
    # Basic residual statistics
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q50': np.percentile(residuals, 50),  # median
        'q75': np.percentile(residuals, 75)
    }
    
    # Normality test (use subset if too many data points)
    sample_size = min(5000, len(residuals))
    sample_residuals = np.random.choice(residuals, sample_size, replace=False) if len(residuals) > sample_size else residuals
    
    try:
        shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
        is_normal = shapiro_p > 0.05
    except:
        shapiro_stat, shapiro_p = np.nan, np.nan
        is_normal = False
    
    # Autocorrelation check
    if len(residuals) > 10:
        try:
            autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        except:
            autocorr_lag1 = np.nan
    else:
        autocorr_lag1 = np.nan
    
    # Heteroscedasticity check
    try:
        correlation_residuals_pred = np.corrcoef(residuals, y_pred)[0, 1]
    except:
        correlation_residuals_pred = np.nan
    
    # Print results
    print(f"   Mean residual: {residual_stats['mean']:.4f}")
    print(f"   Std residual:  {residual_stats['std']:.4f}")
    print(f"   Residual range: [{residual_stats['min']:.4f}, {residual_stats['max']:.4f}]")
    
    if not np.isnan(shapiro_p):
        print(f"   Normality test: {'PASS' if is_normal else 'FAIL'} (p={shapiro_p:.4f})")
    
    if not np.isnan(autocorr_lag1):
        autocorr_status = "HIGH" if abs(autocorr_lag1) > 0.3 else "MODERATE" if abs(autocorr_lag1) > 0.1 else "LOW"
        print(f"   Lag-1 autocorr: {autocorr_lag1:.4f} ({autocorr_status})")
    
    if not np.isnan(correlation_residuals_pred):
        hetero_status = "HIGH" if abs(correlation_residuals_pred) > 0.3 else "MODERATE" if abs(correlation_residuals_pred) > 0.1 else "LOW"
        print(f"   Heteroscedasticity: {abs(correlation_residuals_pred):.4f} ({hetero_status})")
    
    return {
        'residual_stats': residual_stats,
        'normality_test': {
            'statistic': shapiro_stat, 
            'p_value': shapiro_p, 
            'is_normal': is_normal
        },
        'autocorr_lag1': autocorr_lag1,
        'heteroscedasticity': correlation_residuals_pred,
        'residuals': residuals
    }


def evaluate_peak_performance(y_true: np.array, 
                             y_pred: np.array,
                             peak_threshold_percentile: float = 90) -> dict:
    """
    Evaluate forecasting performance specifically for peak consumption periods
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        peak_threshold_percentile: Percentile to define peak periods
        
    Returns:
        Dictionary with peak performance evaluation
    """
    print(f"\n⚡ PEAK PERFORMANCE ANALYSIS (≥{peak_threshold_percentile}th percentile)")
    print("=" * 50)
    
    # Define peak threshold
    peak_threshold = np.percentile(y_true, peak_threshold_percentile)
    
    # Identify peak and non-peak periods
    peak_mask = y_true >= peak_threshold
    non_peak_mask = ~peak_mask
    
    results = {'peak_threshold': peak_threshold}
    
    # Peak period analysis
    if np.sum(peak_mask) > 0:
        peak_metrics = compute_regression_metrics(y_true[peak_mask], y_pred[peak_mask])
        
        # Peak detection metrics
        pred_peaks = y_pred >= peak_threshold
        true_positives = np.sum(peak_mask & pred_peaks)
        false_positives = np.sum(~peak_mask & pred_peaks)
        false_negatives = np.sum(peak_mask & ~pred_peaks)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Peak threshold: {peak_threshold:.2f} kWh")
        print(f"   Peak periods: {np.sum(peak_mask)} / {len(y_true)} ({np.sum(peak_mask)/len(y_true)*100:.1f}%)")
        print(f"   Peak MAE: {peak_metrics['MAE']:.4f} kWh")
        print(f"   Peak RMSE: {peak_metrics['RMSE']:.4f} kWh")
        print(f"   Peak MAPE: {peak_metrics['MAPE']:.2f}%")
        print(f"   Peak detection precision: {precision:.3f}")
        print(f"   Peak detection recall: {recall:.3f}")
        print(f"   Peak detection F1-score: {f1_score:.3f}")
        
        results.update({
            'peak_count': np.sum(peak_mask),
            'peak_percentage': np.sum(peak_mask)/len(y_true)*100,
            'peak_metrics': peak_metrics,
            'peak_precision': precision,
            'peak_recall': recall,
            'peak_f1_score': f1_score
        })
    else:
        print("   No peak periods found")
        results.update({'peak_count': 0, 'peak_metrics': None})
    
    # Non-peak period analysis
    if np.sum(non_peak_mask) > 0:
        non_peak_metrics = compute_regression_metrics(y_true[non_peak_mask], y_pred[non_peak_mask])
        print(f"   Non-peak MAE: {non_peak_metrics['MAE']:.4f} kWh")
        print(f"   Non-peak MAPE: {non_peak_metrics['MAPE']:.2f}%")
        results['non_peak_metrics'] = non_peak_metrics
    
    return results


def create_evaluation_summary(results: dict, model_name: str = "Model") -> dict:
    """
    Create a comprehensive evaluation summary
    
    Args:
        results: Results dictionary from model training
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation summary
    """
    print(f"\n📋 EVALUATION SUMMARY - {model_name}")
    print("=" * 50)
    
    summary = {
        'model_name': model_name,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Extract test metrics
    if 'actuals' in results and 'predictions' in results:
        y_true = results['actuals']['test']
        y_pred = results['predictions']['test']
        test_metrics = compute_regression_metrics(y_true, y_pred)
        summary['test_metrics'] = test_metrics
        
        print_regression_results(test_metrics, "Test Set")
        
        # Performance assessment
        performance_grade = "A" if test_metrics['MAPE'] < 10 else "B" if test_metrics['MAPE'] < 20 else "C"
        print(f"\n   🎯 Performance Grade: {performance_grade}")
        
        # Recommendations
        recommendations = []
        if test_metrics['MAPE'] > 20:
            recommendations.append("High MAPE suggests need for model improvement")
        if test_metrics['R2'] < 0.7:
            recommendations.append("Low R² indicates poor model fit")
        if test_metrics.get('mean_error', 0) > 0.5:
            recommendations.append("Significant bias detected - check feature engineering")
        
        if recommendations:
            print(f"\n   💡 Recommendations:")
            for rec in recommendations:
                print(f"      • {rec}")
        else:
            print(f"\n   ✅ Model performance looks good!")
        
        summary['performance_grade'] = performance_grade
        summary['recommendations'] = recommendations
    
    return summary


if __name__ == "__main__":
    print("📊 Forecast Evaluation Module")
    print("=" * 40)
    print("✅ CORE FUNCTIONS:")
    print("   🎯 compute_regression_metrics() - Main metrics calculation")
    print("   📊 print_regression_results() - Formatted output")
    print("   🔍 evaluate_forecast_model() - Comprehensive evaluation")
    print("   📈 compare_forecast_models() - Model comparison")
    print("   🔍 evaluate_forecast_residuals() - Residual analysis")
    print("   ⚡ evaluate_peak_performance() - Peak period analysis")
    print("=" * 40)
    print("Usage:")
    print("  from src.evaluation.forecast_evaluation import compute_regression_metrics")
    print("  metrics = compute_regression_metrics(y_true, y_pred)") 