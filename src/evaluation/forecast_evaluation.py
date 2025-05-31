"""
📊 FORECAST EVALUATION - Model Performance Assessment
===================================================

Comprehensive evaluation functions for forecasting models.
Includes metrics calculation, model comparison, cross-validation, and error analysis.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from prophet.diagnostics import cross_validation, performance_metrics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_forecast_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """
    Calculate comprehensive forecast evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Normalized metrics
    nmae = mae / np.mean(y_true) * 100 if np.mean(y_true) != 0 else float('inf')
    nrmse = rmse / np.mean(y_true) * 100 if np.mean(y_true) != 0 else float('inf')
    
    # Error statistics
    errors = y_pred - y_true
    mean_error = np.mean(errors)  # Bias
    std_error = np.std(errors)
    
    # Directional accuracy (for time series)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    
    # Peak hour performance (if applicable)
    peak_threshold = np.percentile(y_true, 90)
    peak_mask = y_true >= peak_threshold
    if np.sum(peak_mask) > 0:
        peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
        peak_mape = mean_absolute_percentage_error(y_true[peak_mask], y_pred[peak_mask]) * 100
    else:
        peak_mae = None
        peak_mape = None
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'nmae': nmae,
        'nrmse': nrmse,
        'mean_error': mean_error,
        'std_error': std_error,
        'directional_accuracy': directional_accuracy,
        'peak_mae': peak_mae,
        'peak_mape': peak_mape
    }

def evaluate_forecast_model(y_true: np.array, 
                           y_pred: np.array,
                           model_name: str = "Model") -> dict:
    """
    Evaluate forecast model performance with detailed output
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for reporting
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = calculate_forecast_metrics(y_true, y_pred)
    
    print(f"\n📊 {model_name} Forecast Evaluation:")
    print(f"   MAE:  {metrics['mae']:.3f} kWh")
    print(f"   RMSE: {metrics['rmse']:.3f} kWh") 
    print(f"   MAPE: {metrics['mape']:.2f}%")
    print(f"   R²:   {metrics['r2']:.3f}")
    
    return metrics

# Convenience functions for specific models
def evaluate_prophet_forecast(y_true: np.array, 
                             y_pred: np.array,
                             model_name: str = "Prophet") -> dict:
    """Evaluate Prophet forecast performance"""
    return evaluate_forecast_model(y_true, y_pred, model_name)

def evaluate_xgboost_forecast(y_true: np.array, 
                             y_pred: np.array,
                             model_name: str = "XGBoost") -> dict:
    """Evaluate XGBoost forecast performance"""
    return evaluate_forecast_model(y_true, y_pred, model_name)

def prophet_cross_validation(train_df: pd.DataFrame,
                            target_col: str = "total_kwh",
                            initial: str = "730 days",
                            period: str = "30 days", 
                            horizon: str = "30 days") -> pd.DataFrame:
    """
    Perform time series cross-validation with Prophet
    
    Args:
        train_df: Training dataframe
        target_col: Target variable
        initial: Initial training period
        period: Spacing between cutoffs
        horizon: Forecast horizon
        
    Returns:
        Cross-validation results dataframe and metrics
    """
    print("🔄 Prophet Cross-Validation...")
    
    # Import here to avoid circular imports
    from src.models.prophet_forecasting import prepare_prophet_data, create_prophet_model, add_prophet_regressors
    
    # Prepare data
    prophet_df, regressor_cols = prepare_prophet_data(train_df, target_col)
    
    # Create and fit model
    model = create_prophet_model()
    if regressor_cols:
        add_prophet_regressors(model, regressor_cols)
    
    model.fit(prophet_df)
    
    # Perform cross-validation
    cv_results = cross_validation(
        model, 
        initial=initial,
        period=period,
        horizon=horizon,
        parallel="processes"
    )
    
    # Calculate performance metrics
    cv_metrics = performance_metrics(cv_results)
    
    print("✅ Cross-validation completed")
    print(f"📊 Average MAPE: {cv_metrics['mape'].mean():.2f}")
    print(f"📊 Average MAE: {cv_metrics['mae'].mean():.3f}")
    
    return cv_results, cv_metrics

def xgboost_time_series_cv(train_df: pd.DataFrame,
                          feature_cols: list,
                          target_col: str = "total_kwh",
                          n_splits: int = 5,
                          params: dict = None) -> dict:
    """
    Perform time series cross-validation for XGBoost
    
    Args:
        train_df: Training dataframe
        feature_cols: List of feature columns
        target_col: Target variable
        n_splits: Number of CV splits
        params: Model parameters
        
    Returns:
        Cross-validation results
    """
    print(f"🔄 XGBoost Time Series Cross-Validation ({n_splits} splits)...")
    
    # Import here to avoid circular imports
    from src.models.xgboost_forecasting import create_xgboost_model
    
    # Prepare data
    X = train_df[feature_cols].copy()
    y = train_df[target_col].values
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Remove NaN values
    mask = ~(X.isna().any(axis=1) | pd.isna(y))
    X, y = X[mask], y[mask]
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"   Fold {fold + 1}/{n_splits}...")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Train model
        model = create_xgboost_model(params)
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predict and evaluate
        val_pred = model.predict(X_val_cv)
        metrics = evaluate_xgboost_forecast(y_val_cv, val_pred, f"CV Fold {fold + 1}")
        cv_scores.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {
        'mae': np.mean([s['mae'] for s in cv_scores]),
        'rmse': np.mean([s['rmse'] for s in cv_scores]),
        'mape': np.mean([s['mape'] for s in cv_scores]),
        'r2': np.mean([s['r2'] for s in cv_scores])
    }
    
    print(f"\n📊 AVERAGE CV METRICS:")
    print(f"   MAE:  {avg_metrics['mae']:.3f} ± {np.std([s['mae'] for s in cv_scores]):.3f}")
    print(f"   RMSE: {avg_metrics['rmse']:.3f} ± {np.std([s['rmse'] for s in cv_scores]):.3f}")
    print(f"   MAPE: {avg_metrics['mape']:.2f}% ± {np.std([s['mape'] for s in cv_scores]):.2f}%")
    print(f"   R²:   {avg_metrics['r2']:.3f} ± {np.std([s['r2'] for s in cv_scores]):.3f}")
    
    return {
        'cv_scores': cv_scores,
        'avg_metrics': avg_metrics,
        'n_splits': n_splits
    }

def evaluate_multi_household_results(results_dict: dict, 
                                   n_households: int,
                                   metric_type: str = 'test') -> dict:
    """
    Evaluate and aggregate results from multi-household forecasting
    
    Args:
        results_dict: Dictionary with household results
        n_households: Number of households
        metric_type: Type of metrics to aggregate ('test', 'val', 'train')
        
    Returns:
        Dictionary with aggregate metrics
    """
    print(f"📊 Evaluating Multi-Household Results ({n_households} households)")
    
    # Get household list
    households = results_dict.get('households', [])
    
    # Extract metrics for each household
    metrics_list = []
    for household_id in households:
        if household_id in results_dict:
            household_results = results_dict[household_id]
            
            # Calculate metrics if not already present
            if 'y_true' in household_results and 'y_pred' in household_results:
                metrics = calculate_forecast_metrics(
                    household_results['y_true'], 
                    household_results['y_pred']
                )
                metrics_list.append(metrics)
            elif 'metrics' in household_results and metric_type in household_results['metrics']:
                metrics_list.append(household_results['metrics'][metric_type])
    
    if not metrics_list:
        print("⚠️  No metrics found for households")
        return {}
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        'mae': np.mean([m['mae'] for m in metrics_list]),
        'rmse': np.mean([m['rmse'] for m in metrics_list]),
        'mape': np.mean([m['mape'] for m in metrics_list]),
        'r2': np.mean([m['r2'] for m in metrics_list])
    }
    
    print(f"\n📊 AGGREGATE {metric_type.upper()} METRICS ({n_households} households):")
    print(f"   Average MAE:  {aggregate_metrics['mae']:.3f} kWh")
    print(f"   Average RMSE: {aggregate_metrics['rmse']:.3f} kWh")
    print(f"   Average MAPE: {aggregate_metrics['mape']:.2f}%")
    print(f"   Average R²:   {aggregate_metrics['r2']:.3f}")
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'individual_metrics': metrics_list,
        'n_households': n_households
    }

def compare_forecast_models(results_dict: dict) -> pd.DataFrame:
    """
    Compare multiple forecasting models
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        
    Returns:
        Comparison dataframe
    """
    print("📊 COMPARING FORECAST MODELS")
    print("=" * 40)
    
    comparison_data = []
    
    for model_name, results in results_dict.items():
        # Extract metrics - check multiple possible structures
        metrics = None
        
        if 'y_true' in results and 'y_pred' in results:
            # Calculate metrics from predictions
            metrics = calculate_forecast_metrics(results['y_true'], results['y_pred'])
        elif 'metrics' in results and 'test' in results['metrics']:
            metrics = results['metrics']['test']
        elif 'metrics' in results and isinstance(results['metrics'], dict):
            metrics = results['metrics']
        
        if metrics is None:
            print(f"⚠️  No metrics found for {model_name}")
            continue
        
        comparison_data.append({
            'Model': model_name,
            'MAE': metrics.get('mae', np.nan),
            'RMSE': metrics.get('rmse', np.nan),
            'MAPE (%)': metrics.get('mape', np.nan),
            'R²': metrics.get('r2', np.nan),
            'NMAE (%)': metrics.get('nmae', np.nan),
            'Directional Acc (%)': metrics.get('directional_accuracy', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        print("⚠️  No valid metrics found for comparison")
        return comparison_df
    
    # Rank models by primary metrics
    comparison_df['MAE_Rank'] = comparison_df['MAE'].rank()
    comparison_df['RMSE_Rank'] = comparison_df['RMSE'].rank()
    comparison_df['MAPE_Rank'] = comparison_df['MAPE (%)'].rank()
    comparison_df['R²_Rank'] = comparison_df['R²'].rank(ascending=False)
    
    # Calculate average rank
    rank_cols = ['MAE_Rank', 'RMSE_Rank', 'MAPE_Rank', 'R²_Rank']
    comparison_df['Avg_Rank'] = comparison_df[rank_cols].mean(axis=1)
    comparison_df = comparison_df.sort_values('Avg_Rank')
    
    print("\n📋 MODEL COMPARISON RESULTS:")
    print(comparison_df.round(3).to_string(index=False))
    
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
    print(f"🔍 RESIDUAL ANALYSIS - {model_name}")
    print("=" * 40)
    
    residuals = y_true - y_pred
    
    # Basic residual statistics
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q50': np.percentile(residuals, 50),
        'q75': np.percentile(residuals, 75)
    }
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    is_normal = shapiro_p > 0.05
    
    # Autocorrelation (if enough data points)
    if len(residuals) > 10:
        autocorr_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr_lag1 = None
    
    # Heteroscedasticity check (residuals vs predictions)
    correlation_residuals_pred = np.corrcoef(residuals, y_pred)[0, 1]
    
    print(f"   Mean residual: {residual_stats['mean']:.4f}")
    print(f"   Std residual:  {residual_stats['std']:.4f}")
    print(f"   Normality test: {'PASS' if is_normal else 'FAIL'} (p={shapiro_p:.4f})")
    if autocorr_lag1 is not None:
        print(f"   Lag-1 autocorr: {autocorr_lag1:.4f}")
    print(f"   Heteroscedasticity: {abs(correlation_residuals_pred):.4f}")
    
    return {
        'residual_stats': residual_stats,
        'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': is_normal},
        'autocorr_lag1': autocorr_lag1,
        'heteroscedasticity': correlation_residuals_pred,
        'residuals': residuals
    }

def calculate_forecast_intervals(y_pred: np.array, residuals: np.array, confidence_level: float = 0.95) -> dict:
    """
    Calculate prediction intervals for forecasts
    
    Args:
        y_pred: Point predictions
        residuals: Model residuals from training/validation
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with prediction intervals
    """
    # Calculate residual standard deviation
    residual_std = np.std(residuals)
    
    # Calculate z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Calculate intervals
    margin_of_error = z_score * residual_std
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error
    
    return {
        'point_forecast': y_pred,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'margin_of_error': margin_of_error,
        'residual_std': residual_std
    }

def evaluate_forecast_by_periods(y_true: np.array, 
                                y_pred: np.array, 
                                dates: np.array = None,
                                period_type: str = 'monthly') -> pd.DataFrame:
    """
    Evaluate forecast performance by time periods
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Array of dates
        period_type: Type of period ('monthly', 'quarterly', 'seasonal')
        
    Returns:
        DataFrame with performance by period
    """
    if dates is None:
        print("⚠️  No dates provided, cannot evaluate by periods")
        return pd.DataFrame()
    
    print(f"📅 FORECAST EVALUATION BY {period_type.upper()} PERIODS")
    print("=" * 50)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'actual': y_true,
        'predicted': y_pred
    })
    
    # Add period columns
    if period_type == 'monthly':
        df['period'] = df['date'].dt.to_period('M')
    elif period_type == 'quarterly':
        df['period'] = df['date'].dt.to_period('Q')
    elif period_type == 'seasonal':
        df['season'] = df['date'].dt.month % 12 // 3 + 1
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df['period'] = df['season'].map(season_map)
    else:
        print(f"⚠️  Unknown period type: {period_type}")
        return pd.DataFrame()
    
    # Calculate metrics by period
    period_results = []
    
    for period in df['period'].unique():
        period_data = df[df['period'] == period]
        
        if len(period_data) > 0:
            metrics = calculate_forecast_metrics(
                period_data['actual'].values,
                period_data['predicted'].values
            )
            
            period_results.append({
                'Period': str(period),
                'Count': len(period_data),
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE (%)': metrics['mape'],
                'R²': metrics['r2'],
                'Mean_Actual': period_data['actual'].mean(),
                'Mean_Predicted': period_data['predicted'].mean()
            })
    
    period_df = pd.DataFrame(period_results)
    
    if not period_df.empty:
        print(period_df.round(3).to_string(index=False))
    
    return period_df

def evaluate_peak_hour_forecasting(y_true: np.array, 
                                  y_pred: np.array,
                                  peak_threshold_percentile: float = 90) -> dict:
    """
    Evaluate forecasting performance specifically for peak hours/periods
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        peak_threshold_percentile: Percentile to define peak periods
        
    Returns:
        Dictionary with peak hour evaluation results
    """
    print("⚡ PEAK HOUR FORECASTING EVALUATION")
    print("=" * 40)
    
    # Define peak threshold
    peak_threshold = np.percentile(y_true, peak_threshold_percentile)
    
    # Identify peak periods
    peak_mask = y_true >= peak_threshold
    non_peak_mask = ~peak_mask
    
    # Peak period metrics
    if np.sum(peak_mask) > 0:
        peak_metrics = calculate_forecast_metrics(y_true[peak_mask], y_pred[peak_mask])
        
        # Peak detection accuracy
        pred_peaks = y_pred >= peak_threshold
        peak_detection_accuracy = np.mean(peak_mask == pred_peaks) * 100
        peak_precision = np.sum(peak_mask & pred_peaks) / np.sum(pred_peaks) * 100 if np.sum(pred_peaks) > 0 else 0
        peak_recall = np.sum(peak_mask & pred_peaks) / np.sum(peak_mask) * 100
        
        print(f"   Peak threshold: {peak_threshold:.2f} kWh (≥{peak_threshold_percentile}th percentile)")
        print(f"   Peak periods: {np.sum(peak_mask)} out of {len(y_true)} ({np.sum(peak_mask)/len(y_true)*100:.1f}%)")
        print(f"   Peak MAE: {peak_metrics['mae']:.3f} kWh")
        print(f"   Peak MAPE: {peak_metrics['mape']:.2f}%")
        print(f"   Peak detection accuracy: {peak_detection_accuracy:.1f}%")
        print(f"   Peak precision: {peak_precision:.1f}%")
        print(f"   Peak recall: {peak_recall:.1f}%")
    else:
        peak_metrics = None
        peak_detection_accuracy = None
        peak_precision = None
        peak_recall = None
        print("   No peak periods found with current threshold")
    
    # Non-peak period metrics
    if np.sum(non_peak_mask) > 0:
        non_peak_metrics = calculate_forecast_metrics(y_true[non_peak_mask], y_pred[non_peak_mask])
        print(f"   Non-peak MAE: {non_peak_metrics['mae']:.3f} kWh")
        print(f"   Non-peak MAPE: {non_peak_metrics['mape']:.2f}%")
    else:
        non_peak_metrics = None
    
    return {
        'peak_threshold': peak_threshold,
        'peak_periods_count': np.sum(peak_mask),
        'peak_periods_pct': np.sum(peak_mask)/len(y_true)*100,
        'peak_metrics': peak_metrics,
        'non_peak_metrics': non_peak_metrics,
        'peak_detection_accuracy': peak_detection_accuracy,
        'peak_precision': peak_precision,
        'peak_recall': peak_recall,
        'peak_mask': peak_mask
    }

def create_forecast_summary_report(results_dict: dict) -> dict:
    """
    Create comprehensive summary report for forecast evaluation
    
    Args:
        results_dict: Dictionary with model results
        
    Returns:
        Dictionary with summary report
    """
    print("📋 CREATING FORECAST SUMMARY REPORT")
    print("=" * 40)
    
    summary = {
        'models_evaluated': list(results_dict.keys()),
        'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_comparison': compare_forecast_models(results_dict)
    }
    
    # Best performing model by different metrics
    comparison_df = summary['model_comparison']
    
    if not comparison_df.empty:
        best_models = {
            'best_mae': comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model'],
            'best_rmse': comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model'],
            'best_mape': comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model'],
            'best_r2': comparison_df.loc[comparison_df['R²'].idxmax(), 'Model'],
            'best_overall': comparison_df.loc[comparison_df['Avg_Rank'].idxmin(), 'Model']
        }
        
        summary['best_models'] = best_models
        
        print("\n🏆 BEST PERFORMING MODELS:")
        for metric, model in best_models.items():
            print(f"   {metric.replace('_', ' ').title()}: {model}")
    
    # Model recommendations
    if 'best_overall' in summary.get('best_models', {}):
        best_model = summary['best_models']['best_overall']
        summary['recommendation'] = f"Recommended model: {best_model} (best overall performance)"
    else:
        summary['recommendation'] = "Unable to determine best model"
    
    print(f"\n💡 {summary['recommendation']}")
    
    return summary

if __name__ == "__main__":
    print("📊 Forecast Evaluation Module")
    print("Usage: from src.evaluation.forecast_evaluation import calculate_forecast_metrics") 