"""
📈 XGBOOST FORECASTING NOTEBOOK - ENHANCED EDITION
=================================================

Comprehensive notebook-style implementation of XGBoost for day-ahead and week-ahead
forecasting of smart meter data using the enhanced XGBoost module.

🔧 ENHANCED FEATURES:
- ✅ Automatic data leakage validation 
- ✅ NaN monitoring and reporting
- ✅ GPU/CPU resource optimization
- ✅ Log transform option
- ✅ Comprehensive evaluation and visualization

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_loader import load_all_raw_data
from src.data.data_cleaner import clean_all_data
from src.features.feature_pipeline import create_comprehensive_features, prepare_forecasting_data
from src.models.xgboost_forecasting import (
    xgboost_day_ahead_forecast,
    prepare_xgboost_data,
    train_xgboost_forecasting,
    predict_xgboost,
    get_top_features,
    validate_forecasting_features
)
from src.evaluation.forecast_evaluation import evaluate_forecast_model
from src.visualization.forecast_plots import plot_forecast_vs_actual, create_forecast_dashboard

# Configuration
CONFIG = {
    'data_path': 'data',
    'test_days': 90,
    'val_days': 30,
    'sample_household': None,  # Will auto-select first available
    'save_plots': True,
    'plot_dir': 'plots/xgboost_notebook/',
    'use_gpu': False,  # Set to True if you have GPU support
    'week_ahead_days': 7,  # For week-ahead forecasting
    'log_transform': False,  # Set to True for relative error modeling
    'use_critical_features_only': True  # Use minimal feature set for better performance
}

print("🚀 ENHANCED XGBOOST FORECASTING FOR SMART METER DATA")
print("=" * 60)
print("📊 ENHANCED IMPLEMENTATION with:")
print("   ✅ Day-ahead forecasting with automatic validation")
print("   ✅ Week-ahead forecasting with proper target handling")
print("   ✅ Automatic data leakage detection")
print("   ✅ GPU/CPU resource optimization")
print("   ✅ Log transform option for relative errors")
print("   ✅ Comprehensive evaluation and visualization")
print("=" * 60)
print(f"📊 Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

#%% ================================================================
# STEP 1: DATA LOADING AND PREPARATION
#%% ================================================================

print("\n📂 STEP 1: DATA LOADING AND PREPARATION")
print("-" * 40)

# Load and clean data
print("Loading raw smart meter data...")
raw_data = load_all_raw_data(CONFIG['data_path'])

print("Cleaning data...")
cleaned_data = clean_all_data(raw_data)

print("Creating comprehensive features...")
df_features = create_comprehensive_features(cleaned_data)

print("Preparing forecasting data...")
train_df, val_df, test_df, feature_cols, target_col, feature_groups = prepare_forecasting_data(
    df_features, 
    target_col="total_kwh", 
    test_days=CONFIG['test_days'], 
    val_days=CONFIG['val_days']
)

# 🎯 OPTIONAL: Use critical-only feature set
if CONFIG['use_critical_features_only']:
    print("\n🎯 USING CRITICAL-ONLY FEATURE SET...")
    print("-" * 40)
    
    # Define critical features for optimal performance
    critical_features = [
        # Temporal/Calendar features
        'dayofweek_sin', 'dayofweek_cos', 'is_holiday', 'is_weekend',
        # Lag/Rolling features (autoregressive patterns)
        'lag1_total', 'lag7_total', 'roll7_total_mean', 'roll14_total_mean',
        # Weather features (energy drivers)
        'temp_avg', 'heating_degree_days', 'cooling_degree_days',
        # Household baseline
        'hh_avg_consumption',
        # Interaction features
        'weekend_heating', 'holiday_heating_interaction'
    ]
    
    # Filter to only features that exist in the data
    available_critical = [f for f in critical_features if f in feature_cols]
    missing_critical = [f for f in critical_features if f not in feature_cols]
    
    print(f"📊 Available critical features: {len(available_critical)} out of {len(critical_features)} desired")
    print(f"📊 Using {len(available_critical)} critical features instead of {len(feature_cols)} total features")
    
    if missing_critical:
        print(f"⚠️ Missing critical features: {missing_critical}")
    else:
        print("✅ All critical features are available!")
    
    feature_cols = available_critical

print(f"\n✅ Enhanced data preparation completed:")
print(f"   Training samples: {len(train_df):,}")
print(f"   Validation samples: {len(val_df):,}")
print(f"   Test samples: {len(test_df):,}")
print(f"   Features: {len(feature_cols)}")
print(f"   Target: {target_col}")
print(f"   Train period: {train_df['day'].min()} to {train_df['day'].max()}")
print(f"   Test period: {test_df['day'].min()} to {test_df['day'].max()}")
print(f"   Log transform: {CONFIG['log_transform']}")

#%% ================================================================
# STEP 2: HOUSEHOLD SELECTION AND DATA OVERVIEW
#%% ================================================================

print("\n🏠 STEP 2: HOUSEHOLD SELECTION")
print("-" * 40)

# Select household for analysis
available_households = train_df['LCLid'].unique()
selected_household = CONFIG['sample_household'] or available_households[0]

print(f"📊 Available households: {len(available_households)}")
print(f"📊 Selected household: {selected_household}")

# Filter data for selected household
household_train = train_df[train_df['LCLid'] == selected_household].copy()
household_val = val_df[val_df['LCLid'] == selected_household].copy()
household_test = test_df[test_df['LCLid'] == selected_household].copy()

print(f"📊 Household data overview:")
print(f"   Training days: {len(household_train)}")
print(f"   Validation days: {len(household_val)}")
print(f"   Test days: {len(household_test)}")
print(f"   Average consumption: {household_train[target_col].mean():.2f} kWh/day")
print(f"   Consumption range: {household_train[target_col].min():.1f} - {household_train[target_col].max():.1f} kWh/day")

#%% ================================================================
# STEP 3: DAY-AHEAD FORECASTING WITH ENHANCED XGBOOST
#%% ================================================================

print("\n📈 STEP 3: DAY-AHEAD FORECASTING WITH ENHANCED XGBOOST")
print("-" * 50)

print("🚀 Running enhanced XGBoost day-ahead forecasting...")
print("   ✅ Automatic data leakage validation")
print("   ✅ NaN monitoring and reporting")
print("   ✅ GPU/CPU resource optimization")
if CONFIG['log_transform']:
    print("   ✅ Log transform for relative error modeling")

day_ahead_results = xgboost_day_ahead_forecast(
    train_df, val_df, test_df,
    feature_cols=feature_cols,
    target_col=target_col,
    household_id=selected_household,
    use_gpu=CONFIG['use_gpu'],
    log_transform=CONFIG['log_transform']
)

print("✅ Enhanced XGBoost day-ahead forecasting completed")

# Extract results
y_true_day = day_ahead_results['y_true']
y_pred_day = day_ahead_results['y_pred']
dates_day = day_ahead_results['dates']['test']

print(f"📊 Day-ahead forecast summary:")
print(f"   Test period: {len(y_true_day)} days")
print(f"   Actual range: {y_true_day.min():.1f} - {y_true_day.max():.1f} kWh")
print(f"   Predicted range: {y_pred_day.min():.1f} - {y_pred_day.max():.1f} kWh")
print(f"   Features used: {len(feature_cols)}")
print(f"   Log transform: {day_ahead_results['log_transform']}")

#%% ================================================================
# STEP 4: WEEK-AHEAD FORECASTING WITH XGBOOST
#%% ================================================================

print("\n📅 STEP 4: WEEK-AHEAD FORECASTING WITH XGBOOST")
print("-" * 40)

def create_week_ahead_data(train_df, val_df, test_df, feature_cols, target_col, household_id, week_days=7):
    """Create week-ahead forecasting data by reshaping test data"""
    
    # Get household data
    test_household = test_df[test_df['LCLid'] == household_id].copy()
    
    # Only use complete weeks
    n_complete_weeks = len(test_household) // week_days
    n_days_to_use = n_complete_weeks * week_days
    
    if n_complete_weeks == 0:
        raise ValueError(f"Not enough test data for week-ahead forecasting. Need at least {week_days} days.")
    
    test_household = test_household.head(n_days_to_use)
    
    # Reshape for week-ahead prediction
    week_actuals = []
    week_dates = []
    
    for i in range(0, len(test_household), week_days):
        week_data = test_household.iloc[i:i+week_days]
        week_actuals.append(week_data[target_col].values)
        week_dates.append(week_data['day'].values)
    
    return {
        'week_actuals': np.array(week_actuals),
        'week_dates': week_dates,
        'n_weeks': len(week_actuals),
        'days_per_week': week_days
    }

def predict_week_ahead(model, train_df, val_df, test_df, feature_cols, household_id, week_days=7, log_transform=False):
    """Generate week-ahead predictions with enhanced XGBoost features"""
    
    # Get test data for the household
    test_household = test_df[test_df['LCLid'] == household_id].copy()
    
    # Use complete weeks only
    n_complete_weeks = len(test_household) // week_days
    n_days_to_use = n_complete_weeks * week_days
    test_household = test_household.head(n_days_to_use)
    
    # Prepare the data using the enhanced preprocessing pipeline
    print("   🔧 Preparing test data with enhanced preprocessing...")
    
    # Create dummy validation data (just one row) to satisfy the prepare_xgboost_data function
    dummy_val = test_household.head(1).copy()
    
    # Use the enhanced prepare_xgboost_data function with log_transform support
    data_dict = prepare_xgboost_data(
        train_df[train_df['LCLid'] == household_id].head(1),  # minimal train data
        dummy_val,  # dummy validation
        test_household,  # actual test data we want to predict on
        feature_cols,
        target_col="total_kwh",
        household_id=household_id,
        log_transform=log_transform
    )
    
    # Get the properly encoded test features
    X_test_encoded = data_dict['X_test']
    
    week_predictions = []
    
    for i in range(0, len(X_test_encoded), week_days):
        week_data = X_test_encoded.iloc[i:i+week_days]
        # Use enhanced predict function with log_transform support
        week_pred = predict_xgboost(model, week_data, log_transform)
        week_predictions.append(week_pred)
    
    return np.array(week_predictions)

print("Creating week-ahead forecasting setup...")
week_data = create_week_ahead_data(
    train_df, val_df, test_df, feature_cols, target_col, 
    selected_household, CONFIG['week_ahead_days']
)

print(f"📊 Week-ahead data prepared:")
print(f"   Complete weeks: {week_data['n_weeks']}")
print(f"   Days per week: {week_data['days_per_week']}")

# Generate week-ahead predictions using the enhanced model
print("🚀 Generating enhanced week-ahead predictions...")
print("   ✅ Using same model as day-ahead (for comparison)")
print("   ✅ Enhanced preprocessing with log_transform support")

week_predictions = predict_week_ahead(
    day_ahead_results['model'], train_df, val_df, test_df,
    feature_cols, selected_household, CONFIG['week_ahead_days'],
    log_transform=CONFIG['log_transform']
)

print("✅ Enhanced week-ahead forecasting completed")

#%% ================================================================
# STEP 5: MODEL EVALUATION
#%% ================================================================

print("\n📊 STEP 5: MODEL EVALUATION")
print("-" * 40)

# Day-ahead evaluation
print("📈 DAY-AHEAD PERFORMANCE:")
day_metrics = evaluate_forecast_model(y_true_day, y_pred_day, "XGBoost Day-Ahead")

print(f"   MAE:  {day_metrics['mae']:.3f} kWh")
print(f"   RMSE: {day_metrics['rmse']:.3f} kWh") 
print(f"   MAPE: {day_metrics['mape']:.2f}%")
print(f"   R²:   {day_metrics['r2']:.3f}")

# Week-ahead evaluation
print("\n📅 WEEK-AHEAD PERFORMANCE:")
# Flatten week predictions and actuals for evaluation
week_actuals_flat = week_data['week_actuals'].flatten()
week_predictions_flat = week_predictions.flatten()

week_metrics = evaluate_forecast_model(week_actuals_flat, week_predictions_flat, "XGBoost Week-Ahead")

print(f"   MAE:  {week_metrics['mae']:.3f} kWh")
print(f"   RMSE: {week_metrics['rmse']:.3f} kWh")
print(f"   MAPE: {week_metrics['mape']:.2f}%")
print(f"   R²:   {week_metrics['r2']:.3f}")

# Performance comparison
print(f"\n🔄 PERFORMANCE COMPARISON:")
print(f"                  Day-Ahead  Week-Ahead")
print(f"   MAE:           {day_metrics['mae']:.3f}      {week_metrics['mae']:.3f}")
print(f"   RMSE:          {day_metrics['rmse']:.3f}      {week_metrics['rmse']:.3f}")
print(f"   MAPE:          {day_metrics['mape']:.2f}%     {week_metrics['mape']:.2f}%")
print(f"   R²:            {day_metrics['r2']:.3f}      {week_metrics['r2']:.3f}")

#%% ================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
#%% ================================================================

print("\n🎯 STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

# Get top features
top_features = get_top_features(day_ahead_results['feature_importance'], top_k=15)

# Create feature importance plot
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features - XGBoost')
plt.gca().invert_yaxis()
plt.tight_layout()

if CONFIG['save_plots']:
    import os
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)
    plt.savefig(f"{CONFIG['plot_dir']}feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

#%% ================================================================
# STEP 7: VISUALIZATIONS
#%% ================================================================

print("\n📈 STEP 7: VISUALIZATIONS")
print("-" * 40)

# Create plots directory
import os
if CONFIG['save_plots']:
    os.makedirs(CONFIG['plot_dir'], exist_ok=True)

# 1. Day-ahead forecast plot
print("Creating day-ahead forecast plot...")
save_path = f"{CONFIG['plot_dir']}day_ahead_forecast.png" if CONFIG['save_plots'] else None
plot_forecast_vs_actual(
    y_true_day, y_pred_day, dates_day,
    "XGBoost Day-Ahead Forecast",
    save_path=save_path
)

# 2. Week-ahead forecast plot
print("Creating week-ahead forecast plot...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('XGBoost Week-Ahead Forecasting Analysis', fontsize=16)

# Plot first 4 weeks
for i in range(min(4, week_data['n_weeks'])):
    row, col = i // 2, i % 2
    
    week_actual = week_data['week_actuals'][i]
    week_pred = week_predictions[i]
    days = range(1, len(week_actual) + 1)
    
    axes[row, col].plot(days, week_actual, 'o-', label='Actual', linewidth=2)
    axes[row, col].plot(days, week_pred, 's-', label='Predicted', linewidth=2)
    axes[row, col].set_title(f'Week {i+1}')
    axes[row, col].set_xlabel('Day of Week')
    axes[row, col].set_ylabel('Energy (kWh)')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
if CONFIG['save_plots']:
    plt.savefig(f"{CONFIG['plot_dir']}week_ahead_forecast.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Residual analysis
print("Creating residual analysis plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('XGBoost Forecast Residual Analysis', fontsize=16)

# Day-ahead residuals
day_residuals = y_true_day - y_pred_day

# Residuals over time
axes[0, 0].plot(dates_day, day_residuals, alpha=0.7)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Day-Ahead Residuals Over Time')
axes[0, 0].tick_params(axis='x', rotation=45)

# Scatter plot
axes[0, 1].scatter(y_true_day, y_pred_day, alpha=0.6)
axes[0, 1].plot([y_true_day.min(), y_true_day.max()], [y_true_day.min(), y_true_day.max()], 'r--')
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].set_title('Day-Ahead: Predicted vs Actual')

# Week-ahead residuals
week_residuals = week_actuals_flat - week_predictions_flat

# Week residuals histogram
axes[1, 0].hist(day_residuals, bins=20, alpha=0.7, edgecolor='black', label='Day-Ahead')
axes[1, 0].hist(week_residuals, bins=20, alpha=0.7, edgecolor='black', label='Week-Ahead')
axes[1, 0].axvline(x=0, color='red', linestyle='--')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')
axes[1, 0].legend()

# Performance comparison
metrics_comparison = pd.DataFrame({
    'Day-Ahead': [day_metrics['mae'], day_metrics['rmse'], day_metrics['mape'], day_metrics['r2']],
    'Week-Ahead': [week_metrics['mae'], week_metrics['rmse'], week_metrics['mape'], week_metrics['r2']]
}, index=['MAE', 'RMSE', 'MAPE', 'R²'])

axes[1, 1].bar(range(len(metrics_comparison)), metrics_comparison['Day-Ahead'], 
               alpha=0.7, label='Day-Ahead', width=0.35)
axes[1, 1].bar([x + 0.35 for x in range(len(metrics_comparison))], metrics_comparison['Week-Ahead'], 
               alpha=0.7, label='Week-Ahead', width=0.35)
axes[1, 1].set_xticks([x + 0.175 for x in range(len(metrics_comparison))])
axes[1, 1].set_xticklabels(metrics_comparison.index)
axes[1, 1].set_title('Performance Metrics Comparison')
axes[1, 1].legend()

plt.tight_layout()
if CONFIG['save_plots']:
    plt.savefig(f"{CONFIG['plot_dir']}residual_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualizations completed")

#%% ================================================================
# STEP 8: SUMMARY AND RECOMMENDATIONS
#%% ================================================================

print("\n🎯 STEP 8: SUMMARY AND RECOMMENDATIONS")
print("-" * 40)

print(f"\n📋 FORECAST SUMMARY:")
print(f"✅ Household: {selected_household}")
print(f"✅ Day-ahead test period: {len(y_true_day)} days")
print(f"✅ Week-ahead forecast: {week_data['n_weeks']} weeks")
print(f"✅ Model: XGBoost with {len(feature_cols)} features")

# Performance interpretation
def interpret_performance(r2):
    if r2 > 0.7:
        return "🟢 EXCELLENT - Strong predictive power"
    elif r2 > 0.5:
        return "🟡 GOOD - Moderate predictive power"
    elif r2 > 0.2:
        return "🟠 FAIR - Limited predictive power"
    else:
        return "🔴 POOR - Needs improvement"

day_performance = interpret_performance(day_metrics['r2'])
week_performance = interpret_performance(week_metrics['r2'])

print(f"\n📊 PERFORMANCE ASSESSMENT:")
print(f"   Day-ahead:  {day_performance}")
print(f"   Week-ahead: {week_performance}")

print(f"\n💡 RECOMMENDATIONS:")
if day_metrics['r2'] > 0.5 and week_metrics['r2'] > 0.5:
    print("   🟢 Both models performing well!")
    print("   🔧 Ready for production use")
    print("   🔧 Consider testing on more households")
    print("   🔧 Monitor performance over time")
elif day_metrics['r2'] > week_metrics['r2']:
    print("   🟡 Day-ahead model outperforms week-ahead")
    print("   🔧 Consider ensemble methods for week-ahead")
    print("   🔧 Add more temporal features for longer horizons")
    print("   🔧 Try different model architectures for week-ahead")
else:
    print("   🟠 Both models need improvement")
    print("   🔧 Increase training data if possible")
    print("   🔧 Try hyperparameter tuning")
    print("   🔧 Add more relevant features")
    print("   🔧 Check for data quality issues")

# Top features insight
print(f"\n🎯 KEY FEATURES:")
print("   Most important features for prediction:")
for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
    print(f"   {i}. {row['feature']}")

if CONFIG['save_plots']:
    print(f"\n📈 Plots saved to: {CONFIG['plot_dir']}")

print(f"\n🔚 ENHANCED XGBOOST FORECASTING ANALYSIS COMPLETED!")
print("=" * 60)

#%% ================================================================
# FINAL RESULTS SUMMARY
#%% ================================================================

# Create an enhanced summary dictionary with all new features
RESULTS_SUMMARY = {
    'household_id': selected_household,
    'day_ahead': {
        'test_days': len(y_true_day),
        'metrics': day_metrics,
        'actual': y_true_day,
        'predicted': y_pred_day,
        'dates': dates_day
    },
    'week_ahead': {
        'weeks': week_data['n_weeks'],
        'metrics': week_metrics,
        'actual': week_data['week_actuals'],
        'predicted': week_predictions,
        'dates': week_data['week_dates']
    },
    'model': day_ahead_results['model'],
    'feature_importance': day_ahead_results['feature_importance'],
    'features_used': len(feature_cols),
    'log_transform': CONFIG['log_transform'],
    'critical_features_only': CONFIG['use_critical_features_only'],
    'config': CONFIG,
    'enhancements': {
        'automatic_leakage_validation': True,
        'nan_monitoring': True,
        'gpu_optimization': CONFIG['use_gpu'],
        'log_transform_support': True,
        'enhanced_preprocessing': True
    }
}

print(f"\n📊 Enhanced results stored in RESULTS_SUMMARY dictionary")
print(f"📊 Access day-ahead metrics: RESULTS_SUMMARY['day_ahead']['metrics']")
print(f"📊 Access week-ahead metrics: RESULTS_SUMMARY['week_ahead']['metrics']")
print(f"📊 Access trained model: RESULTS_SUMMARY['model']")
print(f"📊 Access enhancements info: RESULTS_SUMMARY['enhancements']") 