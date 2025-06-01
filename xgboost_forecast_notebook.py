"""
📈 XGBOOST FORECASTING NOTEBOOK - ENHANCED EDITION
=================================================

Comprehensive notebook-style implementation of XGBoost for day-ahead and week-ahead
forecasting of smart meter data using the enhanced clean pipeline architecture.

🔧 ENHANCED FEATURES:
- ✅ Clean separation of concerns (features → splitting → modeling)
- ✅ Optuna hyperparameter optimization 
- ✅ GPU/CPU resource optimization
- ✅ Log transform option for relative error modeling
- ✅ Comprehensive evaluation and visualization
- ✅ Leakage-safe feature engineering

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from data.data_loader import load_all_raw_data
from data.data_cleaner import clean_and_merge_all_data     # or clean_all_data, whichever you use
from features.feature_pipeline import create_comprehensive_features
from features.splitters import prepare_forecasting_data, prepare_weekahead_data
from models.xgboost_forecasting import (
    train_and_evaluate_dayahead,
    train_and_evaluate_weekahead,
    prepare_xgboost_data,
    predict_xgboost
)
from evaluation.forecast_evaluation import (
    compute_regression_metrics,
    print_regression_results,
    evaluate_forecast_model,
    compare_forecast_models,
    evaluate_peak_performance
)

# Configuration
CONFIG = {
    'data_path': 'data',
    'test_days': 90,
    'val_days': 30,
    'sample_household': None,  # Will auto-select first available
    'save_plots': True,
    'plot_dir': 'plots/xgboost_notebook/',
    'use_gpu': False,  # Set to True if you have GPU
    'log_transform': True,  # Use log transform for relative error modeling
    'n_trials': 60,  # Optuna optimization trials
    'seed': 42
}

print("🚀 ENHANCED XGBOOST FORECASTING WITH CLEAN PIPELINE")
print("=" * 60)
print("📊 NEW CLEAN ARCHITECTURE:")
print("   1️⃣ Feature Engineering (leakage-safe)")
print("   2️⃣ Data Splitting (chronological)")
print("   3️⃣ Model Building (with Optuna)")
print("   4️⃣ Evaluation & Visualization")
print("=" * 60)
print("✅ ENHANCED FEATURES:")
print("   🔍 Optuna hyperparameter optimization")
print("   ⚡ GPU acceleration support" if CONFIG['use_gpu'] else "   💻 CPU processing")
print("   📈 Log transform for relative errors" if CONFIG['log_transform'] else "   📊 Linear scale modeling")
print("   🔒 Automatic leakage prevention")
print("   📊 Comprehensive model evaluation")
print("=" * 60)

#%% ================================================================
# STEP 1: DATA LOADING AND FEATURE ENGINEERING
#%% ================================================================

print("\n1️⃣ STEP 1: DATA LOADING AND FEATURE ENGINEERING")
print("-" * 50)

# Load and clean data
print("📂 Loading raw smart meter data...")
raw_data = load_all_raw_data(CONFIG['data_path'])

print("🧹 Cleaning data...")
cleaned_data = clean_and_merge_all_data(raw_data)   # or clean_all_data(raw_data), depending on your naming

print("🔧 Creating comprehensive leakage-safe features...")
df_features = create_comprehensive_features(cleaned_data)

print(f"✅ Feature engineering completed:")
print(f"   📊 Total samples: {len(df_features):,}")
print(f"   📅 Date range: {df_features['day'].min()} to {df_features['day'].max()}")
print(f"   🏠 Households: {df_features['LCLid'].nunique()}")
print(f"   🔧 Features created: {len(df_features.columns)} total columns")

#%% ================================================================
# STEP 2: DATA SPLITTING (CHRONOLOGICAL & LEAKAGE-SAFE)
#%% ================================================================

print("\n2️⃣ STEP 2: DATA SPLITTING")
print("-" * 30)

print("📊 Preparing day-ahead forecasting data...")
# Note: We no longer pass `household_meta` here—static columns (Acorn, stdorToU, etc.) have
# already been merged into df_features upstream.
train_df, val_df, test_df, feature_cols, target_col, feature_groups = prepare_forecasting_data(
    df_features,
    target_col="total_kwh", 
    test_days=CONFIG['test_days'], 
    val_days=CONFIG['val_days']
)

print(f"✅ Day-ahead data preparation:")
print(f"   📈 Training samples: {len(train_df):,}")
print(f"   🔍 Validation samples: {len(val_df):,}")
print(f"   🎯 Test samples: {len(test_df):,}")
print(f"   🔧 Features: {len(feature_cols)}")
print(f"   🎯 Target: {target_col}")

print("\n📅 Preparing week-ahead forecasting data...")
# Again, no `household_meta` argument
train_df7, val_df7, test_df7, feature_cols7, target7, feature_groups7 = prepare_weekahead_data(
    raw_data,
    df_features,
    test_days=CONFIG['test_days'],
    val_days=CONFIG['val_days']
)

print(f"✅ Week-ahead data preparation:")
print(f"   📈 Training samples: {len(train_df7):,}")
print(f"   🔍 Validation samples: {len(val_df7):,}")
print(f"   🎯 Test samples: {len(test_df7):,}")
print(f"   🔧 Features: {len(feature_cols7)}")
print(f"   🎯 Target: {target7}")

# Show feature breakdown
print(f"\n📊 Feature groups breakdown:")
for group_name, group_features in feature_groups.items():
    print(f"   {group_name}: {len(group_features)} features")

#%% ================================================================
# STEP 3: HOUSEHOLD SELECTION
#%% ================================================================

print("\n3️⃣ STEP 3: HOUSEHOLD SELECTION")
print("-" * 35)

# Select household for analysis
available_households = train_df['LCLid'].unique()
selected_household = CONFIG['sample_household'] or available_households[0]

print(f"🏠 Available households: {len(available_households)}")
print(f"🏠 Selected household: {selected_household}")

# Get household statistics using the correct target column (label_1, not total_kwh)
household_train = train_df[train_df['LCLid'] == selected_household]
print(f"📊 Household consumption profile:")
print(f"   📈 Average: {household_train[target_col].mean():.2f} kWh/day")
print(f"   📊 Range: {household_train[target_col].min():.1f} - {household_train[target_col].max():.1f} kWh/day")
print(f"   📈 Std dev: {household_train[target_col].std():.2f} kWh/day")

#%% ================================================================
# STEP 4: DAY-AHEAD FORECASTING WITH ENHANCED XGBOOST
#%% ================================================================

print("\n4️⃣ STEP 4: DAY-AHEAD FORECASTING")
print("-" * 40)

print("🚀 Starting enhanced day-ahead forecasting pipeline...")
print(f"   🔍 Optuna trials: {CONFIG['n_trials']}")
print(f"   ⚡ GPU acceleration: {CONFIG['use_gpu']}")
print(f"   📈 Log transform: {CONFIG['log_transform']}")

# Filter data for selected household
household_train = train_df[train_df['LCLid'] == selected_household].copy()
household_val = val_df[val_df['LCLid'] == selected_household].copy()
household_test = test_df[test_df['LCLid'] == selected_household].copy()

print(f"📊 Household data: {len(household_train)} train, {len(household_val)} val, {len(household_test)} test")

# Run enhanced day-ahead forecasting
day_ahead_results = train_and_evaluate_dayahead(
    household_train,
    household_val, 
    household_test,
    feature_cols,
    target_col=target_col,
    use_gpu=CONFIG['use_gpu'],
    n_trials=CONFIG['n_trials'],
    log_transform=CONFIG['log_transform']
)

print("✅ Day-ahead forecasting completed!")


from evaluation.forecast_evaluation import compute_regression_metrics

y_true_day = day_ahead_results['actuals']['test']
y_pred_day = day_ahead_results['predictions']['test']
day_metrics = compute_regression_metrics(y_true_day, y_pred_day)

dates_day = household_test['day'].values

print(f"\n📈 DAY-AHEAD RESULTS:")
print_regression_results(day_metrics, "Test Set")

#%% ================================================================
# STEP 5: WEEK-AHEAD FORECASTING
#%% ================================================================

print("\n5️⃣ STEP 5: WEEK-AHEAD FORECASTING")
print("-" * 40)

print("🚀 Starting enhanced week-ahead forecasting pipeline...")

# Filter week-ahead data for selected household
household_train7 = train_df7[train_df7['LCLid'] == selected_household].copy()
household_val7 = val_df7[val_df7['LCLid'] == selected_household].copy()
household_test7 = test_df7[test_df7['LCLid'] == selected_household].copy()

print(f"📊 Week-ahead data: {len(household_train7)} train, {len(household_val7)} val, {len(household_test7)} test")

# Run enhanced week-ahead forecasting
week_ahead_results = train_and_evaluate_weekahead(
    household_train7,
    household_val7,
    household_test7,
    feature_cols7,
    target_col=target7,
    use_gpu=CONFIG['use_gpu'],
    n_trials=CONFIG['n_trials'],
    log_transform=CONFIG['log_transform']
)

print("✅ Week-ahead forecasting completed!")

# Extract key results
y_true_week = week_ahead_results['actuals']['test']
y_pred_week = week_ahead_results['predictions']['test']
week_metrics = compute_regression_metrics(y_true_week, y_pred_week)

print(f"\n📅 WEEK-AHEAD RESULTS:")
print_regression_results(week_metrics, "Test Set")

#%% ================================================================
# STEP 6: MODEL COMPARISON AND ANALYSIS
#%% ================================================================

print("\n6️⃣ STEP 6: MODEL COMPARISON AND ANALYSIS")
print("-" * 45)

# Compare models
model_results = {
    'Day-Ahead XGBoost': day_ahead_results,
    'Week-Ahead XGBoost': week_ahead_results
}

print("📊 Comparing forecasting horizons...")
comparison_df = compare_forecast_models(model_results)

# Performance summary
print(f"\n📈 PERFORMANCE SUMMARY:")
print(f"   Day-ahead MAPE: {day_metrics['MAPE']:.2f}%")
print(f"   Week-ahead MAPE: {week_metrics['MAPE']:.2f}%")
print(f"   Day-ahead R²: {day_metrics['R2']:.3f}")
print(f"   Week-ahead R²: {week_metrics['R2']:.3f}")

# Feature importance analysis
print(f"\n🎯 TOP 10 IMPORTANT FEATURES (Day-Ahead):")
top_features = day_ahead_results['feature_importance'].head(10)
for idx, row in top_features.iterrows():
    print(f"   {idx+1:2d}. {row['feature']}: {row['importance']:.4f}")

#%% ================================================================
# STEP 7: PEAK PERFORMANCE ANALYSIS
#%% ================================================================

print("\n7️⃣ STEP 7: PEAK PERFORMANCE ANALYSIS")
print("-" * 45)

# Analyze peak consumption forecasting
day_peak_analysis = evaluate_peak_performance(y_true_day, y_pred_day, 90)
week_peak_analysis = evaluate_peak_performance(y_true_week, y_pred_week, 90)


