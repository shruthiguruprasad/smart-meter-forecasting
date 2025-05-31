"""
🚀 DAY-AHEAD FORECASTING DEMO
============================

Comprehensive demonstration of day-ahead electricity consumption forecasting
using Prophet and XGBoost models with the complete pipeline.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.data_loader import load_all_raw_data
from src.data.data_cleaner import clean_all_data
from src.features.feature_pipeline import create_comprehensive_features, prepare_forecasting_data
from src.models.prophet_forecasting import prophet_day_ahead_forecast
from src.models.xgboost_forecasting import xgboost_day_ahead_forecast, get_top_features
from src.evaluation.forecast_evaluation import (
    evaluate_forecast_model, compare_forecast_models, create_forecast_summary_report,
    evaluate_multi_household_results
)
from src.visualization.forecast_plots import (
    plot_forecast_vs_actual, plot_model_comparison, 
    create_forecast_dashboard, plot_feature_importance
)

def run_day_ahead_forecasting_demo(data_path: str = "data", 
                                  test_days: int = 90,
                                  val_days: int = 30,
                                  n_households: int = 3,
                                  run_prophet: bool = True,
                                  run_xgboost: bool = True,
                                  save_plots: bool = True):
    """
    Run comprehensive day-ahead forecasting demonstration
    
    Args:
        data_path: Path to data directory
        test_days: Number of days for test set
        val_days: Number of days for validation set
        n_households: Number of households to forecast
        run_prophet: Whether to run Prophet model
        run_xgboost: Whether to run XGBoost model
        save_plots: Whether to save plots
    """
    print("🚀 DAY-AHEAD ELECTRICITY CONSUMPTION FORECASTING DEMO")
    print("=" * 60)
    
    # 1. Load and clean data
    print("\n📂 STEP 1: LOADING AND CLEANING DATA")
    print("-" * 40)
    
    # Load raw data
    raw_data = load_all_raw_data(data_path)
    
    # Clean data
    cleaned_data = clean_all_data(raw_data)
    
    # 2. Create comprehensive features
    print("\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-" * 40)
    
    df_features = create_comprehensive_features(cleaned_data)
    
    # 3. Prepare data for forecasting
    print("\n📊 STEP 3: PREPARING FORECASTING DATA")
    print("-" * 40)
    
    train_df, val_df, test_df, feature_cols, target_col, feature_groups = prepare_forecasting_data(
        df_features, target_col="total_kwh", test_days=test_days, val_days=val_days
    )
    
    print(f"✅ Data prepared for forecasting:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: {target_col}")
    print(f"   Train period: {train_df['day'].min()} to {train_df['day'].max()}")
    print(f"   Test period: {test_df['day'].min()} to {test_df['day'].max()}")
    
    # 4. Run forecasting models
    print("\n🤖 STEP 4: RUNNING FORECASTING MODELS")
    print("-" * 40)
    
    results = {}
    
    # Select a sample household for demonstration
    available_households = train_df['LCLid'].unique()
    sample_household = available_households[0]
    print(f"📊 Sample household for demo: {sample_household}")
    
    # Prophet Model
    if run_prophet:
        print("\n📈 Running Prophet forecasting...")
        try:
            prophet_results = prophet_day_ahead_forecast(
                train_df, test_df, target_col=target_col, 
                household_id=sample_household, include_regressors=True
            )
            
            # Evaluate Prophet results using evaluation module
            prophet_metrics = evaluate_forecast_model(
                prophet_results['y_true'], 
                prophet_results['y_pred'], 
                "Prophet"
            )
            prophet_results['metrics'] = prophet_metrics
            
            results['Prophet'] = prophet_results
            print("✅ Prophet forecasting completed")
        except Exception as e:
            print(f"❌ Prophet forecasting failed: {e}")
    
    # XGBoost Model
    if run_xgboost:
        print("\n🚀 Running XGBoost forecasting...")
        try:
            xgboost_results = xgboost_day_ahead_forecast(
                train_df, val_df, test_df, feature_cols, target_col=target_col,
                household_id=sample_household, use_gpu=False
            )
            
            # Evaluate XGBoost results using evaluation module
            xgboost_metrics = evaluate_forecast_model(
                xgboost_results['y_true'], 
                xgboost_results['y_pred'], 
                "XGBoost"
            )
            xgboost_results['metrics'] = xgboost_metrics
            
            results['XGBoost'] = xgboost_results
            print("✅ XGBoost forecasting completed")
        except Exception as e:
            print(f"❌ XGBoost forecasting failed: {e}")
    
    # 5. Model evaluation and comparison
    print("\n📊 STEP 5: MODEL EVALUATION AND COMPARISON")
    print("-" * 40)
    
    if len(results) > 0:
        # Compare models
        comparison_df = compare_forecast_models(results)
        
        # Create summary report
        summary_report = create_forecast_summary_report(results)
        
        # Show top features (if XGBoost available)
        if 'XGBoost' in results:
            print("\n🎯 TOP FEATURES (XGBoost):")
            top_features = get_top_features(results['XGBoost']['feature_importance'], top_k=15)
    
    # 6. Create visualizations
    print("\n📈 STEP 6: CREATING VISUALIZATIONS")
    print("-" * 40)
    
    if save_plots:
        plot_dir = "plots/forecasting/"
        import os
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None
    
    # Individual model plots
    for model_name, model_results in results.items():
        print(f"\n📊 Creating plots for {model_name}...")
        
        # Get common data format
        y_true = model_results['y_true']
        y_pred = model_results['y_pred']
        dates = model_results.get('dates', None)
        
        # Forecast vs Actual plot
        save_path = f"{plot_dir}{model_name}_forecast_vs_actual.png" if save_plots else None
        plot_forecast_vs_actual(
            y_true, y_pred, dates, model_name, 
            save_path=save_path
        )
        
        # Dashboard
        save_path = f"{plot_dir}{model_name}_dashboard.png" if save_plots else None
        create_forecast_dashboard(
            model_results, y_true, y_pred, dates, model_name,
            save_path=save_path
        )
    
    # Model comparison plot
    if len(results) > 1:
        print("\n📊 Creating model comparison plots...")
        for metric in ['mae', 'rmse', 'mape', 'r2']:
            save_path = f"{plot_dir}model_comparison_{metric}.png" if save_plots else None
            plot_model_comparison(results, metric=metric, save_path=save_path)
    
    # Feature importance plot (XGBoost)
    if 'XGBoost' in results:
        print("\n📊 Creating feature importance plot...")
        save_path = f"{plot_dir}feature_importance.png" if save_plots else None
        plot_feature_importance(
            results['XGBoost']['feature_importance'], 
            top_k=20, save_path=save_path
        )
    
    # 7. Multi-household forecasting example
    print("\n🏠 STEP 7: MULTI-HOUSEHOLD FORECASTING EXAMPLE")
    print("-" * 40)
    
    if run_xgboost and n_households > 1:
        print(f"📊 Running XGBoost for {n_households} households...")
        from src.models.xgboost_forecasting import xgboost_multi_household_forecast
        
        try:
            multi_results = xgboost_multi_household_forecast(
                train_df, val_df, test_df, feature_cols, target_col=target_col,
                n_households=n_households
            )
            
            # Evaluate multi-household results
            multi_evaluation = evaluate_multi_household_results(
                multi_results, n_households, metric_type='test'
            )
            multi_results.update(multi_evaluation)
            
            # Multi-household plot
            if save_plots:
                from src.visualization.forecast_plots import plot_multi_household_results
                save_path = f"{plot_dir}multi_household_results.png"
                plot_multi_household_results(multi_results, metric='mae', save_path=save_path)
            
        except Exception as e:
            print(f"❌ Multi-household forecasting failed: {e}")
    
    # 8. Summary and recommendations
    print("\n🎯 STEP 8: SUMMARY AND RECOMMENDATIONS")
    print("-" * 40)
    
    if len(results) > 0:
        print("\n📋 FORECASTING DEMO COMPLETED SUCCESSFULLY!")
        print(f"✅ Models evaluated: {list(results.keys())}")
        
        if 'best_models' in summary_report:
            print(f"🏆 Best model overall: {summary_report['best_models']['best_overall']}")
        
        print(f"\n💡 {summary_report['recommendation']}")
        
        if save_plots:
            print(f"📈 Plots saved in: {plot_dir}")
    else:
        print("❌ No models completed successfully")
    
    print("\n🔚 DEMO COMPLETED")
    return results

def run_quick_demo():
    """Run a quick demo with minimal data"""
    print("🚀 QUICK FORECASTING DEMO")
    print("=" * 30)
    
    # Run with reduced parameters for quick testing
    results = run_day_ahead_forecasting_demo(
        data_path="data",
        test_days=30,    # Shorter test period
        val_days=15,     # Shorter validation period
        n_households=2,  # Fewer households
        run_prophet=True,
        run_xgboost=True,
        save_plots=True
    )
    
    return results

if __name__ == "__main__":
    print("📊 Day-Ahead Forecasting Demo")
    print("=" * 40)
    
    # Choose demo type
    demo_type = input("Choose demo type (1=Quick, 2=Full): ").strip()
    
    if demo_type == "1":
        results = run_quick_demo()
    else:
        results = run_day_ahead_forecasting_demo()
    
    print("\n🎉 Demo completed! Check the results and plots.") 