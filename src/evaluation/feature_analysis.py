"""
📊 FEATURE ANALYSIS - Stage 0: Consumption Driver Analysis & Evaluation
=====================================================================

Stage 0: Consumption Driver Analysis
- Global driver ranking via SHAP values
- Feature group importance analysis
- Local household-level insights
- Comprehensive evaluation and reporting

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_performance(model_dict: dict, data_dict: dict) -> dict:
    """
    Comprehensive model performance evaluation
    
    Args:
        model_dict: Results from train_xgboost_model
        data_dict: Results from prepare_modeling_data
        
    Returns:
        Dictionary with detailed performance metrics
    """
    print("📈 EVALUATING MODEL PERFORMANCE")
    print("=" * 32)
    
    model = model_dict['model']
    
    # Handle case where data_dict might have different keys
    if 'X_train' in data_dict and 'X_test' in data_dict:
        X_train = data_dict['X_train'].copy()
        X_test = data_dict['X_test'].copy()
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
    else:
        # Handle case where we might have different data structure
        raise ValueError("Expected X_train, X_test, y_train, y_test in data_dict")
    
    # Handle categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    if len(categorical_cols) > 0:
        print(f"🔄 Encoding {len(categorical_cols)} categorical variables...")
        from sklearn.preprocessing import LabelEncoder
        
        # Create and fit label encoders
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on train data only
            le.fit(X_train[col].astype(str))
            # Transform both train and test
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # Convert to DMatrix for XGBoost predictions
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Get predictions using DMatrix
    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)
    
    # Calculate comprehensive metrics
    performance = {
        'train': {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred),
            'mape': np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred),
            'mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        }
    }
    
    # Additional analysis
    performance['variance_explained'] = performance['test']['r2'] * 100
    performance['avg_consumption'] = y_test.mean()
    performance['relative_error'] = performance['test']['mae'] / performance['avg_consumption'] * 100
    
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   Variance Explained: {performance['variance_explained']:.1f}%")
    print(f"   Test R²: {performance['test']['r2']:.4f}")
    print(f"   Test MAE: {performance['test']['mae']:.4f} kWh")
    print(f"   Test RMSE: {performance['test']['rmse']:.4f} kWh")
    print(f"   Test MAPE: {performance['test']['mape']:.1f}%")
    print(f"   Relative Error: {performance['relative_error']:.1f}%")
    
    # Return additional data for other functions
    performance['label_encoders'] = label_encoders
    performance['X_train'] = X_train
    performance['X_test'] = X_test
    performance['y_train'] = y_train
    performance['y_test'] = y_test
    
    return performance

def get_global_driver_ranking(shap_dict: dict, top_k: int = 20) -> pd.DataFrame:
    """
    Get global ranking of consumption drivers via SHAP values
    
    Args:
        shap_dict: Results from calculate_shap_values
        top_k: Number of top drivers to return
        
    Returns:
        DataFrame with top consumption drivers ranked by importance
    """
    print(f"🏆 RANKING TOP {top_k} CONSUMPTION DRIVERS")
    print("=" * 40)
    
    shap_values = shap_dict['shap_values']
    feature_names = shap_dict['feature_names']
    
    # Calculate global feature importance (mean absolute SHAP value)
    global_importance = np.abs(shap_values).mean(axis=0)
    
    # Create ranking dataframe
    driver_ranking = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': global_importance,
        'rank': range(1, len(feature_names) + 1)
    }).sort_values('shap_importance', ascending=False)
    
    # Reset rank after sorting
    driver_ranking['rank'] = range(1, len(driver_ranking) + 1)
    
    # Add percentage contribution
    total_importance = driver_ranking['shap_importance'].sum()
    driver_ranking['contribution_pct'] = (driver_ranking['shap_importance'] / total_importance) * 100
    
    # Get top drivers
    top_drivers = driver_ranking.head(top_k)
    
    print(f"📊 TOP {top_k} CONSUMPTION DRIVERS:")
    print("-" * 50)
    for _, row in top_drivers.iterrows():
        print(f"   {row['rank']:2d}. {row['feature']:<25} | SHAP: {row['shap_importance']:.4f} | {row['contribution_pct']:.1f}%")
    
    print(f"\n✅ Top {top_k} drivers explain {top_drivers['contribution_pct'].sum():.1f}% of model decisions")
    
    return driver_ranking

def analyze_feature_groups(shap_dict: dict, data_dict: dict) -> pd.DataFrame:
    """
    Analyze consumption drivers by feature groups
    
    Args:
        shap_dict: Results from calculate_shap_values
        data_dict: Results from prepare_modeling_data
        
    Returns:
        DataFrame with group-level importance analysis
    """
    print("📊 ANALYZING CONSUMPTION DRIVERS BY FEATURE GROUPS")
    print("=" * 51)
    
    shap_values = shap_dict['shap_values']
    feature_names = shap_dict['feature_names']
    feature_groups = data_dict['feature_groups']
    
    group_analysis = []
    
    for group_name, group_features in feature_groups.items():
        # Find features that exist in SHAP analysis
        existing_features = [f for f in group_features if f in feature_names]
        
        if existing_features:
            # Get SHAP values for this group
            feature_indices = [feature_names.index(f) for f in existing_features]
            group_shap = shap_values[:, feature_indices]
            
            # Calculate group metrics
            group_importance = np.abs(group_shap).mean()
            total_importance = np.abs(shap_values).mean()
            
            group_analysis.append({
                'group': group_name,
                'importance': group_importance,
                'contribution_pct': (group_importance / total_importance) * 100,
                'feature_count': len(existing_features),
                'avg_per_feature': group_importance / len(existing_features),
                'features': existing_features
            })
    
    # Convert to dataframe and sort
    group_df = pd.DataFrame(group_analysis)
    group_df = group_df.sort_values('importance', ascending=False)
    group_df['rank'] = range(1, len(group_df) + 1)
    
    print(f"📊 FEATURE GROUP IMPORTANCE RANKING:")
    print("-" * 60)
    for _, row in group_df.iterrows():
        print(f"   {row['rank']}. {row['group']:<20} | {row['contribution_pct']:5.1f}% | {row['feature_count']:2d} features")
    
    # Show most important group details
    top_group = group_df.iloc[0]
    print(f"\n🏆 MOST IMPORTANT GROUP: {top_group['group'].upper()}")
    print(f"   Explains {top_group['contribution_pct']:.1f}% of consumption variance")
    print(f"   Contains {top_group['feature_count']} features")
    print(f"   Top features: {', '.join(top_group['features'][:3])}")
    
    return group_df

def get_local_household_insights(shap_dict: dict, data_dict: dict, 
                               household_samples: int = 3) -> dict:
    """
    Generate local household-level consumption insights
    
    Args:
        shap_dict: Results from calculate_shap_values
        data_dict: Results from prepare_modeling_data
        household_samples: Number of households to analyze
        
    Returns:
        Dictionary with household-level insights
    """
    print(f"🏠 GENERATING LOCAL HOUSEHOLD INSIGHTS ({household_samples} examples)")
    print("=" * 55)
    
    shap_values = shap_dict['shap_values']
    X_shap = shap_dict['X_shap']
    expected_value = shap_dict['expected_value']
    
    # Select diverse households for analysis
    consumption_levels = X_shap.index
    
    # High, medium, low consumption examples
    if len(consumption_levels) >= household_samples:
        sample_indices = np.linspace(0, len(consumption_levels)-1, household_samples, dtype=int)
        sample_households = consumption_levels[sample_indices]
    else:
        sample_households = consumption_levels[:household_samples]
    
    household_insights = {}
    
    for i, household_idx in enumerate(sample_households):
        household_pos = X_shap.index.get_loc(household_idx)
        household_shap = shap_values[household_pos]
        household_features = X_shap.iloc[household_pos]
        
        # Get top positive and negative contributors
        feature_contributions = pd.DataFrame({
            'feature': X_shap.columns,
            'shap_value': household_shap,
            'feature_value': household_features.values
        })
        
        top_positive = feature_contributions.nlargest(5, 'shap_value')
        top_negative = feature_contributions.nsmallest(5, 'shap_value')
        
        predicted_consumption = expected_value + household_shap.sum()
        
        household_insights[f'household_{i+1}'] = {
            'index': household_idx,
            'predicted_consumption': predicted_consumption,
            'top_positive_drivers': top_positive,
            'top_negative_drivers': top_negative,
            'total_shap_impact': household_shap.sum()
        }
        
        print(f"\n🏠 HOUSEHOLD {i+1} (Index: {household_idx}):")
        print(f"   Predicted consumption: {predicted_consumption:.2f} kWh")
        print(f"   Top positive drivers:")
        for _, row in top_positive.head(3).iterrows():
            print(f"     {row['feature']:<20}: +{row['shap_value']:6.3f} (value: {row['feature_value']:6.2f})")
        
        print(f"   Top negative drivers:")
        for _, row in top_negative.head(3).iterrows():
            print(f"     {row['feature']:<20}: {row['shap_value']:7.3f} (value: {row['feature_value']:6.2f})")
    
    return household_insights

def generate_consumption_driver_report(stage0_results: dict) -> dict:
    """
    Generate comprehensive consumption driver analysis report
    
    Args:
        stage0_results: Complete Stage 0 modeling results
        
    Returns:
        Dictionary with comprehensive analysis report
    """
    print("📋 GENERATING COMPREHENSIVE CONSUMPTION DRIVER REPORT")
    print("=" * 54)
    
    # Extract components
    data_dict = stage0_results['data']
    model_dict = stage0_results['model']
    shap_dict = stage0_results['shap']
    
    # Run all analyses
    print("\n" + "="*15 + " PERFORMANCE EVALUATION " + "="*15)
    performance = evaluate_model_performance(model_dict, data_dict)
    
    print("\n" + "="*15 + " GLOBAL DRIVER RANKING " + "="*15)
    driver_ranking = get_global_driver_ranking(shap_dict, top_k=15)
    
    print("\n" + "="*15 + " FEATURE GROUP ANALYSIS " + "="*15)
    group_analysis = analyze_feature_groups(shap_dict, data_dict)
    
    print("\n" + "="*15 + " LOCAL INSIGHTS " + "="*15)
    household_insights = get_local_household_insights(shap_dict, data_dict, household_samples=3)
    
    # Create comprehensive report
    report = {
        'summary': {
            'model_performance': performance,
            'variance_explained': performance['variance_explained'],
            'top_driver': driver_ranking.iloc[0]['feature'],
            'most_important_group': group_analysis.iloc[0]['group'],
            'total_features': len(data_dict['feature_cols']),
            'feature_groups': len(data_dict['feature_groups']),
            'model_quality': 'EXCELLENT' if performance['test']['r2'] > 0.95 else 'GOOD' if performance['test']['r2'] > 0.80 else 'FAIR'
        },
        'driver_ranking': driver_ranking,
        'group_analysis': group_analysis,
        'household_insights': household_insights,
        'performance': performance
    }
    
    # Print executive summary
    print("\n🎉 CONSUMPTION DRIVER ANALYSIS COMPLETED!")
    print("=" * 46)
    print("📊 EXECUTIVE SUMMARY:")
    print(f"   Model Quality: {report['summary']['model_quality']}")
    print(f"   Variance Explained: {performance['variance_explained']:.1f}%")
    print(f"   Test RMSE: {performance['test']['rmse']:.3f} kWh")
    print(f"   Test R²: {performance['test']['r2']:.4f}")
    print(f"   Top driver: {driver_ranking.iloc[0]['feature']}")
    print(f"   Most important group: {group_analysis.iloc[0]['group']} ({group_analysis.iloc[0]['contribution_pct']:.1f}%)")
    
    return report

if __name__ == "__main__":
    print("📊 Feature Analysis - Stage 0: Consumption Driver Analysis & Evaluation")
    print("Usage: from src.evaluation.feature_analysis import generate_consumption_driver_report") 