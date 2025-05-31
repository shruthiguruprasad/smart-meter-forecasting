"""
📊 FEATURE PLOTS - Stage 0: SHAP & Feature Importance Visualizations
===================================================================

Stage 0: Visualization for Consumption Driver Analysis
- SHAP summary plots and dependence plots
- Feature group importance charts
- Local household waterfall plots
- Interactive dashboards for consumption drivers

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_top_drivers(driver_ranking: pd.DataFrame, 
                    top_k: int = 15,
                    figsize: tuple = (12, 8),
                    save_path: str = None) -> plt.Figure:
    """
    Plot top consumption drivers from SHAP analysis
    
    Args:
        driver_ranking: DataFrame from get_global_driver_ranking
        top_k: Number of top drivers to plot
        figsize: Figure size tuple
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    print(f"📊 Creating top {top_k} drivers visualization...")
    
    # Get top drivers
    top_drivers = driver_ranking.head(top_k)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_drivers)), top_drivers['shap_importance'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_drivers))))
    
    # Customize plot
    ax.set_yticks(range(len(top_drivers)))
    ax.set_yticklabels(top_drivers['feature'], fontsize=10)
    ax.set_xlabel('SHAP Importance (Mean |SHAP Value|)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Electricity Consumption Drivers\n(SHAP Feature Importance)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, value, pct) in enumerate(zip(bars, top_drivers['shap_importance'], top_drivers['contribution_pct'])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f} ({pct:.1f}%)', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show most important at top
    ax.invert_yaxis()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Add annotation
    total_contribution = top_drivers['contribution_pct'].sum()
    ax.text(0.98, 0.02, f'Top {top_k} features explain {total_contribution:.1f}% of model decisions',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

def plot_feature_group_importance(group_analysis: pd.DataFrame,
                                 figsize: tuple = (10, 6),
                                 save_path: str = None) -> plt.Figure:
    """
    Plot feature group importance analysis
    
    Args:
        group_analysis: DataFrame from analyze_feature_groups
        figsize: Figure size tuple
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    print("📊 Creating feature group importance visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(range(len(group_analysis)), group_analysis['contribution_pct'], 
                  color=plt.cm.Set3(np.linspace(0, 1, len(group_analysis))))
    
    # Customize plot
    ax.set_xticks(range(len(group_analysis)))
    ax.set_xticklabels([g.replace('_', ' ').title() for g in group_analysis['group']], 
                       rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Contribution to Consumption Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Group Importance\n(Consumption Driver Categories)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar, value, count in zip(bars, group_analysis['contribution_pct'], group_analysis['feature_count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%\n({count} features)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis limit with some padding
    ax.set_ylim(0, group_analysis['contribution_pct'].max() * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

def plot_shap_dependence_interactions(shap_dict: dict, 
                                     data_dict: dict,
                                     key_features: list = None,
                                     figsize: tuple = (15, 10),
                                     save_path: str = None) -> plt.Figure:
    """
    Plot SHAP dependence plots for key features with interactions
    
    Args:
        shap_dict: Results from calculate_shap_values
        data_dict: Results from prepare_modeling_data
        key_features: List of features to plot (auto-detect if None)
        figsize: Figure size tuple
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    print("📊 Creating SHAP dependence interaction plots...")
    
    shap_values = shap_dict['shap_values']
    X_shap = shap_dict['X_shap']
    
    # Auto-detect key features if not provided
    if key_features is None:
        # Get top features by SHAP importance
        global_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(global_importance)[-4:]  # Top 4 features
        key_features = [X_shap.columns[i] for i in top_indices]
    
    # Define interaction features for coloring
    interaction_map = {
        'heating_degree_days': 'is_weekend',
        'cooling_degree_days': 'is_weekend', 
        'lag7_total': 'is_holiday',
        'temp_avg': 'is_weekend',
        'peak_kwh': 'dayofweek',
        'total_kwh': 'is_weekend'
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    for i, feature in enumerate(key_features[:4]):
        ax = axes[i]
        
        # Get interaction feature for coloring
        interaction_feature = interaction_map.get(feature, None)
        if interaction_feature and interaction_feature in X_shap.columns:
            interaction_values = X_shap[interaction_feature]
        else:
            interaction_values = None
        
        # Plot SHAP dependence
        if interaction_values is not None:
            shap.plots.partial_dependence(
                feature, model=None, X=X_shap, ice=False,
                model_expected_value=True, feature_expected_value=True,
                ax=ax, show=False
            )
        else:
            # Simple scatter plot if no interaction
            feature_values = X_shap[feature]
            feature_idx = list(X_shap.columns).index(feature)
            feature_shap = shap_values[:, feature_idx]
            
            ax.scatter(feature_values, feature_shap, alpha=0.6, s=10)
            ax.set_xlabel(feature, fontweight='bold')
            ax.set_ylabel('SHAP value', fontweight='bold')
        
        ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('SHAP Dependence Plots\n(Key Feature Interactions)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

def plot_shap_waterfall_local(household_insights: dict,
                             figsize: tuple = (12, 8),
                             save_path: str = None) -> plt.Figure:
    """
    Plot SHAP waterfall plots for local household examples
    
    Args:
        household_insights: Results from get_local_household_insights
        figsize: Figure size tuple
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    print("📊 Creating local household SHAP waterfall plots...")
    
    n_households = len(household_insights)
    fig, axes = plt.subplots(1, n_households, figsize=(figsize[0], figsize[1]))
    
    if n_households == 1:
        axes = [axes]
    
    for i, (household_key, insights) in enumerate(household_insights.items()):
        ax = axes[i]
        
        # Get top contributors (positive and negative)
        top_positive = insights['top_positive_drivers'].head(3)
        top_negative = insights['top_negative_drivers'].head(3)
        
        # Combine and sort by absolute value
        contributors = pd.concat([top_positive, top_negative]).sort_values('shap_value', key=abs, ascending=True)
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'green' for x in contributors['shap_value']]
        bars = ax.barh(range(len(contributors)), contributors['shap_value'], color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(range(len(contributors)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in contributors['feature']], fontsize=9)
        ax.set_xlabel('SHAP Value (kWh)', fontweight='bold')
        ax.set_title(f'Household {i+1}\nPredicted: {insights["predicted_consumption"]:.1f} kWh', 
                     fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, contributors['shap_value']):
            label_x = value + (0.01 if value >= 0 else -0.01)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                    ha='left' if value >= 0 else 'right', va='center', fontsize=8, fontweight='bold')
        
        # Add baseline line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Local Household Consumption Drivers\n(SHAP Waterfall Analysis)', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

def plot_partial_dependence(model_dict: dict, data_dict: dict,
                           top_features: list = None,
                           figsize: tuple = (12, 8),
                           save_path: str = None) -> plt.Figure:
    """
    Plot partial dependence curves for top features
    
    Args:
        model_dict: Results from train_xgboost_model
        data_dict: Results from prepare_modeling_data
        top_features: List of features to plot (auto-detect if None)
        figsize: Figure size tuple
        save_path: Path to save plot (optional)
        
    Returns:
        Matplotlib figure object
    """
    print("📊 Creating partial dependence plots...")
    
    model = model_dict['model']
    X_train = data_dict['X_train']
    
    # Auto-detect top features if not provided
    if top_features is None:
        # Get feature importance from XGBoost
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-6:]  # Top 6 features
        top_features = [X_train.columns[i] for i in top_indices]
    
    # Create partial dependence display
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate partial dependence
    disp = PartialDependenceDisplay.from_estimator(
        model, X_train, features=top_features[:6], ax=ax,
        n_cols=3, n_jobs=-1, random_state=42
    )
    
    # Customize the plot
    fig.suptitle('Partial Dependence Plots\n(Average Effect of Features on Consumption)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add grid to each subplot
    for axis in disp.axes_.ravel():
        axis.grid(True, alpha=0.3)
        axis.set_xlabel(axis.get_xlabel(), fontweight='bold')
        axis.set_ylabel('Partial Dependence', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig

def create_driver_analysis_dashboard(analysis_report: dict,
                                   save_dir: str = "plots/",
                                   show_plots: bool = True) -> dict:
    """
    Create comprehensive dashboard of consumption driver visualizations
    
    Args:
        analysis_report: Results from generate_consumption_driver_report
        save_dir: Directory to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with all created figure objects
    """
    print("🎨 CREATING COMPREHENSIVE CONSUMPTION DRIVER DASHBOARD")
    print("=" * 56)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # 1. Top drivers plot
    print("📊 Creating top drivers visualization...")
    figures['top_drivers'] = plot_top_drivers(
        analysis_report['driver_ranking'], 
        top_k=15,
        save_path=f"{save_dir}/top_drivers.png"
    )
    
    # 2. Feature group importance
    print("📊 Creating feature group importance...")
    figures['group_importance'] = plot_feature_group_importance(
        analysis_report['group_analysis'],
        save_path=f"{save_dir}/group_importance.png"
    )
    
    # 3. Local household insights
    print("📊 Creating household waterfall plots...")
    figures['household_waterfalls'] = plot_shap_waterfall_local(
        analysis_report['household_insights'],
        save_path=f"{save_dir}/household_waterfalls.png"
    )
    
    print(f"\n🎉 DASHBOARD CREATION COMPLETED!")
    print(f"✅ {len(figures)} visualizations created")
    print(f"✅ Plots saved to: {save_dir}")
    
    if show_plots:
        plt.show()
    
    return figures

if __name__ == "__main__":
    print("📊 Feature Plots - Stage 0: SHAP & Feature Importance Visualizations")
    print("Usage: from src.visualization.feature_plots import create_driver_analysis_dashboard") 