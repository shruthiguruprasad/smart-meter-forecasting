import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import prepare_plotting_features

def plot_cluster_evolution(df_hh, cluster_col="cluster"):
    """
    Plot the evolution of cluster distributions over time
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    plt.figure(figsize=(14, 6))
    df["day"] = pd.to_datetime(df["day"])
    df["month"] = df["day"].dt.to_period("M")
    monthly_counts = df[df[cluster_col].notna()].groupby(
        ["month", cluster_col]).size().unstack(fill_value=0)
    monthly_share = monthly_counts.div(monthly_counts.sum(axis=1), axis=0)
    monthly_share.plot(kind="bar", stacked=True, figsize=(14, 5), colormap="Set2")
    plt.title("Monthly Distribution of Clusters (Behavioral Archetypes Over Time)")
    plt.xlabel("Month")
    plt.ylabel("Share of Active Household-Days")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

def plot_pattern_evolution_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the evolution of consumption patterns over time
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    # Check if we have time-based data
    if "day" not in df.columns:
        print("⚠️ Day column not found. Unable to perform pattern evolution analysis.")
        return None
    
    # Ensure day is datetime
    df["day"] = pd.to_datetime(df["day"])
    
    # Create month and year columns
    df["month"] = df["day"].dt.month
    df["year"] = df["day"].dt.year
    df["yearmonth"] = df["day"].dt.to_period("M")
    
    # Filter data
    df_evol = df[df[cluster_col].notna()].copy()
    df_evol[cluster_col] = df_evol[cluster_col].astype(int)
    
    # 1. Monthly cluster distribution
    plot_cluster_evolution(df_hh, cluster_col)
    
    # 2. Average consumption trends by cluster
    plt.figure(figsize=(14, 6))
    monthly_consumption = df_evol.groupby(["yearmonth", cluster_col])["total_kwh"].mean().unstack()
    monthly_consumption.plot(figsize=(14, 6), marker='o')
    plt.title("Monthly Average Consumption by Cluster")
    plt.xlabel("Month")
    plt.ylabel("Average Daily kWh")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    
    # 3. Seasonal patterns by cluster over time
    if "season" in df_evol.columns:
        plt.figure(figsize=(14, 8))
        seasonal_consumption = df_evol.groupby(["year", "season", cluster_col])["total_kwh"].mean().unstack()
        
        for year in df_evol["year"].unique():
            if year in seasonal_consumption.index.get_level_values(0):
                plt.subplot(1, len(df_evol["year"].unique()), 
                           list(df_evol["year"].unique()).index(year) + 1)
                
                year_data = seasonal_consumption.loc[year]
                year_data.plot(kind="bar", colormap="viridis")
                plt.title(f"Seasonal Consumption Patterns {year}")
                plt.xlabel("Season")
                plt.ylabel("Average Daily kWh")
                plt.legend(title="Cluster")
        
        plt.tight_layout()
        plt.show()
    
    # 4. Cluster transition heatmap
    if len(df_evol["LCLid"].unique()) > 1:  # Only if we have multiple households
        # Track transitions between months
        transitions = []
        
        for lclid in df_evol["LCLid"].unique():
            user_data = df_evol[df_evol["LCLid"] == lclid].sort_values("day")
            
            # Skip if we have only one record
            if len(user_data) <= 1:
                continue
                
            # Get pairs of consecutive cluster assignments
            prev_clusters = user_data[cluster_col].values[:-1]
            next_clusters = user_data[cluster_col].values[1:]
            
            for prev, next_c in zip(prev_clusters, next_clusters):
                transitions.append((prev, next_c))
        
        if transitions:
            # Create transition matrix
            unique_clusters = sorted(df_evol[cluster_col].unique())
            transition_matrix = pd.DataFrame(0, 
                                           index=unique_clusters, 
                                           columns=unique_clusters)
            
            for prev, next_c in transitions:
                transition_matrix.loc[prev, next_c] += 1
            
            # Convert to percentages (row-wise)
            transition_pct = transition_matrix.div(transition_matrix.sum(axis=1), axis=0) * 100
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_pct, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.title("Cluster Transition Probabilities (%)")
            plt.xlabel("Next Cluster")
            plt.ylabel("Previous Cluster")
            plt.tight_layout()
            plt.show()
    
    print("✅ Pattern Evolution Analysis Complete!")
    return df_evol
