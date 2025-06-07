import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .core import prepare_plotting_features, hh_cols

def plot_cluster_switching_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and plot cluster switching behavior
    Returns the switch counts for use in timeline plotting
    """
    # Prepare features if needed
    df = df_hh
    
    # Calculate switching statistics
    df_switch = df[["LCLid", "day", cluster_col]].dropna()
    df_switch["day"] = pd.to_datetime(df_switch["day"])
    df_switch = df_switch.sort_values(["LCLid", "day"])
    
    # Compute switches for each user
    switch_counts = {}
    
    for lclid, user_df in df_switch.groupby("LCLid"):
        if len(user_df) <= 1:
            continue
            
        user_df = user_df.sort_values("day")
        
        # Create shifted column to compare with previous cluster
        user_df["prev_cluster"] = user_df[cluster_col].shift(1)
        
        # Count switches (when current != previous)
        switches = (user_df[cluster_col] != user_df["prev_cluster"]).sum()
        total_days = len(user_df)
        
        # Only count users with at least 2 records
        if total_days > 1:
            switch_rate = switches / (total_days - 1)  # -1 because first day has no previous
            switch_counts[lclid] = {
                "switches": switches,
                "total_days": total_days,
                "switch_rate": switch_rate
            }
    
    # Convert to DataFrame for analysis
    switch_df = pd.DataFrame.from_dict(switch_counts, orient="index")
    
    # Plot switch rate distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(switch_df["switch_rate"], bins=20, kde=True)
    plt.axvline(switch_df["switch_rate"].mean(), color='r', linestyle='--', 
               label=f"Mean: {switch_df['switch_rate'].mean():.2f}")
    plt.axvline(switch_df["switch_rate"].median(), color='g', linestyle='--', 
               label=f"Median: {switch_df['switch_rate'].median():.2f}")
    plt.title("Distribution of Cluster Switching Rates")
    plt.xlabel("Switch Rate (Proportion of Days with Cluster Change)")
    plt.ylabel("Count of Households")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot relationship between total days and switches
    plt.figure(figsize=(10, 6))
    plt.scatter(switch_df["total_days"], switch_df["switches"], alpha=0.3)
    plt.title("Relationship Between Observation Period and Number of Switches")
    plt.xlabel("Total Days Observed")
    plt.ylabel("Number of Cluster Switches")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # If socioeconomic data available, analyze switch rates by ACORN group
    if "Acorn_grouped" in df.columns:
        acorn_switch = df[["LCLid", "Acorn_grouped"]].drop_duplicates().set_index("LCLid")
        switch_acorn = switch_df.join(acorn_switch, how="inner")
        
        if not switch_acorn.empty and "Acorn_grouped" in switch_acorn.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=switch_acorn, x="Acorn_grouped", y="switch_rate")
            plt.title("Cluster Switching Rate by ACORN Group")
            plt.xlabel("ACORN Group")
            plt.ylabel("Switch Rate")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    print(f"Average switch rate: {switch_df['switch_rate'].mean():.3f}")
    print(f"Median switch rate: {switch_df['switch_rate'].median():.3f}")
    print(f"Percentage of stable households (switch rate < 0.1): {(switch_df['switch_rate'] < 0.1).mean() * 100:.1f}%")
    print(f"Percentage of volatile households (switch rate > 0.5): {(switch_df['switch_rate'] > 0.5).mean() * 100:.1f}%")
    
    return switch_df

def plot_cluster_timelines(df_hh, switch_counts, top_n=5, cluster_col="cluster"):
    """
    Plot cluster assignment timelines for the most volatile households
    """
    # Prepare features if needed
    df = df_hh
    
    # Sort households by switch rate
    if isinstance(switch_counts, pd.DataFrame) and "switch_rate" in switch_counts.columns:
        volatile_users = switch_counts.sort_values("switch_rate", ascending=False).head(top_n).index.tolist()
    else:
        print("⚠️ Invalid switch_counts provided. Using random households instead.")
        volatile_users = np.random.choice(df["LCLid"].unique(), size=min(top_n, len(df["LCLid"].unique())), replace=False)
    
    # Plot timelines for each user
    for lclid in volatile_users:
        df_user = df[df["LCLid"] == lclid].sort_values("day")
        
        if len(df_user) <= 1:
            continue
        
        plt.figure(figsize=(12, 4))
        plt.plot(df_user["day"], df_user[cluster_col], 
                marker='o', linestyle='-', markersize=3)
        plt.title(f"Cluster Assignment Timeline — LCLid: {lclid}")
        plt.xlabel("Date")
        plt.ylabel("Cluster")
        plt.yticks(sorted(df_user[cluster_col].unique()))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

def plot_cluster_evolution(df_hh, cluster_col="cluster"):
    """
    Plot the evolution of cluster distributions over time
    """
    # Prepare features if needed
    df = df_hh
    
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

def plot_pattern_stability_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze the stability of consumption patterns over time
    Shows how patterns evolve month-by-month for each cluster
    """
    # Prepare features if needed
    df = df_hh
    
    # Check if we have time-based data
    if "day" not in df.columns:
        print("⚠️ Day column not found. Unable to perform pattern stability analysis.")
        return None
    
    # Ensure day is datetime
    df["day"] = pd.to_datetime(df["day"])
    
    # Create month and year columns
    df["month"] = df["day"].dt.month
    df["month_name"] = df["day"].dt.strftime('%b')
    
    # Filter data
    df_stab = df[df[cluster_col].notna()].copy()
    df_stab[cluster_col] = df_stab[cluster_col].astype(int)
    
    # Monthly average consumption by cluster
    plt.figure(figsize=(14, 7))
    
    # Calculate monthly averages
    monthly_avg = df_stab.groupby([cluster_col, "month"])["total_kwh"].mean().unstack()
    
    # Transpose for better visualization (clusters as columns, months as rows)
    monthly_avg_T = monthly_avg.T
    
    # Create month names for better x-axis labels
    month_names = {i: pd.Timestamp(2023, i, 1).strftime('%b') for i in range(1, 13)}
    monthly_avg_T.index = monthly_avg_T.index.map(month_names)
    
    # Plot each cluster's pattern stability
    for cluster in sorted(df_stab[cluster_col].unique()):
        if cluster in monthly_avg_T.columns:
            plt.plot(monthly_avg_T.index, monthly_avg_T[cluster], 
                   marker='o', label=f"Cluster {cluster}", linewidth=2)
    
    plt.title("Pattern Stability Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Average Daily kWh")
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Seasonal pattern variability
    if "season" in df_stab.columns:
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df_stab, x=cluster_col, y="daily_variability" if "daily_variability" in df_stab.columns 
                   else "total_kwh", hue="season")
        plt.title("Pattern Variability by Season", fontsize=14, fontweight='bold')
        plt.xlabel("Cluster")
        plt.ylabel("Daily Variability" if "daily_variability" in df_stab.columns else "Total Daily kWh")
        plt.legend(title="Season")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return df_stab

def plot_seasonal_load_signature_grid(df_hh, cluster_col="cluster"):
    """
    Plot a 2×2 grid of temporal evolution visualizations:
    - Seasonal Load Signature Evolution (daily kWh by cluster over months)
    - Seasonal Pattern Evolution by Cluster (half-hourly patterns by season)
    - Pattern Stability Over Time (consistency of patterns)
    - Pattern Variability by Season (daily variability measures)
    """
    # Prepare features if needed
    df = df_hh
    
    # Check for required columns
    if "day" not in df.columns:
        print("⚠️ Day column not found. Unable to perform seasonal analysis.")
        return None
    
    # Ensure day is datetime
    df["day"] = pd.to_datetime(df["day"])
    df["month"] = df["day"].dt.month
    df["month_name"] = df["day"].dt.strftime('%b')
    
    if "season" not in df.columns and "month" in df.columns:
        # Create season if it doesn't exist
        season_map = {
            1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
            5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"
        }
        df["season"] = df["month"].map(season_map)
    
    # Filter data
    df_plot = df[df[cluster_col].notna()].copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(int)
    
    # Create 2×2 grid layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Seasonal Load Signature Evolution (top-left)
    # Calculate monthly average by cluster
    monthly_data = df_plot.groupby([df_plot["day"].dt.to_period("M"), cluster_col])["total_kwh"].mean().unstack()
    
    # Plot monthly evolution
    for cluster in sorted(df_plot[cluster_col].unique()):
        if cluster in monthly_data.columns:
            ax1.plot(range(len(monthly_data.index)), monthly_data[cluster], 
                   marker='o', markersize=3, label=f"Cluster {cluster}", linewidth=2)
    
    # Format x-axis with month labels
    if len(monthly_data.index) > 0:
        month_labels = [str(idx).split("-")[1] for idx in monthly_data.index]
        ax1.set_xticks(range(len(month_labels)))
        
        # Show a subset of labels if there are many months
        if len(month_labels) > 12:
            step = len(month_labels) // 12 + 1
            ax1.set_xticklabels(month_labels[::step])
        else:
            ax1.set_xticklabels(month_labels)
            
    ax1.set_title("Seasonal Load Signature Evolution", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Daily kWh")
    ax1.legend(title="Cluster")
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal Pattern Evolution for a specific cluster (top-right)
    # Select a representative cluster (middle cluster or cluster 0)
    clusters = sorted(df_plot[cluster_col].unique())
    representative_cluster = clusters[len(clusters) // 2] if len(clusters) > 0 else 0
    
    # Get seasonal patterns for the representative cluster
    if "season" in df_plot.columns and all(col in df_plot.columns for col in hh_cols):
        cluster_data = df_plot[df_plot[cluster_col] == representative_cluster]
        
        for season in sorted(cluster_data["season"].unique()):
            season_data = cluster_data[cluster_data["season"] == season]
            if len(season_data) > 0:
                avg_pattern = season_data[hh_cols].mean().values
                ax2.plot(range(48), avg_pattern, label=season, linewidth=2)
        
        ax2.set_title(f"Seasonal Pattern Evolution - Cluster {representative_cluster}", 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel("Half-Hour Interval")
        ax2.set_ylabel("Average kWh")
        ax2.set_xticks(range(0, 48, 4))
        ax2.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
        ax2.legend(title="Season")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Seasonal data not available", ha='center', va='center')
        ax2.set_title("Seasonal Pattern Evolution", fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    # 3. Pattern Stability Over Time (bottom-left)
    # Calculate monthly averages for normalized comparison
    monthly_avg = df_plot.groupby([cluster_col, "month"])["total_kwh"].mean().unstack()
    
    # Plot each cluster's pattern stability
    for cluster in sorted(df_plot[cluster_col].unique()):
        if cluster in monthly_avg.index:
            monthly_values = monthly_avg.loc[cluster].values
            ax3.plot(range(1, len(monthly_values) + 1), monthly_values, 
                   marker='o', label=f"Cluster {cluster}", linewidth=2)
    
    ax3.set_title("Pattern Stability Over Time", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Average Daily kWh")
    ax3.set_xticks(range(1, 13))
    ax3.legend(title="Cluster")
    ax3.grid(True, alpha=0.3)
    
    # 4. Pattern Variability by Season (bottom-right)
    if "season" in df_plot.columns:
        if "daily_variability" not in df_plot.columns:
            # Calculate simple variability measure if not available
            df_plot["daily_variability"] = df_plot[hh_cols].apply(
                lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0, axis=1
            )
        
        sns.boxplot(data=df_plot, x=cluster_col, y="daily_variability", 
                   hue="season", ax=ax4, palette="viridis")
        
        ax4.set_title("Pattern Variability by Season", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Cluster")
        ax4.set_ylabel("Daily Variability")
        ax4.legend(title="Season")
        ax4.grid(True, axis='y', alpha=0.3)
    else:
        # Fallback - show variability by cluster only
        if "daily_variability" not in df_plot.columns:
            df_plot["daily_variability"] = df_plot[hh_cols].apply(
                lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0, axis=1
            )
        
        sns.boxplot(data=df_plot, x=cluster_col, y="daily_variability", ax=ax4, palette="viridis")
        ax4.set_title("Pattern Variability", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Cluster")
        ax4.set_ylabel("Daily Variability")
        ax4.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle("Temporal Pattern Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.93)
    plt.show()
    
    return df_plot

def plot_pattern_evolution_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the evolution of consumption patterns over time
    """
    # Prepare features if needed
    df = df_hh
    
    # Check if we have time-based data
    if "day" not in df.columns:
        print("⚠️ Day column not found. Unable to perform pattern evolution analysis.")
        return None
    
    # First display the comprehensive 2×2 grid analysis
    df_plot = plot_seasonal_load_signature_grid(df_hh, cluster_col)
    
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
    
    # 2. Cluster switching analysis
    switch_counts = plot_cluster_switching_analysis(df_hh, cluster_col)
    
    # 3. Timelines for most volatile households
    plot_cluster_timelines(df_hh, switch_counts, top_n=3, cluster_col=cluster_col)
    
    # 4. Average consumption trends by cluster
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
    
    # 5. Seasonal patterns by cluster over time
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
    
    # 6. Cluster transition heatmap
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
