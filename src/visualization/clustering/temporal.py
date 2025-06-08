import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
      # Removed cluster transition heatmap visualization to simplify output
    # The heatmap showed the probability of households transitioning between clusters
    
    print("✅ Pattern Evolution Analysis Complete!")
    return df_evol
