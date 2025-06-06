import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .core import prepare_plotting_features

def plot_cluster_switching_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and plot cluster switching behavior
    Returns the switch counts for use in timeline plotting
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
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
    df = prepare_plotting_features(df_hh)
    
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

def plot_cluster_stability_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze the stability of cluster assignments over time for each household
    Computes stability scores and visualizes relationships with socio-economic factors
    """
    print("🔍 Running Cluster Stability Analysis...")
    print("=" * 60)
    
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    # Filter and prepare data
    df_stability = df[["LCLid", "day", cluster_col]].dropna()
    df_stability["day"] = pd.to_datetime(df_stability["day"])
    df_stability = df_stability.sort_values(["LCLid", "day"])
    
    # Add month column for temporal analysis
    df_stability["month"] = df_stability["day"].dt.to_period("M")
    
    # Calculate stability metrics
    stability_data = {}
    
    for lclid, user_df in df_stability.groupby("LCLid"):
        # Skip users with only one observation
        if len(user_df) <= 1:
            continue
        
        # Count the number of months with data
        month_count = user_df["month"].nunique()
        
        # Calculate dominant cluster and its proportion
        cluster_counts = user_df[cluster_col].value_counts()
        dominant_cluster = cluster_counts.idxmax()
        dominant_proportion = cluster_counts.max() / len(user_df)
        
        # Calculate entropy of the cluster distribution (lower is more stable)
        probs = cluster_counts / len(user_df)
        entropy = -(probs * np.log2(probs)).sum()
        
        # Calculate normalized entropy (0 to 1, where 1 is maximally unstable)
        max_entropy = np.log2(cluster_counts.count())  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate stability score (0 to 1, where 1 is perfectly stable)
        stability_score = 1 - normalized_entropy
        
        # Store metrics
        stability_data[lclid] = {
            "dominant_cluster": dominant_cluster,
            "dominant_proportion": dominant_proportion,
            "entropy": entropy,
            "stability_score": stability_score,
            "month_count": month_count
        }
    
    # Convert to DataFrame
    stability_df = pd.DataFrame.from_dict(stability_data, orient="index")
    
    # 1. Distribution of stability scores
    plt.figure(figsize=(10, 6))
    sns.histplot(stability_df["stability_score"], bins=20, kde=True)
    plt.axvline(stability_df["stability_score"].mean(), color='r', linestyle='--', 
               label=f"Mean: {stability_df['stability_score'].mean():.2f}")
    plt.axvline(stability_df["stability_score"].median(), color='g', linestyle='--', 
               label=f"Median: {stability_df['stability_score'].median():.2f}")
    plt.title("Distribution of Cluster Stability Scores")
    plt.xlabel("Stability Score (0=unstable, 1=stable)")
    plt.ylabel("Count of Households")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Relationship between observation period and stability
    plt.figure(figsize=(10, 6))
    plt.scatter(stability_df["month_count"], stability_df["stability_score"], alpha=0.3)
    plt.title("Relationship Between Observation Period and Stability")
    plt.xlabel("Number of Months with Data")
    plt.ylabel("Stability Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Stability by dominant cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=stability_df, x="dominant_cluster", y="stability_score")
    plt.title("Stability Score by Dominant Cluster")
    plt.xlabel("Dominant Cluster")
    plt.ylabel("Stability Score")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 4. Socioeconomic analysis of stability if data available
    if "Acorn_grouped" in df.columns:
        acorn_data = df[["LCLid", "Acorn_grouped"]].drop_duplicates().set_index("LCLid")
        stability_acorn = stability_df.join(acorn_data, how="inner")
        
        if not stability_acorn.empty and "Acorn_grouped" in stability_acorn.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=stability_acorn, x="Acorn_grouped", y="stability_score")
            plt.title("Stability Score by ACORN Group")
            plt.xlabel("ACORN Group")
            plt.ylabel("Stability Score")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Statistical tests between ACORN groups
            acorn_groups = stability_acorn["Acorn_grouped"].unique()
            if len(acorn_groups) >= 2:
                print("\nStatistical tests for stability differences between ACORN groups:")
                
                # Pick two largest groups for comparison
                group_counts = stability_acorn["Acorn_grouped"].value_counts()
                group_names = group_counts.index[:2]
                
                group1 = stability_acorn[stability_acorn["Acorn_grouped"] == group_names[0]]["stability_score"]
                group2 = stability_acorn[stability_acorn["Acorn_grouped"] == group_names[1]]["stability_score"]
                
                # Run t-test
                ttest_result = stats.ttest_ind(group1, group2, equal_var=False)
                print(f"T-test between {group_names[0]} and {group_names[1]}: p-value = {ttest_result.pvalue:.4f}")
                
                if ttest_result.pvalue < 0.05:
                    print(f"✅ Significant difference in stability between {group_names[0]} and {group_names[1]}")
                else:
                    print(f"❌ No significant difference in stability between {group_names[0]} and {group_names[1]}")
    
    # 5. Relationship between stability and consumption
    plt.figure(figsize=(10, 5))
    consumption_stability = stability_df.merge(
        df.groupby("LCLid")["total_kwh"].mean().reset_index(), 
        on="LCLid",
        how="left"
    )
    
    sns.scatterplot(
        data=consumption_stability, 
        x="total_kwh", 
        y="stability_score", 
        alpha=0.5,
        hue="month_count" if consumption_stability["month_count"].nunique() > 1 else None
    )
    plt.title("Cluster Stability vs. Average Consumption")
    plt.xlabel("Average Daily Consumption (kWh)")
    plt.ylabel("Stability Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return the stability scores for further analysis
    print("\n✅ Cluster Stability Analysis Complete!")
    return stability_df
