import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .core import prepare_plotting_features
from .temporal import plot_cluster_switching_analysis, plot_cluster_timelines

def plot_cluster_stability_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze the stability of cluster assignments over time for each household
    Computes stability scores and visualizes relationships with socio-economic factors
    """
    print("🔍 Running Cluster Stability Analysis...")
    print("=" * 60)
    
    # Prepare features if needed
    df = df_hh
    
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
