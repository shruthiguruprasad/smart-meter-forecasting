import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import prepare_plotting_features

def cluster_profile_summary(df_hh, cluster_col="cluster"):
    """
    Generate a summary profile for each cluster
    """
    def most_frequent(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    df_summary = df[(df[cluster_col].notna()) & (df["total_kwh"].notna())].copy()
    
    if "peak_hour" in df_summary.columns:
        df_summary["peak_hour"] = df_summary["peak_hour"].astype(float)
    df_summary[cluster_col] = df_summary[cluster_col].astype(int)

    # Define aggregation based on available columns
    agg_dict = {
        "total_kwh": ["mean", "std"]
    }
    
    # Add optional columns if they exist
    if "peak_hour" in df_summary.columns:
        agg_dict["peak_hour"] = ["mean", most_frequent]
    if "stdorToU" in df_summary.columns:
        agg_dict["stdorToU"] = [most_frequent]
    if "Acorn_grouped" in df_summary.columns:
        agg_dict["Acorn_grouped"] = [most_frequent]
    if "daily_variability" in df_summary.columns:
        agg_dict["daily_variability"] = ["mean"]

    cluster_profiles = df_summary.groupby(cluster_col).agg(agg_dict).round(2)
    
    print("\nCluster Profile Summary:\n")
    print(cluster_profiles)
    return cluster_profiles

def plot_acorn_distribution(df_hh, cluster_col="cluster"):
    """
    Plot the distribution of ACORN groups by cluster
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    if "Acorn_grouped" not in df.columns:
        print("⚠️ Acorn_grouped column not found in the dataset. Unable to plot ACORN distribution.")
        return None
    
    # Get unique households for each cluster
    df_acorn = df.drop_duplicates(subset=["LCLid"])[["LCLid", cluster_col, "Acorn_grouped"]]
    df_acorn = df_acorn.dropna(subset=[cluster_col, "Acorn_grouped"])
    df_acorn[cluster_col] = df_acorn[cluster_col].astype(int)
    
    # Calculate percentage distribution
    acorn_counts = df_acorn.groupby([cluster_col, "Acorn_grouped"]).size().unstack(fill_value=0)
    acorn_dist = acorn_counts.div(acorn_counts.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 6))
    acorn_dist.plot(kind="bar", stacked=True, colormap="tab20")
    plt.title("ACORN Group Distribution by Cluster")
    plt.ylabel("Percentage of Households")
    plt.xlabel("Cluster ID")
    plt.legend(title="ACORN Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return acorn_dist

def plot_socioeconomic_intervention_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze consumption patterns by socioeconomic group and tariff type
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    # Check if socioeconomic data is available
    if "Acorn_grouped" not in df.columns:
        print("⚠️ Acorn_grouped column not found. Unable to perform socioeconomic analysis.")
        return None
    
    # 1. Plot ACORN distribution by cluster
    acorn_dist = plot_acorn_distribution(df_hh, cluster_col)
    
    # 2. Consumption by ACORN group and cluster
    plt.figure(figsize=(12, 6))
    df_acorn = df.dropna(subset=[cluster_col, "Acorn_grouped", "total_kwh"])
    df_acorn[cluster_col] = df_acorn[cluster_col].astype(int)
    
    sns.boxplot(data=df_acorn, x="Acorn_grouped", y="total_kwh", hue=cluster_col)
    plt.title("Daily Consumption by ACORN Group and Cluster")
    plt.xlabel("ACORN Group")
    plt.ylabel("Total Daily kWh")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    
    # 3. Tariff impact analysis if available
    if "stdorToU" in df.columns:
        plt.figure(figsize=(12, 6))
        df_tariff = df.dropna(subset=[cluster_col, "stdorToU", "total_kwh"])
        df_tariff[cluster_col] = df_tariff[cluster_col].astype(int)
        
        sns.boxplot(data=df_tariff, x=cluster_col, y="total_kwh", hue="stdorToU")
        plt.title("Impact of Tariff Type on Consumption by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Total Daily kWh")
        plt.legend(title="Tariff Type")
        plt.tight_layout()
        plt.show()
        
        # 4. Time-of-day consumption by tariff type
        if all(f"hh_{i}" in df.columns for i in range(48)):
            plt.figure(figsize=(14, 6))
            
            hh_cols = [f"hh_{i}" for i in range(48)]
            
            for tariff in df_tariff["stdorToU"].unique():
                tariff_profile = df_tariff[df_tariff["stdorToU"] == tariff][hh_cols].mean()
                plt.plot(range(48), tariff_profile, label=tariff)
            
            # Add peak and off-peak periods if using ToU tariff
            plt.axvspan(0, 7*2, alpha=0.2, color='green', label='Off-Peak')
            plt.axvspan(14*2, 16*2, alpha=0.2, color='green')
            plt.axvspan(19*2, 48, alpha=0.2, color='green')
            plt.axvspan(16*2, 19*2, alpha=0.2, color='red', label='Peak')
            
            plt.title("Average Daily Load Profile by Tariff Type")
            plt.xlabel("Half-Hour of Day")
            plt.ylabel("Average kWh")
            plt.xticks(range(0, 48, 4), [f"{h}:00" for h in range(0, 24, 2)])
            plt.legend(title="Tariff Type")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    print("✅ Socioeconomic Intervention Analysis Complete!")
    return df_acorn
