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
    df = df_hh
    
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
    Analyze consumption patterns by socioeconomic group and tariff type using a 2×2 grid layout
    
    This visualization explores:
    - ACORN Group Distribution by Cluster
    - Load Shifting Potential (Day vs Night consumption)
    - Consumption by ACORN group and cluster
    - Intervention Priority Matrix (consumption vs variability)
    """
    # Prepare features if needed
    df = df_hh
    
    # Check if socioeconomic data is available
    if "Acorn_grouped" not in df.columns:
        print("⚠️ Acorn_grouped column not found. Unable to perform socioeconomic analysis.")
        return None
    
    # Filter data for plotting
    df_plot = df[df[cluster_col].notna()].copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(int)
    
    # Create 2×2 grid layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. ACORN Group Distribution by Cluster (top-left)
    if "LCLid" in df_plot.columns:
        # Get unique households for each cluster
        df_acorn = df_plot.drop_duplicates(subset=["LCLid"])[["LCLid", cluster_col, "Acorn_grouped"]]
        df_acorn = df_acorn.dropna(subset=[cluster_col, "Acorn_grouped"])
    else:
        # Fallback to using all data points
        df_acorn = df_plot[["Acorn_grouped", cluster_col]].dropna()
    
    # Calculate percentage distribution
    acorn_counts = df_acorn.groupby([cluster_col, "Acorn_grouped"]).size().unstack(fill_value=0)
    acorn_dist = acorn_counts.div(acorn_counts.sum(axis=1), axis=0) * 100
    
    acorn_dist.plot(kind="bar", stacked=True, ax=ax1, colormap="tab20")
    ax1.set_title("Socio-Economic Distribution by Load Archetype", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Percentage of Households")
    ax1.legend(title="ACORN Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. Load Shifting Potential (top-right)
    hh_cols = [f"hh_{i}" for i in range(48)]
    
    if 'day_night_ratio' in df_plot.columns:
        sns.boxplot(data=df_plot, x=cluster_col, y='day_night_ratio', ax=ax2, palette="viridis")
    else:
        # Calculate basic day/night ratio
        day_cols = [f"hh_{i}" for i in range(12, 36)]  # 6 AM to 6 PM
        night_cols = [f"hh_{i}" for i in list(range(0, 12)) + list(range(36, 48))]  # 6 PM to 6 AM
        
        df_temp = df_plot.copy()
        df_temp['day_sum'] = df_temp[day_cols].sum(axis=1)
        df_temp['night_sum'] = df_temp[night_cols].sum(axis=1)
        # Replace zeros in night_sum with a small value to avoid division by zero
        df_temp.loc[df_temp['night_sum'] == 0, 'night_sum'] = 0.001
        df_temp['day_night_ratio'] = df_temp['day_sum'] / df_temp['night_sum']
        
        # Cap extreme values for better visualization
        upper_limit = df_temp['day_night_ratio'].quantile(0.95)
        df_temp.loc[df_temp['day_night_ratio'] > upper_limit, 'day_night_ratio'] = upper_limit
        
        sns.boxplot(data=df_temp, x=cluster_col, y='day_night_ratio', ax=ax2, palette="viridis")
    
    ax2.set_title('Load Shifting Potential by Archetype', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Day-Night Consumption Ratio')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Day/Night')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Consumption by ACORN group and cluster (bottom-left)
    if 'peak_to_mean_ratio' in df_plot.columns:
        sns.boxplot(data=df_plot, x='Acorn_grouped', y='peak_to_mean_ratio', 
                   hue=cluster_col, ax=ax3, palette="viridis")
        ax3.set_title('Peak Load Reduction Potential', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Peak to Mean Ratio')
    else:
        sns.boxplot(data=df_plot, x='Acorn_grouped', y='total_kwh', 
                   hue=cluster_col, ax=ax3, palette="viridis")
        ax3.set_title('Consumption Patterns by Socio-Economic Group', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Daily kWh')
    
    ax3.set_xlabel('ACORN Group')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title="Cluster")
    ax3.grid(True, alpha=0.3)
    
    # 4. Intervention Priority Matrix (bottom-right)
    # Calculate intervention scores by cluster
    if 'daily_variability' in df_plot.columns:
        variability_col = 'daily_variability'
    else:
        # Create a variability metric if not available
        df_plot['calculated_variability'] = df_plot[hh_cols].apply(lambda x: x.std() / x.mean() if x.mean() > 0 else 0, axis=1)
        variability_col = 'calculated_variability'
    
    cluster_stats = df_plot.groupby(cluster_col).agg({
        'total_kwh': 'mean',
        variability_col: 'mean'
    }).reset_index()
    
    scatter = ax4.scatter(
        cluster_stats['total_kwh'], 
        cluster_stats[variability_col], 
        c=cluster_stats[cluster_col], 
        s=200, 
        alpha=0.7, 
        cmap='viridis'
    )
    
    # Add cluster labels to points
    for i, row in cluster_stats.iterrows():
        ax4.annotate(
            f"Cluster {int(row[cluster_col])}", 
            (row['total_kwh'], row[variability_col]),
            xytext=(7, 7), 
            textcoords='offset points',
            fontweight='bold'
        )
    
    # Add quadrant labels and dividing lines
    median_kwh = cluster_stats['total_kwh'].median()
    median_var = cluster_stats[variability_col].median()
    
    ax4.axhline(y=median_var, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=median_kwh, color='red', linestyle='--', alpha=0.5)
    
    # Add quadrant annotations
    ax4.text(
        cluster_stats['total_kwh'].max() * 0.9, 
        cluster_stats[variability_col].max() * 0.9,
        "High Consumption\nHigh Variability\n(Peak Reduction)", 
        ha='center', va='center', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    
    ax4.text(
        cluster_stats['total_kwh'].min() * 1.1, 
        cluster_stats[variability_col].max() * 0.9,
        "Low Consumption\nHigh Variability\n(Load Stabilization)", 
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    
    ax4.set_title('Intervention Priority Matrix', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Average Daily Consumption (kWh)')
    ax4.set_ylabel('Consumption Variability')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle("Socioeconomic Intervention & Optimization Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.93)
    plt.show()
    
    # If tariff analysis is available, show as a separate visualization
    if "stdorToU" in df.columns:
        plot_tariff_impact_analysis(df, cluster_col)
    
    print("✅ Socioeconomic Intervention Analysis Complete!")
    return df_acorn

def plot_tariff_impact_analysis(df, cluster_col="cluster"):
    """
    Analyze the impact of different tariff types on consumption patterns
    
    This visualization shows:
    - Impact of tariff type on consumption by cluster
    - Time-of-day consumption patterns by tariff type with peak/off-peak periods
    """
    if "stdorToU" not in df.columns:
        print("⚠️ Tariff information (stdorToU) not found. Unable to perform tariff analysis.")
        return None
    
    # Filter data for plotting
    df_tariff = df.dropna(subset=[cluster_col, "stdorToU", "total_kwh"]).copy()
    df_tariff[cluster_col] = df_tariff[cluster_col].astype(int)
    
    # 1. Tariff impact on consumption by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_tariff, x=cluster_col, y="total_kwh", hue="stdorToU", palette="Set2")
    plt.title("Impact of Tariff Type on Consumption by Cluster", fontsize=14, fontweight='bold')
    plt.xlabel("Cluster")
    plt.ylabel("Total Daily kWh")
    plt.legend(title="Tariff Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Time-of-day consumption by tariff type
    hh_cols = [f"hh_{i}" for i in range(48)]
    if all(col in df_tariff.columns for col in hh_cols):
        plt.figure(figsize=(14, 7))
        
        for tariff in df_tariff["stdorToU"].unique():
            tariff_profile = df_tariff[df_tariff["stdorToU"] == tariff][hh_cols].mean()
            plt.plot(range(48), tariff_profile, label=tariff, linewidth=2)
        
        # Add peak and off-peak periods if using ToU tariff
        plt.axvspan(0, 7*2, alpha=0.2, color='green', label='Off-Peak')
        plt.axvspan(14*2, 16*2, alpha=0.2, color='green')
        plt.axvspan(19*2, 48, alpha=0.2, color='green')
        plt.axvspan(16*2, 19*2, alpha=0.2, color='red', label='Peak')
        
        plt.title("Average Daily Load Profile by Tariff Type", fontsize=14, fontweight='bold')
        plt.xlabel("Half-Hour of Day")
        plt.ylabel("Average kWh")
        plt.xticks(range(0, 48, 4), [f"{h}:00" for h in range(0, 24, 2)])
        plt.legend(title="Tariff Type")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return df_tariff
