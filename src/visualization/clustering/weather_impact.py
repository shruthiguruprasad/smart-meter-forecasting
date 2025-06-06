import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import prepare_plotting_features

def plot_hdd_vs_kwh(df_hh, cluster_col="cluster"):
    """
    Plot the relationship between heating degree days and consumption by cluster
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    df_weather = df[(df["total_kwh"].notna()) & 
                    (df["heating_degree_days"].notna()) & 
                    (df[cluster_col].notna())]
    df_weather[cluster_col] = df_weather[cluster_col].astype(int)
    
    sns.lmplot(
        data=df_weather.sample(n=min(100_000, len(df_weather)), random_state=42),
        x="heating_degree_days", y="total_kwh",
        hue=cluster_col,
        scatter_kws={"alpha": 0.05},
        line_kws={"linewidth": 2},
        height=5, aspect=1.5
    )
    plt.title("Cluster-wise Sensitivity to Heating Degree Days (HDD)")
    plt.xlabel("HDD (Base 18°C)")
    plt.ylabel("Total Daily kWh")
    plt.tight_layout()
    plt.show()
    
def plot_weather_impact_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the impact of weather on different clusters
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    # Filter data
    df_weather = df[(df["total_kwh"].notna()) & 
                    (df["heating_degree_days"].notna()) & 
                    (df[cluster_col].notna())]
    df_weather[cluster_col] = df_weather[cluster_col].astype(int)
    
    # 1. Temperature correlation by cluster
    plt.figure(figsize=(10, 6))
    
    # Compute correlations between HDD and consumption for each cluster
    corr_by_cluster = df_weather.groupby(cluster_col).apply(
        lambda x: x[["heating_degree_days", "total_kwh"]].corr().iloc[0, 1]
    ).sort_values(ascending=False)
    
    sns.barplot(x=corr_by_cluster.index, y=corr_by_cluster.values)
    plt.title("Temperature Sensitivity by Cluster (Correlation with HDD)")
    plt.xlabel("Cluster")
    plt.ylabel("Correlation Coefficient")
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. HDD vs kWh plot
    plot_hdd_vs_kwh(df_hh, cluster_col)
    
    # 3. Consumption boxplot by temperature range and cluster
    plt.figure(figsize=(12, 6))
    df_weather["temp_range"] = pd.cut(df_weather["heating_degree_days"], 
                                     bins=5, 
                                     labels=["Very Warm", "Warm", "Moderate", "Cool", "Cold"])
    
    sns.boxplot(data=df_weather, x="temp_range", y="total_kwh", hue=cluster_col)
    plt.title("Consumption Distribution by Temperature Range and Cluster")
    plt.xlabel("Temperature Range")
    plt.ylabel("Total Daily kWh")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
    
    # 4. Average daily consumption profile by temperature range
    if all(f"hh_{i}" in df_weather.columns for i in range(48)):
        plt.figure(figsize=(12, 6))
        
        hh_cols = [f"hh_{i}" for i in range(48)]
        
        for temp_range in df_weather["temp_range"].unique():
            temp_profile = df_weather[df_weather["temp_range"] == temp_range][hh_cols].mean()
            plt.plot(range(48), temp_profile, label=temp_range)
        
        plt.title("Average Daily Load Profile by Temperature Range")
        plt.xlabel("Half-Hour of Day")
        plt.ylabel("Average kWh")
        plt.xticks(range(0, 48, 4), [f"{h}:00" for h in range(0, 24, 2)])
        plt.legend(title="Temperature Range")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print("✅ Weather Impact Analysis Complete!")
    return df_weather
