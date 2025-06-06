import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import prepare_plotting_features, hh_cols

def plot_load_shape_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the load shapes for each cluster
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    # Check if half-hourly data is available
    if not all(f'hh_{i}' in df.columns for i in range(48)):
        print("⚠️ Half-hourly consumption data not found. Unable to perform load shape analysis.")
        return None
    
    # Filter data for plotting
    df_plot = df[df[cluster_col].notna()].copy()
    df_plot[cluster_col] = df_plot[cluster_col].astype(int)
    
    # 1. Average daily load profiles by cluster
    plt.figure(figsize=(12, 6))
    
    cluster_profiles = {}
    for cluster in sorted(df_plot[cluster_col].unique()):
        cluster_data = df_plot[df_plot[cluster_col] == cluster]
        
        # Calculate the average profile
        avg_profile = cluster_data[hh_cols].mean().values
        cluster_profiles[cluster] = avg_profile
        
        # Plot the profile
        plt.plot(range(48), avg_profile, label=f"Cluster {cluster}")
    
    plt.title("Average Daily Load Profile by Cluster")
    plt.xlabel("Half-Hour of Day")
    plt.ylabel("Average kWh")
    plt.xticks(range(0, 48, 4), [f"{h}:00" for h in range(0, 24, 2)])
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Load profile heatmap for each cluster
    clusters = sorted(df_plot[cluster_col].unique())
    fig, axs = plt.subplots(1, len(clusters), figsize=(5*len(clusters), 5), sharey=True)
    
    if len(clusters) == 1:
        axs = [axs]  # Convert to list for consistent indexing
    
    for i, cluster in enumerate(clusters):
        cluster_data = df_plot[df_plot[cluster_col] == cluster]
        
        # Calculate normalized average profile
        avg_profile = cluster_data[hh_cols].mean().values
        normalized_profile = avg_profile / avg_profile.max()
        
        # Create a 24x2 reshape for better visualization (hours x half-hours)
        profile_matrix = normalized_profile.reshape(24, 2)
        
        # Plot heatmap
        sns.heatmap(profile_matrix, cmap="YlOrRd", ax=axs[i], cbar=(i == len(clusters)-1))
        axs[i].set_title(f"Cluster {cluster}")
        axs[i].set_xlabel("Half-Hour")
        
        if i == 0:
            axs[i].set_ylabel("Hour of Day")
        
        # Set y-axis ticks to show hours
        axs[i].set_yticks(range(0, 24, 3))
        axs[i].set_yticklabels(range(0, 24, 3))
        
        # Set x-axis ticks
        axs[i].set_xticks([0, 1])
        axs[i].set_xticklabels(['00', '30'])
    
    plt.tight_layout()
    plt.show()
    
    # 3. Peak timing analysis
    if 'peak_hour' in df_plot.columns:
        plt.figure(figsize=(12, 6))
        
        for cluster in sorted(df_plot[cluster_col].unique()):
            cluster_data = df_plot[df_plot[cluster_col] == cluster]
            sns.kdeplot(cluster_data['peak_hour'], label=f"Cluster {cluster}")
        
        plt.title("Distribution of Peak Consumption Hours by Cluster")
        plt.xlabel("Hour of Day")
        plt.ylabel("Density")
        plt.xticks(range(0, 24, 2))
        plt.legend(title="Cluster")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 4. Weekday vs Weekend patterns
    if 'is_weekend' in df_plot.columns:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        for day_type, ax, title in zip([0, 1], axs, ["Weekday", "Weekend"]):
            for cluster in sorted(df_plot[cluster_col].unique()):
                cluster_day_data = df_plot[(df_plot[cluster_col] == cluster) & 
                                          (df_plot['is_weekend'] == day_type)]
                
                if len(cluster_day_data) > 0:
                    avg_profile = cluster_day_data[hh_cols].mean().values
                    ax.plot(range(48), avg_profile, label=f"Cluster {cluster}")
            
            ax.set_title(f"Average {title} Load Profile by Cluster")
            ax.set_xlabel("Half-Hour of Day")
            ax.set_xticks(range(0, 48, 4))
            ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
            ax.grid(True, alpha=0.3)
            
            if day_type == 0:
                ax.set_ylabel("Average kWh")
        
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    # 5. Seasonal patterns
    if 'season' in df_plot.columns:
        seasons = sorted(df_plot['season'].unique())
        fig, axs = plt.subplots(len(seasons), 1, figsize=(12, 4*len(seasons)), sharex=True)
        
        if len(seasons) == 1:
            axs = [axs]  # Convert to list for consistent indexing
        
        for i, season in enumerate(seasons):
            for cluster in sorted(df_plot[cluster_col].unique()):
                cluster_season_data = df_plot[(df_plot[cluster_col] == cluster) & 
                                             (df_plot['season'] == season)]
                
                if len(cluster_season_data) > 0:
                    avg_profile = cluster_season_data[hh_cols].mean().values
                    axs[i].plot(range(48), avg_profile, label=f"Cluster {cluster}")
            
            axs[i].set_title(f"{season.capitalize()} Average Load Profile by Cluster")
            axs[i].set_ylabel("Average kWh")
            axs[i].grid(True, alpha=0.3)
            
            if i == len(seasons) - 1:
                axs[i].set_xlabel("Half-Hour of Day")
                axs[i].set_xticks(range(0, 48, 4))
                axs[i].set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
        
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    print("✅ Load Shape Analysis Complete!")
    return cluster_profiles
