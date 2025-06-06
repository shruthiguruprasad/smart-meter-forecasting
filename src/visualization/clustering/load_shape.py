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
    
    # Create a 2×2 grid layout for key visualizations
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Average daily load profiles by cluster (top-left)
    cluster_profiles = {}
    for cluster in sorted(df_plot[cluster_col].unique()):
        cluster_data = df_plot[df_plot[cluster_col] == cluster]
        
        # Calculate the average profile
        avg_profile = cluster_data[hh_cols].mean().values
        cluster_profiles[cluster] = avg_profile
        
        # Plot the profile
        axes[0, 0].plot(range(48), avg_profile, label=f"Cluster {cluster}", linewidth=2)
    
    axes[0, 0].set_title("Average Daily Load Profile by Cluster", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Half-Hour of Day")
    axes[0, 0].set_ylabel("Average kWh")
    axes[0, 0].set_xticks(range(0, 48, 4))
    axes[0, 0].set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
    axes[0, 0].legend(title="Cluster")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Weekday vs Weekend patterns (top-right)
    if 'is_weekend' in df_plot.columns:
        for cluster in sorted(df_plot[cluster_col].unique()):
            cluster_data = df_plot[df_plot[cluster_col] == cluster]
            weekday_data = cluster_data[cluster_data['is_weekend'] == 0][hh_cols].mean()
            weekend_data = cluster_data[cluster_data['is_weekend'] == 1][hh_cols].mean()
            
            axes[0, 1].plot(range(48), weekday_data, 
                         label=f'Cluster {cluster} Weekday', linestyle='-')
            axes[0, 1].plot(range(48), weekend_data, 
                         label=f'Cluster {cluster} Weekend', linestyle='--', alpha=0.7)
        
        axes[0, 1].set_title('Weekday vs Weekend Load Profiles', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Half-Hour of Day')
        axes[0, 1].set_ylabel('Average kWh')
        axes[0, 1].set_xticks(range(0, 48, 4))
        axes[0, 1].set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
        axes[0, 1].legend(title="Cluster & Day Type", ncol=2, fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, "Weekday/Weekend data not available", 
                      ha='center', va='center', fontsize=12)
        axes[0, 1].set_title('Weekday vs Weekend Load Profiles (No Data)', 
                           fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
    
    # 3. Cluster Size Distribution (bottom-left)
    # Count households by cluster (use LCLid if available, otherwise count data points)
    if 'LCLid' in df_plot.columns:
        # Count unique households in each cluster
        unique_households = df_plot.drop_duplicates(subset=['LCLid'])[cluster_col].value_counts().sort_index()
        cluster_counts = unique_households
        count_type = "Households"
    else:
        # Count data points as fallback
        cluster_counts = df_plot[cluster_col].value_counts().sort_index()
        count_type = "Data Points"
    
    total = cluster_counts.sum()
    
    # Plot cluster distribution
    bars = axes[1, 0].bar(
        cluster_counts.index.astype(str), 
        cluster_counts.values,
        color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    )
    
    # Add count and percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        pct = f"{100 * count / total:.1f}%"
        axes[1, 0].text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + (max(cluster_counts.values) * 0.02),  # Small offset
            f"{count:,}\n({pct})", 
            ha='center', va='bottom', 
            fontweight='bold', fontsize=9
        )
    
    axes[1, 0].set_title(f"Cluster Distribution ({count_type})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Cluster")
    axes[1, 0].set_ylabel(f"Number of {count_type}")
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Peak Load Characteristics (bottom-right)
    if 'peak_to_mean_ratio' in df_plot.columns:
        sns.boxplot(data=df_plot, x=cluster_col, y='peak_to_mean_ratio', ax=axes[1, 1], palette="viridis")
        axes[1, 1].set_title('Peak Load Intensity by Cluster', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Peak to Mean Ratio')
    else:
        # Fallback: use total consumption
        sns.boxplot(data=df_plot, x=cluster_col, y='total_kwh', ax=axes[1, 1], palette="viridis")
        axes[1, 1].set_title('Total Consumption by Cluster', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Total Daily kWh')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Load Shape Archetypes & Cluster Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.93)
    plt.show()
    
    # Continue with additional detailed visualizations
    
    # Load profile heatmap for each cluster
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
    
    # Peak timing analysis
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
    
    # Seasonal patterns
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
