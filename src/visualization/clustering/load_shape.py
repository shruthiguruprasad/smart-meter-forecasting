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
    
    # Choose a colormap based on number of clusters
    num_clusters = len(cluster_counts)
    if num_clusters <= 4:
        # For fewer clusters, use distinct colors
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD'][:num_clusters]
    else:
        # For more clusters, use a gradient
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_clusters))
    
    # Plot cluster distribution
    bars = axes[1, 0].bar(
        cluster_counts.index.astype(str), 
        cluster_counts.values,
        color=colors,
        width=0.7
    )
    
    # Add count and percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        pct = f"{100 * count / total:.1f}%"
        # Adjust position based on bar height
        if count/max(cluster_counts.values) > 0.3:  # If bar is tall enough
            axes[1, 0].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() - (max(cluster_counts.values) * 0.05),  # Place inside the bar 
                f"{count:,}\n({pct})", 
                ha='center', va='center', 
                fontweight='bold', fontsize=9,
                color='white'  # White text for contrast
            )
        else:
            axes[1, 0].text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (max(cluster_counts.values) * 0.02),  # Place above the bar
                f"{count:,}\n({pct})", 
                ha='center', va='bottom', 
                fontweight='bold', fontsize=9
            )
    
    # Improve the appearance of the plot
    axes[1, 0].set_title(f"Cluster Distribution ({count_type})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Cluster")
    axes[1, 0].set_ylabel(f"Number of {count_type}")
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limit to provide some space for labels
    y_max = max(cluster_counts.values)
    axes[1, 0].set_ylim(0, y_max * 1.1)  # 10% extra space
    
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
      # Peak timing analysis
    if 'peak_hour' in df_plot.columns:
        plt.figure(figsize=(12, 6))
        
        # Create consistent color scheme
        num_clusters = len(sorted(df_plot[cluster_col].unique()))
        if num_clusters <= 4:
            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD'][:num_clusters]
        else:
            colors = plt.cm.viridis(np.linspace(0, 0.9, num_clusters))
            
        for i, cluster in enumerate(sorted(df_plot[cluster_col].unique())):
            cluster_data = df_plot[df_plot[cluster_col] == cluster]
            color = colors[i] if isinstance(colors[i], str) else colors[i]
            sns.kdeplot(
                cluster_data['peak_hour'], 
                label=f"Cluster {cluster}",
                color=color,
                linewidth=2.5,
                alpha=0.8
            )
        
        plt.title("Distribution of Peak Consumption Hours by Cluster", fontsize=14, fontweight='bold')
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.xticks(range(0, 24, 2))
        plt.xlim(0, 23)  # Ensure the x-axis limits are fixed to hour boundaries
        plt.legend(title="Cluster", loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    print("✅ Load Shape Analysis Complete!")
    return cluster_profiles
