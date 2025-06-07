import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
   
def plot_weather_impact_analysis(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the impact of weather on different clusters in a 2x2 grid layout
    
    Includes:
    - Temperature sensitivity by cluster (correlation with HDD)
    - Cluster-wise sensitivity to heating degree days (HDD vs kWh)
    - Consumption distribution by temperature range and cluster
    - Average daily load profile by temperature range
    """
    # Prepare features if needed
    df = df_hh
    
    # Filter data
    df_weather = df[(df["total_kwh"].notna()) & 
                    (df["heating_degree_days"].notna()) & 
                    (df[cluster_col].notna())]
    df_weather[cluster_col] = df_weather[cluster_col].astype(int)
    
    # Create temperature ranges for plots
    df_weather["temp_range"] = pd.cut(df_weather["heating_degree_days"], 
                                     bins=5, 
                                     labels=["Very Warm", "Warm", "Moderate", "Cool", "Cold"])
    
    # Create a 2x2 grid of subplots with proper sizing
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Temperature correlation by cluster (top-left)
    # Compute correlations between HDD and consumption for each cluster
    corr_by_cluster = df_weather.groupby(cluster_col).apply(
        lambda x: x[["heating_degree_days", "total_kwh"]].corr().iloc[0, 1]
    ).sort_values(ascending=False)
    
    bars = sns.barplot(x=corr_by_cluster.index, y=corr_by_cluster.values, ax=axes[0, 0], palette="viridis")
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        value = corr_by_cluster.values[i]
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{value:.2f}',
            ha='center', va='bottom',
            fontsize=9
        )
    
    axes[0, 0].set_title("Temperature Sensitivity by Cluster (Correlation with HDD)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Cluster")
    axes[0, 0].set_ylabel("Correlation Coefficient")
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, axis='y', alpha=0.3)
    
    # 2. HDD vs kWh plot (top-right) - using a similar approach to plot_hdd_vs_kwh function
    # Sample data for better visualization
    sample_df = df_weather.sample(n=min(50000, len(df_weather)), random_state=42)
    
    for cluster in sorted(sample_df[cluster_col].unique()):
        cluster_data = sample_df[sample_df[cluster_col] == cluster]
        axes[0, 1].scatter(
            x=cluster_data["heating_degree_days"], 
            y=cluster_data["total_kwh"],
            alpha=0.1, 
            label=f'Cluster {cluster}'
        )
        
        # Add regression line
        z = np.polyfit(cluster_data["heating_degree_days"], cluster_data["total_kwh"], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(
            sorted(cluster_data["heating_degree_days"]), 
            p(sorted(cluster_data["heating_degree_days"])),
            linewidth=2
        )
    
    axes[0, 1].set_title("Cluster-wise Sensitivity to Heating Degree Days (HDD)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("HDD (Base 18°C)")
    axes[0, 1].set_ylabel("Total Daily kWh")
    axes[0, 1].legend(title="Cluster", loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Consumption boxplot by temperature range and cluster (bottom-left)
    sns.boxplot(data=df_weather, x="temp_range", y="total_kwh", hue=cluster_col, ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Consumption Distribution by Temperature Range and Cluster", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Temperature Range")
    axes[1, 0].set_ylabel("Total Daily kWh")
    axes[1, 0].legend(title="Cluster", loc='upper right')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, title="Cluster", loc='upper right', ncol=2)
    
    # 4. Average daily consumption profile by temperature range (bottom-right)
    if all(f"hh_{i}" in df_weather.columns for i in range(48)):
        hh_cols = [f"hh_{i}" for i in range(48)]
        
        color_map = plt.cm.get_cmap('viridis', len(df_weather["temp_range"].unique()))
        for i, temp_range in enumerate(sorted(df_weather["temp_range"].unique(), 
                                             key=lambda x: ['Very Warm', 'Warm', 'Moderate', 'Cool', 'Cold'].index(x))):
            if df_weather[df_weather["temp_range"] == temp_range].empty:
                continue
                
            temp_profile = df_weather[df_weather["temp_range"] == temp_range][hh_cols].mean()
            axes[1, 1].plot(
                range(48), 
                temp_profile, 
                label=temp_range,
                color=color_map(i/len(df_weather["temp_range"].unique())),
                linewidth=2
            )
        
        axes[1, 1].set_title("Average Daily Load Profile by Temperature Range", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Half-Hour of Day")
        axes[1, 1].set_ylabel("Average kWh")
        axes[1, 1].set_xticks(range(0, 48, 4))
        axes[1, 1].set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
        axes[1, 1].legend(title="Temperature Range", loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Weather Impact Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.93)
    plt.show()
    
    print("✅ Weather Impact Analysis Complete!")
    return 

def plot_seasonal_weather_patterns(df_hh, cluster_col="cluster"):
    """
    Plot seasonal weather patterns and their relationship to consumption
    
    This visualizes:
    - Seasonal temperature trends
    - Seasonal humidity trends
    - Seasonal consumption patterns
    """
    # Prepare features if needed
    df = df_hh
    
    # Check if required columns exist
    required_cols = ["season", "temperatureMax", "humidity", "total_kwh"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing required columns: {', '.join(missing_cols)}. Cannot create seasonal weather patterns plot.")
        return None
    
    # Filter valid data
    df_seasonal = df.dropna(subset=required_cols + [cluster_col])
    df_seasonal[cluster_col] = df_seasonal[cluster_col].astype(int)
    
    # Calculate seasonal aggregations
    seasonal_weather = df_seasonal.groupby('season').agg({
        'temperatureMax': 'mean',
        'humidity': 'mean',
        'total_kwh': 'mean'
    }).reset_index()
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(seasonal_weather['season']))
    width = 0.25
    
    ax = plt.gca()
    ax_twin = ax.twinx()
    
    bars1 = ax.bar(x - width, seasonal_weather['temperatureMax'], width, 
                    label='Temperature (°C)', color='red', alpha=0.7)
    bars2 = ax.bar(x, seasonal_weather['humidity'], width, 
                    label='Humidity (%)', color='blue', alpha=0.7)
    bars3 = ax_twin.bar(x + width, seasonal_weather['total_kwh'], width, 
                        label='Consumption (kWh)', color='green', alpha=0.7)
    
    # Add bar values
    for bars, values in zip([bars1, bars2, bars3], 
                           [seasonal_weather['temperatureMax'], 
                            seasonal_weather['humidity'], 
                            seasonal_weather['total_kwh']]):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Seasonal Weather-Consumption Patterns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Temperature (°C) / Humidity (%)', fontsize=12)
    ax_twin.set_ylabel('Average Daily kWh', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal_weather['season'])
    
    # Add two legends
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ Seasonal Weather Patterns Analysis Complete!")
    return seasonal_weather


def plot_extreme_weather_impact(df_hh, cluster_col="cluster"):
    """
    Analyze and visualize the impact of extreme weather events on different clusters
    
    This plot shows how consumption patterns change during extreme weather events
    (very hot or very cold days).
    """
    # Prepare features if needed
    df = df_hh
    
    # Check if required columns exist
    if "temperatureMax" not in df.columns and "heating_degree_days" not in df.columns:
        print("⚠️ Neither 'temperatureMax' nor 'heating_degree_days' columns found. Cannot create extreme weather impact plot.")
        return None
    
    # Filter valid data
    df_weather = df.dropna(subset=[cluster_col, "total_kwh"])
    df_weather[cluster_col] = df_weather[cluster_col].astype(int)
    
    # Identify extreme weather days
    if "temperatureMax" in df_weather.columns:
        extreme_weather = df_weather[
            (df_weather['temperatureMax'] > df_weather['temperatureMax'].quantile(0.95)) |
            (df_weather['temperatureMax'] < df_weather['temperatureMax'].quantile(0.05))
        ]
        temp_col = "temperatureMax"
    else:
        extreme_weather = df_weather[
            (df_weather['heating_degree_days'] > df_weather['heating_degree_days'].quantile(0.95)) |
            (df_weather['heating_degree_days'] < df_weather['heating_degree_days'].quantile(0.05))
        ]
        temp_col = "heating_degree_days"
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Boxplot of consumption by cluster during extreme weather
    sns.boxplot(data=extreme_weather, x=cluster_col, y="total_kwh", ax=ax1, palette="viridis")
    ax1.set_title('Extreme Weather Impact by Archetype', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Total kWh (Extreme Weather Days)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of temperature vs consumption with extreme days highlighted
    sns.scatterplot(data=df_weather, x=temp_col, y="total_kwh", 
                   alpha=0.2, color='blue', label='Normal Days', ax=ax2)
    sns.scatterplot(data=extreme_weather, x=temp_col, y="total_kwh", 
                   alpha=0.5, color='red', label='Extreme Days', ax=ax2)
    
    ax2.set_title('Consumption During Extreme vs. Normal Weather', fontsize=14, fontweight='bold')
    if temp_col == "temperatureMax":
        ax2.set_xlabel('Max Temperature (°C)', fontsize=12)
    else:
        ax2.set_xlabel('Heating Degree Days (HDD)', fontsize=12)
    ax2.set_ylabel('Total Daily kWh', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Extreme Weather Events Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ Extreme Weather Impact Analysis Complete!")
    return 
