import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar

# Use existing hh_cols definition from guru.py
hh_cols = [f"hh_{i}" for i in range(48)]

def check_dataframe_columns(df_hh):
    """
    Check what columns are available in the dataframe for plotting
    """
    print("🔍 DATAFRAME COLUMN CHECK")
    print("=" * 50)
    
    required_columns = {
        "Basic Features": [
            "LCLid", "day", "total_kwh", "mean_kwh", "std_kwh", "peak_kwh"
        ],
        "Time Features": [
            "dayofweek", "is_weekend", "season", "month", "is_holiday"
        ],
        "Weather Features": [
            "temperatureMax", "temperatureMin", "humidity"
        ],
        "Time-of-Day Features": [
            "morning_kwh", "afternoon_kwh", "evening_kwh", "night_kwh"
        ],
        "Advanced Features": [
            "is_weekday", "peak_to_mean_ratio", "day_night_ratio", 
            "daily_variability", "temp_impact"
        ],
        "Socio-Economic Features": [
            "Acorn_grouped", "stdorToU"
        ],
        "Half-Hourly Columns": hh_cols[:5] + ["..."] + hh_cols[-5:]  # Show first and last 5
    }
    
    for category, columns in required_columns.items():
        print(f"\n📊 {category}:")
        if category == "Half-Hourly Columns":
            # Special handling for half-hourly columns
            available_hh = [col for col in hh_cols if col in df_hh.columns]
            print(f"   ✅ Available: {len(available_hh)}/48 half-hourly columns")
            if len(available_hh) < 48:
                missing_hh = [col for col in hh_cols if col not in df_hh.columns]
                print(f"   ❌ Missing: {missing_hh[:5]}...")
        else:
            for col in columns:
                if col == "...":
                    continue
                status = "✅" if col in df_hh.columns else "❌"
                print(f"   {status} {col}")
    
    print(f"\n📈 Total columns in dataframe: {len(df_hh.columns)}")
    print(f"📊 DataFrame shape: {df_hh.shape}")
    
    return df_hh

def prepare_plotting_features(df_hh):
    """
    Ensure all required features exist for plotting functions
    Creates missing columns with fallback calculations
    """
    print("🔧 Preparing features for plotting...")
    
    # Create a copy to avoid modifying original data
    df = df_hh.copy()
    
    # 1. Fix is_weekday vs is_weekend inconsistency
    if 'is_weekend' in df.columns and 'is_weekday' not in df.columns:
        df['is_weekday'] = 1 - df['is_weekend']
        print("✅ Created 'is_weekday' from 'is_weekend'")
    elif 'is_weekday' not in df.columns and 'dayofweek' in df.columns:
        df['is_weekday'] = (df['dayofweek'] < 5).astype(int)
        print("✅ Created 'is_weekday' from 'dayofweek'")
    
    # 2. Create peak_to_mean_ratio if missing
    if 'peak_to_mean_ratio' not in df.columns:
        if 'peak_kwh' in df.columns and 'mean_kwh' in df.columns:
            df['peak_to_mean_ratio'] = df['peak_kwh'] / df['mean_kwh']
            df['peak_to_mean_ratio'] = df['peak_to_mean_ratio'].replace([np.inf, -np.inf], np.nan)
            print("✅ Created 'peak_to_mean_ratio' from peak_kwh/mean_kwh")
        else:
            print("⚠️  Cannot create 'peak_to_mean_ratio' - missing peak_kwh or mean_kwh")
    
    # 3. Create day_night_ratio if missing
    if 'day_night_ratio' not in df.columns:
        if all(col in df.columns for col in ['morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh']):
            day_consumption = df['morning_kwh'] + df['afternoon_kwh'] 
            night_consumption = df['evening_kwh'] + df['night_kwh']
            df['day_night_ratio'] = day_consumption / night_consumption
            df['day_night_ratio'] = df['day_night_ratio'].replace([np.inf, -np.inf], np.nan)
            print("✅ Created 'day_night_ratio' from time-of-day columns")
        else:
            print("⚠️  Cannot create 'day_night_ratio' - missing time-of-day columns")
    
    # 4. Create daily_variability if missing
    if 'daily_variability' not in df.columns:
        df['daily_variability'] = df[hh_cols].std(axis=1)
        print("✅ Created 'daily_variability' from half-hourly readings")
    
    # 5. Create temp_impact if missing (simplified proxy)
    if 'temp_impact' not in df.columns and 'temperatureMax' in df.columns:
        # Simple temperature impact approximation
        temp_corr = df.groupby('LCLid').apply(
            lambda x: x['total_kwh'].corr(x['temperatureMax']) if len(x) > 1 else 0
        )
        df['temp_impact'] = df['LCLid'].map(temp_corr)
        print("✅ Created 'temp_impact' as temperature-consumption correlation")
    
    # 6. Ensure other time-of-day features exist
    if not all(col in df.columns for col in ['morning_kwh', 'afternoon_kwh', 'evening_kwh', 'night_kwh']):
        print("⚠️  Creating basic time-of-day features from half-hourly data...")
        df['morning_kwh'] = df[[f"hh_{i}" for i in range(6, 12)]].sum(axis=1)  # 3-6 AM
        df['afternoon_kwh'] = df[[f"hh_{i}" for i in range(24, 36)]].sum(axis=1)  # 12-6 PM
        df['evening_kwh'] = df[[f"hh_{i}" for i in range(36, 48)]].sum(axis=1)  # 6 PM-12 AM
        df['night_kwh'] = df[[f"hh_{i}" for i in range(0, 6)]].sum(axis=1)  # 12-3 AM
        print("✅ Created time-of-day features")
    
    # 7. Ensure basic stats exist
    required_basic_features = ['total_kwh', 'mean_kwh', 'std_kwh', 'peak_kwh']
    for feature in required_basic_features:
        if feature not in df.columns:
            if feature == 'total_kwh':
                df['total_kwh'] = df[hh_cols].sum(axis=1)
            elif feature == 'mean_kwh':
                df['mean_kwh'] = df[hh_cols].mean(axis=1)
            elif feature == 'std_kwh':
                df['std_kwh'] = df[hh_cols].std(axis=1)
            elif feature == 'peak_kwh':
                df['peak_kwh'] = df[hh_cols].max(axis=1)
            print(f"✅ Created '{feature}'")
    
    print(f"🎯 Feature preparation complete! DataFrame shape: {df.shape}")
    return df

def plot_load_shape_analysis(df_hh, cluster_col="cluster"):
    """
    Research Question 1 & 5: Peak usage patterns and load-shape archetypes
    Combined analysis of daily load patterns, cluster characteristics, and peak usage times
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average Daily Load Profile by Cluster (Core archetypes)
    k = df[cluster_col].nunique()
    for i in range(k):
        avg_profile = df[df[cluster_col] == i][hh_cols].mean()
        ax1.plot(range(48), avg_profile, label=f"Cluster {i}", linewidth=2)
    ax1.set_title("Load Shape Archetypes - Daily Profiles")
    ax1.set_xlabel("Half-Hour Interval (0=midnight)")
    ax1.set_ylabel("Average kWh")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Weekday vs Weekend Peak Patterns
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        weekday_data = cluster_data[cluster_data['is_weekday'] == 1][hh_cols].mean()
        weekend_data = cluster_data[cluster_data['is_weekday'] == 0][hh_cols].mean()
        
        ax2.plot(range(48), weekday_data, label=f'Cluster {cluster} Weekday', linestyle='-')
        ax2.plot(range(48), weekend_data, label=f'Cluster {cluster} Weekend', linestyle='--', alpha=0.7)
    
    ax2.set_title('Peak Usage Hours: Weekday vs Weekend')
    ax2.set_xlabel('Half-Hour Interval')
    ax2.set_ylabel('Average kWh')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # 3. Cluster Size Distribution
    cluster_counts = df[cluster_col].value_counts().sort_values(ascending=False)
    total = cluster_counts.sum()
    
    bars = ax3.bar(cluster_counts.index.astype(str), cluster_counts.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        pct = f"{100 * count / total:.1f}%"
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{count}\n({pct})", ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title("Load Shape Archetype Distribution")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Number of Households")
    ax3.grid(axis='y', alpha=0.7)
    
    # 4. Load Shape Characteristics
    if 'peak_to_mean_ratio' in df.columns:
        sns.boxplot(data=df, x=cluster_col, y='peak_to_mean_ratio', ax=ax4)
        ax4.set_title('Peak Load Intensity by Archetype')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Peak to Mean Ratio')
    else:
        # Fallback: use total consumption
        sns.boxplot(data=df, x=cluster_col, y='total_kwh', ax=ax4)
        ax4.set_title('Total Consumption by Archetype')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Total Daily kWh')
    ax4.grid(True)
    
    plt.suptitle("Research Questions 1 & 5: Load Shape Archetypes & Peak Usage Analysis", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_weather_impact_analysis(df_hh, cluster_col="cluster"):
    """
    Research Question 2 & 6: Weather variations and joint weather-socio-economic effects
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Temperature Impact on Total Consumption by Cluster
    sns.scatterplot(data=df, x="temperatureMax", y="total_kwh", 
                   hue=cluster_col, alpha=0.6, ax=ax1)
    ax1.set_title("Temperature Impact on Daily Consumption")
    ax1.set_xlabel("Max Temperature (°C)")
    ax1.set_ylabel("Total Daily kWh")
    ax1.grid(True)
    
    # 2. Seasonal Weather Patterns
    seasonal_weather = df.groupby('season').agg({
        'temperatureMax': 'mean',
        'humidity': 'mean',
        'total_kwh': 'mean'
    }).reset_index()
    
    x = np.arange(len(seasonal_weather['season']))
    width = 0.25
    
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width, seasonal_weather['temperatureMax'], width, 
                    label='Temperature (°C)', color='red', alpha=0.7)
    bars2 = ax2.bar(x, seasonal_weather['humidity'], width, 
                    label='Humidity (%)', color='blue', alpha=0.7)
    bars3 = ax2_twin.bar(x + width, seasonal_weather['total_kwh'], width, 
                        label='Consumption (kWh)', color='green', alpha=0.7)
    
    ax2.set_title('Seasonal Weather-Consumption Patterns')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Temperature (°C) / Humidity (%)')
    ax2_twin.set_ylabel('Average Daily kWh')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seasonal_weather['season'])
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Weather Impact by ACORN Group and Cluster
    if 'temp_impact' in df.columns and 'Acorn_grouped' in df.columns:
        sns.boxplot(data=df, x='Acorn_grouped', y='temp_impact', 
                   hue=cluster_col, ax=ax3)
        ax3.set_title('Temperature Sensitivity by Socio-Economic Group')
        ax3.set_xlabel('ACORN Group')
        ax3.set_ylabel('Temperature Impact (correlation)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        # Fallback: Consumption by ACORN and temperature ranges
        df_temp = df.copy()
        df_temp['temp_range'] = pd.cut(df_temp['temperatureMax'], 
                                      bins=[-10, 5, 15, 25, 35], 
                                      labels=['Cold', 'Cool', 'Mild', 'Warm'])
        if 'Acorn_grouped' in df.columns:
            sns.boxplot(data=df_temp, x='Acorn_grouped', y='total_kwh', 
                       hue='temp_range', ax=ax3)
            ax3.set_title('Consumption by Socio-Economic Group & Temperature')
            ax3.set_xlabel('ACORN Group')
        else:
            sns.boxplot(data=df_temp, x=cluster_col, y='total_kwh', 
                       hue='temp_range', ax=ax3)
            ax3.set_title('Consumption by Cluster & Temperature')
            ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Total kWh')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Extreme Weather Events Impact
    extreme_weather = df[
        (df['temperatureMax'] > df['temperatureMax'].quantile(0.95)) |
        (df['temperatureMax'] < df['temperatureMax'].quantile(0.05))
    ]
    sns.boxplot(data=extreme_weather, x=cluster_col, y='total_kwh', ax=ax4)
    ax4.set_title('Extreme Weather Impact by Archetype')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Total kWh (Extreme Weather Days)')
    ax4.grid(True)
    
    plt.suptitle("Research Questions 2 & 6: Weather Impact & Socio-Economic Interactions", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_socioeconomic_intervention_analysis(df_hh, cluster_col="cluster"):
    """
    Research Question 4 & 8: Interventions and socio-economic optimization strategies
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. ACORN Group Distribution by Cluster
    if "Acorn_grouped" in df.columns:
        acorn_dist = pd.crosstab(df[cluster_col], df["Acorn_grouped"], normalize="index")
        acorn_dist.plot(kind="bar", stacked=True, ax=ax1, 
                       colormap='Set3')
        ax1.set_title("Socio-Economic Distribution by Load Archetype")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Proportion")
        ax1.legend(title="ACORN Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.tick_params(axis='x', rotation=0)
    else:
        # Fallback: Show cluster distribution
        cluster_counts = df[cluster_col].value_counts()
        ax1.bar(cluster_counts.index.astype(str), cluster_counts.values)
        ax1.set_title("Load Archetype Distribution")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Count")
    
    # 2. Load Shifting Potential (Peak vs Off-Peak)
    if 'day_night_ratio' in df.columns:
        sns.boxplot(data=df, x=cluster_col, y='day_night_ratio', ax=ax2)
        ax2.set_title('Load Shifting Potential by Archetype')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Day-Night Consumption Ratio')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Day/Night')
        ax2.legend()
    else:
        # Fallback: Calculate basic day/night ratio
        day_cols = [f"hh_{i}" for i in range(12, 36)]  # 6 AM to 6 PM
        night_cols = [f"hh_{i}" for i in list(range(0, 12)) + list(range(36, 48))]  # 6 PM to 6 AM
        
        df_temp = df.copy()
        df_temp['day_sum'] = df_temp[day_cols].sum(axis=1)
        df_temp['night_sum'] = df_temp[night_cols].sum(axis=1)
        df_temp['day_night_ratio'] = df_temp['day_sum'] / df_temp['night_sum']
        
        sns.boxplot(data=df_temp, x=cluster_col, y='day_night_ratio', ax=ax2)
        ax2.set_title('Load Shifting Potential by Archetype')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Day-Night Consumption Ratio')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Day/Night')
        ax2.legend()
    
    ax2.grid(True)
    
    # 3. Peak Load Optimization Potential
    if 'peak_to_mean_ratio' in df.columns and 'Acorn_grouped' in df.columns:
        sns.boxplot(data=df, x='Acorn_grouped', y='peak_to_mean_ratio', 
                   hue=cluster_col, ax=ax3)
        ax3.set_title('Peak Load Reduction Potential')
        ax3.set_xlabel('ACORN Group')
        ax3.set_ylabel('Peak to Mean Ratio')
        ax3.tick_params(axis='x', rotation=45)
    else:
        # Fallback: Basic consumption analysis by cluster
        if 'Acorn_grouped' in df.columns:
            sns.boxplot(data=df, x='Acorn_grouped', y='total_kwh', 
                       hue=cluster_col, ax=ax3)
            ax3.set_title('Consumption Patterns by Socio-Economic Group')
            ax3.set_xlabel('ACORN Group')
        else:
            sns.boxplot(data=df, x=cluster_col, y='total_kwh', ax=ax3)
            ax3.set_title('Consumption Patterns by Cluster')
            ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Total Daily kWh')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Intervention Priority Matrix
    # Calculate intervention scores by cluster
    cluster_stats = df.groupby(cluster_col).agg({
        'total_kwh': 'mean',
        'daily_variability': 'mean' if 'daily_variability' in df.columns else lambda x: x.std()
    }).reset_index()
    
    if 'daily_variability' not in cluster_stats.columns:
        cluster_stats['daily_variability'] = df.groupby(cluster_col)[hh_cols].apply(lambda x: x.std(axis=1).mean())
    
    scatter = ax4.scatter(cluster_stats['total_kwh'], cluster_stats['daily_variability'], 
                         c=cluster_stats[cluster_col], s=100, alpha=0.7, cmap='viridis')
    
    for i, row in cluster_stats.iterrows():
        ax4.annotate(f"Cluster {row[cluster_col]}", 
                    (row['total_kwh'], row['daily_variability']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_title('Intervention Priority Matrix')
    ax4.set_xlabel('Average Daily Consumption (kWh)')
    ax4.set_ylabel('Consumption Variability')
    ax4.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax4.axhline(y=cluster_stats['daily_variability'].median(), color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=cluster_stats['total_kwh'].median(), color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle("Research Questions 4 & 8: Intervention Strategies & Optimization", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_pattern_evolution_analysis(df_hh, cluster_col="cluster"):
    """
    Research Question 5: Load-shape archetype evolution over time
    """
    # Prepare features if needed
    df = prepare_plotting_features(df_hh)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Seasonal Load Signature
    df_copy = df.copy()
    df_copy['day'] = pd.to_datetime(df_copy['day'])
    df_copy["month_day"] = df_copy['day'].dt.strftime("%m-%d")

    grouped = df_copy.groupby(["month_day", df_copy[cluster_col]])['total_kwh'].mean().reset_index()
    pivot = grouped.pivot(index="month_day", columns=cluster_col, values='total_kwh')
    pivot = pivot.sort_index()

    month_start_idx = []
    month_labels = []
    for m in range(1, 13):
        month_str = f"{m:02d}-01"
        if month_str in pivot.index:
            month_start_idx.append(pivot.index.get_loc(month_str))
            month_labels.append(calendar.month_abbr[m])

    for col in pivot.columns:
        ax1.plot(pivot.index, pivot[col], label=f"Cluster {int(col)}", linewidth=2)

    ax1.set_title("Seasonal Load Signature Evolution")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Daily kWh")
    if month_start_idx and month_labels:
        ax1.set_xticks(ticks=[pivot.index[i] for i in month_start_idx], labels=month_labels)
    ax1.legend()
    ax1.grid(True)
    
    # 2. Seasonal Profile Comparison
    seasons = sorted(df["season"].dropna().unique())
    k = df[cluster_col].nunique()
    
    # Select one representative cluster for seasonal comparison
    main_cluster = df[cluster_col].mode()[0]
    cluster_data = df[df[cluster_col] == main_cluster]
    
    for season in seasons:
        season_data = cluster_data[cluster_data["season"] == season]
        if not season_data.empty:
            avg_profile = season_data[hh_cols].mean()
            ax2.plot(range(48), avg_profile, label=f"{season}", linewidth=2, marker='o', markersize=3)
    
    ax2.set_title(f"Seasonal Pattern Evolution - Cluster {main_cluster}")
    ax2.set_xlabel("Half-Hour Interval")
    ax2.set_ylabel("Average kWh")
    ax2.legend()
    ax2.grid(True)
    
    # 3. Monthly Pattern Stability
    df_copy['month_period'] = pd.to_datetime(df_copy['day']).dt.to_period('M')
    monthly_evolution = df_copy.groupby(['month_period', cluster_col])[hh_cols].mean()
    
    for cluster in sorted(df[cluster_col].unique()):
        if cluster in monthly_evolution.index.get_level_values(1):
            cluster_data = monthly_evolution.xs(cluster, level=1)
            if not cluster_data.empty:
                ax3.plot(range(len(cluster_data)), cluster_data.mean(axis=1), 
                        label=f'Cluster {cluster}', marker='o', linewidth=2)
    
    ax3.set_title('Pattern Stability Over Time')
    ax3.set_xlabel('Month Index')
    ax3.set_ylabel('Average Daily kWh')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Pattern Variability by Season and Cluster
    if 'daily_variability' in df.columns:
        sns.boxplot(data=df, x='season', y='daily_variability', 
                   hue=cluster_col, ax=ax4)
        ax4.set_title('Pattern Variability by Season')
        ax4.set_xlabel('Season')
        ax4.set_ylabel('Daily Variability (kWh)')
    else:
        # Fallback: Use standard deviation across half-hourly readings
        df_var = df.copy()
        df_var['daily_variability'] = df_var[hh_cols].std(axis=1)
        sns.boxplot(data=df_var, x='season', y='daily_variability', 
                   hue=cluster_col, ax=ax4)
        ax4.set_title('Pattern Variability by Season')
        ax4.set_xlabel('Season')
        ax4.set_ylabel('Daily Variability (kWh)')
    
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Research Question 5: Load-Shape Archetype Evolution", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main(df_hh):
    """
    Main function to run all research-question-focused plots
    """
    print("🎯 Smart Meter Analysis: Research Question Focused Plots")
    print("=" * 60)
    
    # Check dataframe columns first
    check_dataframe_columns(df_hh)
    
    # Prepare data first
    print("\n" + "="*60)
    df_prepared = prepare_plotting_features(df_hh)
    
    print("\n" + "="*60)
    print("📊 1. Load Shape Analysis (RQ1 & RQ5)")
    plot_load_shape_analysis(df_prepared)
    
    print("🌤️  2. Weather Impact Analysis (RQ2 & RQ6)")  
    plot_weather_impact_analysis(df_prepared)
    
    print("🏠 3. Socio-Economic & Intervention Analysis (RQ4 & RQ8)")
    plot_socioeconomic_intervention_analysis(df_prepared)
    
    print("📈 4. Pattern Evolution Analysis (RQ5)")
    plot_pattern_evolution_analysis(df_prepared)
    
    print("✅ Analysis Complete!")

# Usage instructions
def usage_instructions():
    """
    Print usage instructions for the plotting functions
    """
    print("\n" + "="*80)
    print("📖 USAGE INSTRUCTIONS")
    print("="*80)
    print("""
🔧 SETUP:
1. Import the module: from optimized_plots import *
2. Load your dataframe: df_hh = pd.read_csv('your_data.csv')

🚀 QUICK START:
# Check what columns you have:
check_dataframe_columns(df_hh)

# Run all plots:
main(df_hh)

# Or run individual plot functions:
plot_load_shape_analysis(df_hh, cluster_col="your_cluster_column")

🎯 INDIVIDUAL FUNCTIONS:
- plot_load_shape_analysis(df_hh, cluster_col="cluster")
- plot_weather_impact_analysis(df_hh, cluster_col="cluster") 
- plot_socioeconomic_intervention_analysis(df_hh, cluster_col="cluster")
- plot_pattern_evolution_analysis(df_hh, cluster_col="cluster")

🔧 FEATURE PREPARATION:
The functions automatically create missing features, but you can also:
df_prepared = prepare_plotting_features(df_hh)

📋 REQUIREMENTS:
- Minimum: Half-hourly columns (hh_0 to hh_47)
- Recommended: cluster column, basic time features
- Optional: weather data, ACORN data for enhanced analysis
""")
    print("="*80)

if __name__ == "__main__":
    usage_instructions()
    print("\n⚠️  Please call main(df_hh) with your dataframe to run the analysis.")