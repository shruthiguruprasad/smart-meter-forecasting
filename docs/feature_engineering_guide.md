# 🔧 Smart Meter Forecasting - Feature Engineering Guide

## 📋 Overview

This document provides a comprehensive guide to the feature engineering system designed for electricity consumption forecasting using the London Smart Meter dataset. The system creates **80+ carefully designed features** that support multi-horizon forecasting across different modeling approaches.

## 🎯 Research Framework Support

Our feature engineering system supports a **3-stage research framework**:

- **Stage 0**: Exploratory & Predictive Modeling (Feature importance with XGBoost + SHAP)
- **Stage 1**: Day-ahead forecasting (1-day horizon)
- **Stage 2**: Multi-horizon forecasting (1 week, 1 month)

## 🏗️ Feature Architecture

### Core Modules

```
src/features/
├── consumption_features.py  # Core consumption patterns
├── temporal_features.py     # Time-based features
├── weather_features.py      # Weather & environmental factors
├── feature_pipeline.py      # Orchestration & integration
└── docs/                   # This documentation
```

---

## 📊 Feature Categories

### 1. 🔋 **Consumption Features** (Foundation)

**Purpose**: Capture fundamental electricity usage patterns and characteristics.

#### Basic Consumption Statistics
- `total_kwh`: Daily total consumption (primary target)
- `mean_kwh`: Average half-hourly consumption
- `peak_kwh`: Maximum half-hourly consumption
- `min_kwh`: Minimum half-hourly consumption
- `std_kwh`: Standard deviation of consumption
- `peak_hour`: Hour of peak consumption (0-47)

#### Time-of-Day Consumption Patterns
- `morning_kwh`: 3-6 AM consumption (early morning)
- `afternoon_kwh`: 6-9 AM consumption (morning peak)
- `evening_kwh`: 9 AM-12 PM consumption (daytime)
- `night_kwh`: Night hours consumption (12 PM-3 AM)
- `peak_period_kwh`: 5-8 PM consumption (UK evening peak)
- `off_peak_kwh`: Midnight-6 AM consumption (lowest tariff)

#### Consumption Ratios & Patterns
- `peak_to_mean_ratio`: Peak intensity indicator
- `peak_to_total_ratio`: Peak contribution to daily total
- `day_night_ratio`: Daytime vs nighttime usage pattern
- `load_factor`: Efficiency indicator (mean/peak)
- `usage_concentration`: How concentrated usage is across the day
- `peak_sharpness`: Sharpness of consumption peaks
- `base_load`: Minimum sustained consumption (10th percentile)
- `base_load_ratio`: Base load contribution to total

#### Variability Measures
- `daily_variability`: Standard deviation of half-hourly data
- `coefficient_of_variation`: Normalized variability measure

### 2. 📅 **Temporal Features** (Time Series Foundation)

**Purpose**: Capture calendar effects, seasonality, and time-based patterns critical for forecasting.

#### Basic Temporal Components
- `dayofweek`: Day of week (0=Monday, 6=Sunday)
- `month`: Month of year (1-12)
- `day_of_year`: Day within year (1-365/366)
- `is_weekend`: Weekend indicator (Sat/Sun)
- `is_weekday`: Weekday indicator (Mon-Fri)
- `is_monday`: Monday indicator (return-to-work effect)
- `is_friday`: Friday indicator (end-of-week effect)

#### Cyclical Encoding (for ML Models)
- `month_sin/cos`: Sinusoidal encoding of month (captures seasonality)
- `dayofweek_sin/cos`: Sinusoidal encoding of day of week

#### Seasonal Indicators
- `is_winter`: Winter season indicator
- `is_summer`: Summer season indicator  
- `is_shoulder_season`: Spring/Autumn indicator
- `quarter`: Quarterly indicator (1-4)
- `is_q1`: Q1 indicator (winter/spring transition)
- `is_q4`: Q4 indicator (autumn/winter transition)

#### Holiday Effects
- `is_holiday`: Any holiday indicator
- `is_christmas`: Christmas period indicator
- `is_new_year`: New Year period indicator
- `is_bank_holiday`: Bank holiday indicator
- `holiday_weekday`: Holiday on weekday interaction
- `holiday_weekend`: Holiday on weekend interaction

#### Peak Period Features (UK-Specific)
- `is_morning_peak`: 7-9 AM peak indicator
- `is_evening_peak`: 5-8 PM peak indicator
- `is_off_peak`: Midnight-6 AM off-peak indicator
- `weekday_evening_peak`: Weekday evening peak interaction
- `weekend_peak_shift`: Weekend morning peak shift

#### Time Trends
- `days_since_start`: Linear time trend (days from dataset start)

### 3. 🌤️ **Weather Features** (External Factors)

**Purpose**: Capture weather impacts on electricity consumption, particularly heating/cooling demand.

#### Temperature Features
- `temp_avg`: Average daily temperature
- `temp_range`: Daily temperature range (max - min)
- `heating_degree_days`: Heating demand indicator (base 15°C)
- `cooling_degree_days`: Cooling demand indicator (base 22°C)
- `is_very_cold`: Extreme cold indicator (<5°C)
- `is_very_hot`: Extreme heat indicator (>25°C)

#### Weather Conditions
- `is_high_humidity`: High humidity indicator (>80%)
- `is_low_humidity`: Low humidity indicator (<30%)
- `is_windy`: High wind indicator (>75th percentile)
- `wind_calm`: Low wind indicator (<25th percentile)
- `is_cloudy`: High cloud cover indicator (>70%)
- `is_clear`: Low cloud cover indicator (<30%)

#### Household-Specific Weather Sensitivity
- `temp_sensitivity`: Household-specific temperature correlation

#### Seasonal Weather Interactions
- `winter_heating_need`: Winter heating degree days
- `summer_cooling_need`: Summer cooling degree days

### 4. 📈 **Time Series Features** (Critical for Forecasting)

**Purpose**: Capture temporal dependencies and patterns essential for forecasting models.

#### Lag Features
- `lag1_total`: Yesterday's consumption (short-term dependency)
- `lag7_total`: Same day last week (weekly pattern)
- `lag14_total`: Same day 2 weeks ago (longer-term pattern)

#### Rolling Window Features
- `roll7_total_mean`: 7-day rolling average (weekly trend)
- `roll14_total_mean`: 14-day rolling average (longer trend)
- `roll7_total_std`: 7-day rolling volatility
- `roll14_total_std`: 14-day rolling volatility

#### Change Features
- `delta1_total`: Day-to-day change
- `pct_change1_total`: Day-to-day percentage change
- `weekly_change_total`: Week-over-week change

### 5. 🏠 **Household Features** (Socio-Economic)

**Purpose**: Capture household characteristics and socio-economic factors affecting consumption.

#### ACORN Group Features
- `acorn_avg_consumption`: Average consumption for ACORN group
- `acorn_consumption_ratio`: Household vs group average ratio
- `acorn_peak_ratio`: Group-specific peak behavior
- `acorn_variability`: Group-specific consumption variability
- `relative_variability`: Household vs group variability

#### Individual Household Characteristics
- `hh_avg_consumption`: Household historical average
- `hh_std_consumption`: Household consumption variability
- `hh_max_consumption`: Household maximum consumption
- `hh_min_consumption`: Household minimum consumption
- `daily_vs_hh_avg`: Daily vs household average ratio
- `daily_vs_hh_max`: Daily vs household maximum ratio

### 6. 🔗 **Interaction Features** (Advanced)

**Purpose**: Capture complex interactions between different feature categories.

#### Weather-Temporal Interactions
- `weekend_heating`: Weekend heating demand
- `weekday_heating`: Weekday heating demand  
- `summer_cooling`: Summer cooling demand

#### Holiday-Consumption Interactions
- `holiday_consumption_boost`: Holiday consumption effects

---

## 🎯 Feature Groups for Model Interpretation

The features are organized into logical groups for SHAP analysis and model interpretation:

```python
feature_groups = {
    'consumption_basic': ['total_kwh', 'mean_kwh', 'peak_kwh', ...],
    'time_of_day': ['morning_kwh', 'afternoon_kwh', 'evening_kwh', ...],
    'consumption_patterns': ['peak_to_mean_ratio', 'load_factor', ...],
    'weather': ['temp_avg', 'heating_degree_days', 'cooling_degree_days', ...],
    'temporal': ['dayofweek', 'is_weekend', 'month', 'season', ...],
    'time_series': ['lag1_total', 'lag7_total', 'roll7_total_mean', ...],
    'household': ['acorn_avg_consumption', 'hh_avg_consumption', ...]
}
```

---

## 🚀 Usage Examples

### Basic Feature Creation

```python
from src.features.feature_pipeline import create_comprehensive_features
from src.data.data_loader import load_all_raw_data
from src.data.data_cleaner import clean_and_merge_all_data

# Load and clean data
raw_data = load_all_raw_data("path/to/data")
df_clean = clean_and_merge_all_data(raw_data)

# Create comprehensive features
df_features = create_comprehensive_features(df_clean)
print(f"Final dataset shape: {df_features.shape}")
```

### Prepare for Modeling

```python
from src.features.feature_pipeline import prepare_forecasting_data

# Prepare train/test split with feature groups
train_df, test_df, feature_cols, target_col, feature_groups = prepare_forecasting_data(
    df_features, 
    target_col="total_kwh",
    test_start="2014-01-01"
)

print(f"Features: {len(feature_cols)}")
print(f"Feature groups: {list(feature_groups.keys())}")
```

### Feature Importance Analysis (Stage 0)

```python
import xgboost as xgb
import shap

# Train XGBoost for feature importance
model = xgb.XGBRegressor(random_state=42)
model.fit(train_df[feature_cols], train_df[target_col])

# SHAP analysis by feature group
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_df[feature_cols])

for group_name, group_features in feature_groups.items():
    group_importance = np.abs(shap_values[:, [feature_cols.index(f) for f in group_features if f in feature_cols]]).mean()
    print(f"{group_name}: {group_importance:.4f}")
```

---

## 📈 Research Applications

### Stage 0: Feature Importance & EDA
- **XGBoost + SHAP**: Use all ~80 features for comprehensive analysis
- **Archetype Discovery**: Use consumption patterns for clustering
- **Driver Analysis**: Feature groups enable targeted interpretation

### Stage 1: Day-Ahead Forecasting
- **Statistical Models**: Use lag features, seasonality, weather
- **ML Models**: Full feature set with feature selection
- **Deep Learning**: Time series features + embeddings for categorical

### Stage 2: Multi-Horizon Forecasting  
- **Recursive Strategy**: Use 1-day model iteratively
- **Direct Strategy**: Extended lags (lag14) and rolling windows
- **Ensemble Methods**: Combine feature groups with different models

---

## 🔧 Technical Notes

### Data Requirements
- **Half-hourly consumption data**: 48 columns (hh_0 to hh_47)
- **Weather data**: Daily temperature, humidity, wind, cloud cover
- **Calendar data**: Dates for temporal feature extraction
- **Household data**: ACORN classifications
- **Holiday data**: UK bank holidays

### Feature Engineering Pipeline
1. **Data Loading**: Raw data ingestion
2. **Data Cleaning**: Missing value handling, outlier detection
3. **Feature Creation**: Comprehensive feature generation
4. **Feature Selection**: Remove redundant/correlated features
5. **Model Preparation**: Train/test splits, feature grouping

### Performance Considerations
- **Memory Usage**: ~80 features × large dataset requires optimization
- **Computation Time**: Rolling windows and groupby operations are intensive
- **Feature Selection**: Use correlation analysis and feature importance for pruning

---

## 📚 References

- **UK Energy Patterns**: Ofgem electricity consumption profiles
- **Weather Impact**: Heating/cooling degree day standards
- **Time Series**: Best practices for lag and rolling window features
- **Smart Meter Research**: London smart meter dataset documentation

---

## 🎉 Summary

This feature engineering system provides a **comprehensive foundation** for electricity consumption forecasting research. With **80+ carefully designed features** organized into logical groups, it supports all stages of the research framework from exploratory analysis to production forecasting models.

**Key Strengths**:
✅ Comprehensive coverage of consumption, weather, temporal, and household factors  
✅ Support for multiple forecasting horizons and model types  
✅ Organized feature groups for interpretable analysis  
✅ Production-ready pipeline with proper data handling  
✅ Aligned with UK electricity market characteristics 