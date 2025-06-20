# 🔌 Smart Meter Forecasting - Advanced Multi-Horizon Electricity Consumption Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📋 Overview

This project implements a **state-of-the-art multi-horizon electricity consumption forecasting framework** using the "Smart Meters in London" dataset. The framework combines advanced machine learning, deep learning, and statistical methods to predict daily electricity consumption at multiple time horizons (1 day, 1 week) for both individual households and aggregate network demand.

### 🌟 Key Features

- **🎯 Multi-Horizon Forecasting**: Simultaneous prediction at 1-day, 7-day, and 30-day horizons
- **🧠 Advanced ML/DL Models**: LSTM, XGBoost, LightGBM, Prophet, and ensemble methods
- **🏠 Household-Level Analytics**: Individual and aggregate consumption patterns
- **🌤️ Weather Integration**: Advanced weather feature engineering with heating/cooling demand
- **📊 Load Shape Clustering**: Automatic household archetype identification
- **🔧 Feature Engineering**: 50+ engineered features including temporal, consumption, and weather patterns

### 🎓 Research Context

This framework addresses critical research questions in smart grid analytics:

- **Energy Demand Forecasting**: How accurately can we predict consumption patterns?
- **Household Segmentation**: What are the key consumption archetypes?
- **Weather Impact Analysis**: How do weather patterns affect different household types?
- **Intervention Strategies**: What targeted demand response strategies emerge from the analysis?

## 🏗️ Project Structure

```bash
smart-meter-forecasting/
├── 📂 src/                          # Source code
│   ├── 📂 data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Smart meter data loading & cleaning
│   │   └── data_cleaner.py          # Data preprocessing utilities
│   ├── 📂 features/                 # Feature engineering
│   │   ├── __init__.py
│   │   ├── consumption_features.py  # Consumption pattern features
│   │   ├── temporal_features.py     # Time-based features
│   │   ├── weather_features.py      # Weather-related features
│   │   ├── feature_pipeline.py      # Complete feature pipeline
│   │   └── splitters.py             # Train/test splitting utilities
│   ├── 📂 models/                   # Forecasting models
│   │   ├── __init__.py
│   │   ├── lstm_model.py            # LSTM neural network
│   │   ├── lstm_cluster_model.py    # Cluster-aware LSTM
│   │   ├── xgboost_forecasting.py   # XGBoost implementation
│   │   └── predictive_model.py      # Base model interface
│   ├── 📂 evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   ├── forecast_evaluation.py   # Evaluation metrics & validation
│   │   └── feature_analysis.py      # Feature importance analysis
│   ├── 📂 visualization/            # Data visualization
│   │   ├── __init__.py
│   │   ├── clustering_plots.py      # Load shape clustering plots
│   │   ├── forecast_plots.py        # Forecast visualization
│   │   ├── feature_plots.py         # Feature analysis plots
│   │   └── 📂 clustering/           # Advanced clustering visualizations
│   │       ├── core.py
│   │       ├── load_shape.py
│   │       ├── socioeconomic.py
│   │       ├── stability.py
│   │       ├── temporal.py
│   │       └── weather_impact.py
│   └── 📂 utils/                    # Utilities
│       ├── __init__.py
│       ├── helpers.py               # Helper functions
│       ├── constants.py             # Project constants
│       ├── memory_utils.py          # Memory optimization
│       ├── sequence_builder.py      # Sequence generation for LSTM
│       └── visualization_helpers.py # Visualization utilities
├── 📂 config/                       # Configuration files
│   └── config.yaml                  # Main configuration
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── lstm-forecasting (1).ipynb  # LSTM development
│   └── predictive-modeling.ipynb    # Model comparison
├── 📂 docs/                         # Documentation
│   ├── README.md                    # Documentation index
│   ├── LSTM_FORECASTING.md          # LSTM guide
│   └── feature_engineering_guide.md # Feature engineering guide
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```


## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- Git
- 8GB+ RAM recommended
- GPU optional but recommended for LSTM training

## 🔧 Features Engineering

### Consumption Features
- Basic statistics (mean, std, peak, min)
- Time-of-day aggregations (morning, evening, night)
- Peak analysis (ratios, timing, sharpness)
- Variability measures (CV, Gini coefficient)
- Load factors (utilization, demand factor)

### Temporal Features
- Calendar features (day of week, month, holidays)
- Seasonal indicators (heating/cooling seasons)
- Holiday categorization

### Weather Features
- Temperature (min, max, degree days)
- Humidity, wind speed, cloud cover
- Weather-based heating/cooling demand

### Lag Features
- Lagged consumption values (1, 2, 3, 7, 14, 30 days)
- Rolling statistics (3, 7, 14, 30 day windows)

## 📈 Evaluation Framework

### Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Validation Strategy
- **Time Series Cross-Validation**: Respects temporal order
- **Multi-horizon evaluation**: 1-day, 7-day, 30-day forecasts
- **Household-level and aggregate-level assessment**

## 📱 Usage Examples

### Basic Usage

```python
from src.data.data_loader import SmartMeterDataLoader
from src.models.lstm_model import LSTMForecaster
from src.features.feature_pipeline import FeaturePipeline

# Load and preprocess data
loader = SmartMeterDataLoader("/path/to/data/")
df = loader.load_and_clean()

# Engineer features
feature_pipeline = FeaturePipeline()
X_train, X_test, y_train, y_test = feature_pipeline.fit_transform(df)

# Train LSTM model
model = LSTMForecaster(
    seq_length=14, 
    n_features=X_train.shape[1],
    horizons=[1, 7, 30]
)
model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)
```

### Cluster Analysis and Visualization

```python
from src.visualization.clustering_plots import ClusterAnalyzer

# Perform load shape clustering
analyzer = ClusterAnalyzer(n_clusters=8)
cluster_results = analyzer.fit_predict(df)

# Generate comprehensive visualizations
analyzer.create_cluster_profiles()
analyzer.plot_temporal_patterns()
analyzer.plot_weather_sensitivity()
```
