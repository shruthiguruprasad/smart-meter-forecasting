# 🔌 Smart Meter Forecasting - Multi-Horizon Electricity Consumption Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements a comprehensive **multi-horizon electricity consumption forecasting framework** using the "Smart Meters in London" dataset. The framework predicts daily electricity consumption at multiple time horizons (1 day, 1 week, 1 month) for both individual households and aggregate network demand.

## 🏗️ Project Structure

```
smart-meter-forecasting/
├── 📂 src/                          # Source code
│   ├── 📂 data/                     # Data loading and processing
│   │   ├── data_loader.py           # Smart meter data loading & cleaning
│   │   └── data_merger.py           # External data integration
│   ├── 📂 features/                 # Feature engineering
│   │   ├── consumption_features.py  # Consumption pattern features
│   │   ├── temporal_features.py     # Time-based features
│   │   ├── weather_features.py      # Weather-related features
│   │   └── feature_pipeline.py      # Complete feature pipeline
│   ├── 📂 models/                   # Forecasting models
│   │   ├── base_model.py            # Abstract base class
│   │   ├── prophet_model.py         # Facebook Prophet
│   │   ├── xgboost_model.py         # XGBoost regressor
│   │   ├── lightgbm_model.py        # LightGBM
│   │   ├── lstm_model.py            # LSTM neural network
│   │   └── ensemble_model.py        # Model ensemble
│   ├── 📂 evaluation/               # Model evaluation
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── model_comparison.py      # Model comparison tools
│   │   └── validation.py            # Cross-validation
│   ├── 📂 visualization/            # Data visualization
│   │   ├── clustering_plots.py      # Load shape clustering
│   │   ├── forecasting_plots.py     # Forecast visualization
│   │   └── analysis_plots.py        # EDA and analysis plots
│   └── 📂 utils/                    # Utilities
│       ├── helpers.py               # Helper functions
│       └── constants.py             # Project constants
├── 📂 config/                       # Configuration files
│   └── config.yaml                  # Main configuration
├── 📂 scripts/                      # Execution scripts
│   ├── run_pipeline.py              # Main pipeline script
│   ├── train_models.py              # Model training script
│   └── evaluate_models.py           # Model evaluation script
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Data exploration
│   ├── 02_feature_engineering.ipynb # Feature analysis
│   ├── 03_model_development.ipynb   # Model development
│   └── 04_results_analysis.ipynb    # Results analysis
├── 📂 tests/                        # Unit tests
├── 📂 docs/                         # Documentation
├── 📂 data/                         # Data directories
├── 📂 models/                       # Saved models
├── 📂 results/                      # Results and outputs
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

## 🎯 Research Questions

1. **RQ3**: How accurately can day-ahead consumption be forecasted using statistical vs. ML approaches?
2. **RQ4**: Do cluster-aware hybrid models improve forecasting performance?
3. **RQ5**: What intervention strategies emerge from load-shape archetype analysis?
4. **RQ6**: How do weather variations affect consumption patterns across different household segments?

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-meter-forecasting.git
cd smart-meter-forecasting

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Configuration

Edit `config/config.yaml` to set your data path and model parameters:

```yaml
data:
  path: "/path/to/smart-meters-in-london/"
  missing_threshold: 0.2
  min_days_per_household: 30

models:
  prophet:
    seasonality_mode: "multiplicative"
  xgboost:
    n_estimators: 1000
    max_depth: 6
```

### 3. Run the Pipeline

```bash
# Run the complete pipeline
python scripts/run_pipeline.py --config config/config.yaml

# Or run individual components
python scripts/train_models.py --model prophet
python scripts/evaluate_models.py --models prophet xgboost
```

## 📊 Models Implemented

### Statistical Models
- **Facebook Prophet**: Time series with seasonality and external regressors
- **ARIMA**: Auto-regressive integrated moving average

### Machine Learning Models  
- **XGBoost**: Gradient boosting with feature importance
- **LightGBM**: Fast gradient boosting
- **Random Forest**: Ensemble of decision trees

### Deep Learning Models
- **LSTM**: Long short-term memory networks
- **Transformer**: Attention-based sequence models

### Ensemble Methods
- **Weighted Average**: Performance-based weighting
- **Stacking**: Meta-learner approach

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
from src.models.prophet_model import ProphetForecaster

# Load data
loader = SmartMeterDataLoader("/path/to/data/")
df = loader.load_and_clean()

# Train model
model = ProphetForecaster(horizons=[1, 7, 30])
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Advanced Pipeline

```python
from scripts.run_pipeline import main

# Run complete pipeline with custom config
main()
```

## 📊 Clustering Analysis

The framework includes load shape clustering to identify household archetypes:

```python
from src.visualization.clustering_plots import create_clustering_plots

# Generate clustering visualizations
create_clustering_plots(df, n_clusters=8)
```

## 🔬 Research Implementation

### Stage 0: Exploratory & Predictive Modeling
- Feature importance analysis using XGBoost + SHAP
- Load shape archetype identification

### Stage 1: Day-ahead Forecasting
- Baseline models (ARIMA, Prophet)
- Advanced models (XGBoost, Transformers, LSTM)
- Performance comparison across household segments

### Stage 2: Multi-horizon Extension
- Direct vs. recursive forecasting strategies
- Uncertainty quantification
- Intervention strategy development

## 📚 Documentation

- **API Documentation**: `docs/api/`
- **Tutorials**: `docs/tutorials/`
- **Research Questions**: `docs/research_questions.md`

## 🧪 Testing

Run tests to ensure code quality:

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

**Shruthi Simha Chippagiri**  
MSc Energy Transition and Data Science  
University of Otago, Dunedin, NZ

---

*Built with ❤️ for advancing smart grid forecasting research* 
