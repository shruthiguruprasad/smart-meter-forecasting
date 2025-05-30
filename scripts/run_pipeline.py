"""
🚀 MAIN PIPELINE - Smart Meter Forecasting Pipeline
===================================================

Complete pipeline for smart meter consumption forecasting.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import SmartMeterDataLoader
from data.data_merger import SmartMeterDataMerger
from features.consumption_features import ConsumptionFeatureEngineer
from models.prophet_model import ProphetForecaster
from models.base_model import ModelEvaluator, TimeSeriesValidator

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config: dict):
    """Create necessary directories"""
    dirs_to_create = [
        config['paths']['data_dir'],
        config['paths']['models_dir'],
        config['paths']['results_dir'],
        config['paths']['plots_dir']
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_prepare_data(config: dict, logger) -> pd.DataFrame:
    """
    Load and prepare smart meter data
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Prepared dataset
    """
    logger.info("🚀 Starting data loading and preparation...")
    
    # Initialize data loader
    loader = SmartMeterDataLoader(data_path=config['data']['path'])
    
    # Load and clean consumption data
    df_consumption = loader.load_and_clean(
        missing_threshold=config['data']['missing_threshold'],
        min_days_per_household=config['data']['min_days_per_household']
    )
    
    # Merge external data
    merger = SmartMeterDataMerger(data_path=config['data']['path'])
    df_merged = merger.merge_all_external_data(df_consumption)
    
    # Print summary
    loader.print_data_summary(df_merged)
    merger.print_merge_summary(df_merged)
    
    return df_merged

def engineer_features(df: pd.DataFrame, config: dict, logger) -> tuple:
    """
    Engineer features for forecasting
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Feature matrix and target matrix
    """
    logger.info("⚡ Starting feature engineering...")
    
    # Initialize feature engineer
    feature_engineer = ConsumptionFeatureEngineer()
    
    # Create consumption features
    df_features = feature_engineer.create_all_consumption_features(df)
    
    # Create targets for multiple horizons
    targets = pd.DataFrame(index=df.index)
    for horizon in config['features']['horizons']:
        target_col = f"target_{horizon}d"
        targets[target_col] = df_features['total_kwh'].shift(-horizon)
    
    # Select feature columns (exclude half-hourly and target columns)
    hh_cols = [f"hh_{i}" for i in range(48)]
    target_cols = [f"target_{h}d" for h in config['features']['horizons']]
    
    feature_cols = [col for col in df_features.columns 
                   if col not in hh_cols + target_cols + ['LCLid', 'day']]
    
    X = df_features[feature_cols].copy()
    y = targets.copy()
    
    # Remove rows with missing targets
    complete_rows = y.notna().all(axis=1)
    X = X[complete_rows]
    y = y[complete_rows]
    
    logger.info(f"✅ Feature engineering completed: {X.shape[1]} features, {len(X)} samples")
    
    return X, y

def train_models(X: pd.DataFrame, y: pd.DataFrame, config: dict, logger) -> dict:
    """
    Train forecasting models
    
    Args:
        X: Feature matrix
        y: Target matrix
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Dictionary of trained models
    """
    logger.info("🏗️ Starting model training...")
    
    models = {}
    horizons = config['features']['horizons']
    
    # Split data
    test_size = int(len(X) * config['evaluation']['test_size'])
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    # Train Prophet model
    logger.info("📊 Training Prophet model...")
    prophet_model = ProphetForecaster(
        horizons=horizons,
        **config['models']['prophet']
    )
    prophet_model.fit(X_train, y_train)
    models['prophet'] = prophet_model
    
    # Evaluate models
    logger.info("📈 Evaluating models...")
    evaluator = ModelEvaluator()
    
    for model_name, model in models.items():
        results = evaluator.evaluate_model(model, X_test, y_test)
        logger.info(f"🎯 {model_name.upper()} Results:")
        for horizon, metrics in results.items():
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            logger.info(f"   {horizon}: RMSE={rmse:.3f}, MAE={mae:.3f}")
    
    return models, (X_train, X_test, y_train, y_test)

def save_results(models: dict, evaluation_data: tuple, config: dict, logger):
    """Save model results and predictions"""
    logger.info("💾 Saving results...")
    
    X_train, X_test, y_train, y_test = evaluation_data
    
    # Save predictions
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(X_test)
        predictions[model_name] = pred
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config['paths']['results_dir'])
    
    for model_name, pred in predictions.items():
        output_path = results_dir / f"{model_name}_predictions_{timestamp}.csv"
        pred.to_csv(output_path)
        logger.info(f"   📄 Saved {model_name} predictions to {output_path}")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Smart Meter Forecasting Pipeline")
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("🔌 Starting Smart Meter Forecasting Pipeline")
    
    try:
        # Create directories
        create_directories(config)
        
        # Step 1: Load and prepare data
        df = load_and_prepare_data(config, logger)
        
        # Step 2: Engineer features
        X, y = engineer_features(df, config, logger)
        
        # Step 3: Train models
        models, evaluation_data = train_models(X, y, config, logger)
        
        # Step 4: Save results
        save_results(models, evaluation_data, config, logger)
        
        logger.info("✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 