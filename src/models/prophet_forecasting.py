"""
📈 PROPHET FORECASTING - Model Building and Prediction
====================================================

Prophet model creation, training, and prediction functions.
Pure model building without evaluation - evaluation functions moved to evaluation folder.

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def prepare_prophet_data(df: pd.DataFrame, 
                        target_col: str = "total_kwh",
                        household_id: str = None) -> pd.DataFrame:
    """
    Prepare data for Prophet forecasting
    
    Args:
        df: Input dataframe with day and target columns
        target_col: Target variable for forecasting
        household_id: Optional specific household ID to filter for
        
    Returns:
        Prophet-formatted dataframe with 'ds' and 'y' columns, and regressor columns list
    """
    prophet_df = df.copy()
    
    # Filter for specific household if provided
    if household_id:
        prophet_df = prophet_df[prophet_df['LCLid'] == household_id].copy()
    
    # Ensure day column is datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['day'])
    prophet_df['y'] = prophet_df[target_col]
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds')
    
    # Keep only Prophet required columns and useful regressors
    prophet_cols = ['ds', 'y']
    
    # Add regressors if available
    regressor_cols = []
    for col in ['temp_mean', 'humidity', 'heating_degree_days', 'cooling_degree_days',
                'is_weekend', 'is_holiday', 'month', 'dayofweek']:
        if col in prophet_df.columns:
            prophet_cols.append(col)
            regressor_cols.append(col)
    
    prophet_df = prophet_df[prophet_cols].dropna()
    
    print(f"📊 Prophet data prepared: {len(prophet_df)} days")
    if regressor_cols:
        print(f"📊 External regressors: {regressor_cols}")
    
    return prophet_df, regressor_cols

def create_prophet_model(include_yearly: bool = True,
                        include_weekly: bool = True,
                        include_daily: bool = False,
                        holidays_prior_scale: float = 10.0,
                        seasonality_prior_scale: float = 10.0,
                        changepoint_prior_scale: float = 0.05) -> Prophet:
    """
    Create and configure Prophet model
    
    Args:
        include_yearly: Include yearly seasonality
        include_weekly: Include weekly seasonality  
        include_daily: Include daily seasonality (False for daily data)
        holidays_prior_scale: Strength of holiday effects
        seasonality_prior_scale: Strength of seasonality effects
        changepoint_prior_scale: Strength of trend changes
        
    Returns:
        Configured Prophet model
    """
    model = Prophet(
        yearly_seasonality=include_yearly,
        weekly_seasonality=include_weekly,
        daily_seasonality=include_daily,
        holidays_prior_scale=holidays_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95
    )
    
    print(f"✅ Prophet model created with yearly={include_yearly}, weekly={include_weekly}")
    return model

def add_prophet_regressors(model: Prophet, regressor_cols: list):
    """
    Add external regressors to Prophet model
    
    Args:
        model: Prophet model
        regressor_cols: List of regressor column names
    """
    for regressor in regressor_cols:
        model.add_regressor(regressor)
        print(f"📊 Added regressor: {regressor}")

def fit_prophet_model(model: Prophet, 
                     train_df: pd.DataFrame,
                     regressor_cols: list = None) -> Prophet:
    """
    Fit Prophet model on training data
    
    Args:
        model: Prophet model
        train_df: Training dataframe with 'ds' and 'y' columns
        regressor_cols: List of external regressors
        
    Returns:
        Fitted Prophet model
    """
    print("🚀 Training Prophet model...")
    
    # Add regressors if provided
    if regressor_cols:
        add_prophet_regressors(model, regressor_cols)
    
    # Fit model
    model.fit(train_df)
    
    print("✅ Prophet model trained successfully")
    return model

def forecast_prophet(model: Prophet,
                    forecast_days: int,
                    last_date: str,
                    future_regressors: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate Prophet forecasts
    
    Args:
        model: Fitted Prophet model
        forecast_days: Number of days to forecast
        last_date: Last date in training data
        future_regressors: Future values for external regressors
        
    Returns:
        Dataframe with forecasts
    """
    print(f"🔮 Generating {forecast_days}-day forecast...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Add future regressor values if provided
    if future_regressors is not None:
        # Merge future regressors
        future_start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        future_end = future_start + pd.Timedelta(days=forecast_days-1)
        
        future_period = future_regressors[
            (future_regressors['ds'] >= future_start) & 
            (future_regressors['ds'] <= future_end)
        ]
        
        # Fill missing regressor values in future dataframe
        for col in future_period.columns:
            if col != 'ds' and col in future.columns:
                future.loc[future['ds'].isin(future_period['ds']), col] = future_period[col].values
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Extract forecast period only
    forecast_period = forecast.tail(forecast_days).copy()
    
    print(f"✅ Forecast generated for {len(forecast_period)} days")
    return forecast_period

def prophet_day_ahead_forecast(train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              target_col: str = "total_kwh",
                              household_id: str = None,
                              include_regressors: bool = True) -> dict:
    """
    Complete day-ahead forecasting pipeline with Prophet
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe  
        target_col: Target variable for forecasting
        household_id: Optional specific household ID
        include_regressors: Whether to include external regressors
        
    Returns:
        Dictionary with model, forecasts, test data, and metadata
    """
    print("🚀 PROPHET DAY-AHEAD FORECASTING")
    print("=" * 40)
    
    # Prepare Prophet data
    train_prophet, regressor_cols = prepare_prophet_data(
        train_df, target_col, household_id
    )
    test_prophet, _ = prepare_prophet_data(
        test_df, target_col, household_id
    )
    
    if not include_regressors:
        regressor_cols = []
        train_prophet = train_prophet[['ds', 'y']]
        test_prophet = test_prophet[['ds', 'y']]
    
    # Create and fit model
    model = create_prophet_model()
    model = fit_prophet_model(model, train_prophet, regressor_cols)
    
    # Generate forecasts for test period
    forecast_days = len(test_prophet)
    last_train_date = train_prophet['ds'].max()
    
    # Prepare future regressors if needed
    future_regressors = None
    if include_regressors and regressor_cols:
        future_regressors = test_prophet[['ds'] + regressor_cols]
    
    forecast = forecast_prophet(
        model, forecast_days, last_train_date, future_regressors
    )
    
    # Prepare results
    results = {
        'model': model,
        'forecast': forecast,
        'test_data': test_prophet,
        'regressor_cols': regressor_cols,
        'forecast_days': forecast_days,
        'y_true': test_prophet['y'].values,
        'y_pred': forecast['yhat'].values,
        'dates': test_prophet['ds'].values
    }
    
    print("✅ Prophet forecasting completed")
    return results

if __name__ == "__main__":
    print("📈 Prophet Forecasting Module")
    print("Usage: from src.models.prophet_forecasting import prophet_day_ahead_forecast") 