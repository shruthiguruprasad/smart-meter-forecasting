"""
src/models/lstm_model.py

TensorFlow/Keras-based LSTM model for electricity consumption forecasting.
Supports global forecasting across multiple households with configurable architecture.

Usage:
    from lstm_model import LSTMForecaster
    model = LSTMForecaster(seq_length=14, n_features=20)
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)

Author: Shruthi Simha Chippagiri
Date: 2025
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


class LSTMForecaster:
    """
    LSTM-based forecasting model for electricity consumption prediction.
    
    Supports:
    - Multi-step sequence input
    - Global training across households
    - Configurable LSTM architecture
    - Optional feature scaling
    - Early stopping and callbacks
    """
    
    def __init__(
        self,
        seq_length: int = 14,
        n_features: int = None,
        hidden_units: list = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        use_embedding: bool = False,
        embedding_dim: int = 8,
        scale_features: bool = True,
        scale_target: bool = True,
        random_state: int = 42
    ):
        """
        Initialize LSTM forecaster.
        
        Parameters
        ----------
        seq_length : int, default 14
            Number of time steps in input sequences.
        n_features : int, optional
            Number of input features. If None, will be inferred from data.
        hidden_units : list, default [128, 64]
            List of hidden units for each LSTM layer.
        dropout : float, default 0.2
            Dropout rate for regularization.
        learning_rate : float, default 0.001
            Learning rate for Adam optimizer.
        use_embedding : bool, default False
            Whether to use household embedding (for household_code feature).
        embedding_dim : int, default 8
            Dimension of household embedding if used.
        scale_features : bool, default True
            Whether to scale input features.
        scale_target : bool, default True
            Whether to scale target values.
        random_state : int, default 42
            Random seed for reproducibility.
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Initialize components
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.history = None
        self.n_households = None
        
        # Initialize scalers if needed
        if self.scale_features:
            self.feature_scaler = StandardScaler()
        if self.scale_target:
            self.target_scaler = StandardScaler()
    
    def build_model(self, n_features: int = None, n_households: int = None):
        """
        Build the LSTM model architecture.
        
        Parameters
        ----------
        n_features : int, optional
            Number of input features. Uses self.n_features if not provided.
        n_households : int, optional
            Number of unique households for embedding.
        """
        if n_features is not None:
            self.n_features = n_features
        if n_households is not None:
            self.n_households = n_households
            
        if self.n_features is None:
            raise ValueError("n_features must be specified either in __init__ or build_model")
        
        # Main sequence input
        sequence_input = layers.Input(
            shape=(self.seq_length, self.n_features), 
            name='sequence_input'
        )
        x = sequence_input
        
        # Optional household embedding 🔧
        use_embed = self.use_embedding and (self.n_households is not None)
        if use_embed:
            household_input = layers.Input(shape=(1,), name='household_input')
            household_embed = layers.Embedding(
                input_dim=self.n_households,
                output_dim=self.embedding_dim,
                name='household_embedding'
            )(household_input)
            household_embed = layers.Flatten()(household_embed)
            
            # Repeat embedding for each time step
            household_repeat = layers.RepeatVector(self.seq_length)(household_embed)
            household_repeat = layers.Reshape((self.seq_length, self.embedding_dim))(household_repeat)
            
            # Concatenate with main input
            x = layers.Concatenate(axis=-1)([x, household_repeat])
            inputs = [sequence_input, household_input]
        else:
            inputs = sequence_input
        
        # LSTM layers
        for i, units in enumerate(self.hidden_units):
            return_sequences = (i < len(self.hidden_units) - 1)
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=0.0,               # 🔧 turn off internal dropout
                recurrent_dropout=0.0,     # 🔧 turn off internal recurrent dropout
                name=f'lstm_{i+1}'
            )(x)
            
            # Apply standalone Dropout if needed 🔧
            if self.dropout > 0 and i < len(self.hidden_units):
                x = layers.Dropout(self.dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=output, name='LSTMForecaster')
        
        # Compile model (with MAE and optional R2) 🔧
        def coeff_determination(y_true, y_pred):
            SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
            return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', coeff_determination]
        )
        
        return self.model
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None, fit_scalers: bool = False):
        """
        Prepare data by scaling if configured.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_length, n_features).
        y : np.ndarray, optional
            Target values of shape (n_samples,).
        fit_scalers : bool, default False
            Whether to fit the scalers (True for training data).
            
        Returns
        -------
        X_scaled : np.ndarray
            Scaled input sequences.
        y_scaled : np.ndarray, optional
            Scaled target values (if y provided).
        """
        X_scaled = X.copy()
        
        # Scale features if configured
        if self.scale_features and self.feature_scaler is not None:
            n_samples, seq_length, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            
            if fit_scalers:
                X_scaled_reshaped = self.feature_scaler.fit_transform(X_reshaped)
            else:
                X_scaled_reshaped = self.feature_scaler.transform(X_reshaped)
            
            X_scaled = X_scaled_reshaped.reshape(n_samples, seq_length, n_features)
        
        # Scale target if provided and configured
        if y is not None:
            y_scaled = y.copy()
            if self.scale_target and self.target_scaler is not None:
                y_reshaped = y.reshape(-1, 1)
                if fit_scalers:
                    y_scaled_reshaped = self.target_scaler.fit_transform(y_reshaped)
                else:
                    y_scaled_reshaped = self.target_scaler.transform(y_reshaped)
                y_scaled = y_scaled_reshaped.ravel()
            
            return X_scaled, y_scaled
        
        return X_scaled
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        household_train: np.ndarray = None,
        household_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 64,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 1
    ):
        """
        Train the LSTM model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training sequences of shape (n_samples, seq_length, n_features).
        y_train : np.ndarray
            Training targets of shape (n_samples,).
        X_val : np.ndarray, optional
            Validation sequences.
        y_val : np.ndarray, optional
            Validation targets.
        household_train : np.ndarray, optional
            Household codes for training (if using embedding).
        household_val : np.ndarray, optional
            Household codes for validation (if using embedding).
        epochs : int, default 50
            Maximum number of training epochs.
        batch_size : int, default 64
            Batch size for training.
        early_stopping : bool, default True
            Whether to use early stopping.
        patience : int, default 10
            Patience for early stopping.
        verbose : int, default 1
            Verbosity level.
            
        Returns
        -------
        self : LSTMForecaster
            The fitted model.
        """
        # Build model if not already built
        if self.model is None:
            n_features = X_train.shape[2]
            n_households = (
                len(np.unique(household_train)) 
                if (self.use_embedding and household_train is not None) 
                else None
            )
            self.build_model(n_features=n_features, n_households=n_households)
        
        # Prepare training data
        X_train_scaled, y_train_scaled = self._prepare_data(X_train, y_train, fit_scalers=True)
        
        # Prepare inputs
        if self.use_embedding and household_train is not None:
            train_inputs = [X_train_scaled, household_train]
        else:
            train_inputs = X_train_scaled
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_scaled = self._prepare_data(X_val, y_val, fit_scalers=False)
            
            if self.use_embedding and household_val is not None:
                val_inputs = [X_val_scaled, household_val]
            else:
                val_inputs = X_val_scaled
            
            validation_data = (val_inputs, y_val_scaled)
        
        # Prepare callbacks
        callbacks_list = []
        if early_stopping and validation_data is not None:
            callbacks_list.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose
                )
            )
            # 🔧 Optional: add ReduceLROnPlateau
            callbacks_list.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=max(1, patience // 2),
                    verbose=verbose
                )
            )
        
        # Train model
        self.history = self.model.fit(
            train_inputs,
            y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray, household_codes: np.ndarray = None):
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_length, n_features).
        household_codes : np.ndarray, optional
            Household codes (if using embedding).
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values of shape (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scalers=False)
        
        # Prepare inputs
        if self.use_embedding and household_codes is not None:
            inputs = [X_scaled, household_codes]
        else:
            inputs = X_scaled
        
        # Make predictions
        y_pred_scaled = self.model.predict(inputs, verbose=0).ravel()
        
        # Inverse transform if scaling was used
        if self.scale_target and self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, household_codes: np.ndarray = None):
        """
        Evaluate the model on given data.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences.
        y : np.ndarray
            True target values.
        household_codes : np.ndarray, optional
            Household codes (if using embedding).
            
        Returns
        -------
        metrics : dict
            Dictionary containing evaluation metrics.
        """
        y_pred = self.predict(X, household_codes)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'bias': np.mean(y_pred - y)
        }
        
        # MAPE (handle division by zero)
        mask = y != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        return metrics
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet."
        return self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        return self


if __name__ == "__main__":
    print("✔️ lstm_model.py loaded.")
    print("   Use LSTMForecaster() to create and train LSTM models.")
