
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_layer = layers.Dense(1)  

    def call(self, inputs):
        score = self.score_layer(inputs)   # [batch, time, 1]
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context


class LSTMForecaster:
    def __init__(
        self,
        seq_length=14,
        n_features=None,
        hidden_units=[128, 64],
        dropout=0.2,
        learning_rate=0.001,
        use_embedding=False,
        embedding_dim=8,
        use_cluster_embedding=False,
        cluster_embedding_dim=4,
        n_clusters=None,
        use_attention=True,
        scale_features=True,
        scale_target=True,
        random_state=42
    ):
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        self.use_cluster_embedding = use_cluster_embedding
        self.cluster_embedding_dim = cluster_embedding_dim
        self.n_clusters = n_clusters
        self.use_attention = use_attention
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.random_state = random_state

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.model = None
        self.feature_scaler = StandardScaler() if self.scale_features else None
        self.target_scaler = StandardScaler() if self.scale_target else None

    def build_model(self, n_features=None, n_households=None, n_clusters=None):
        if n_features is not None:
            self.n_features = n_features
        if n_households is not None:
            self.n_households = n_households
        if n_clusters is not None:
            self.n_clusters = n_clusters

        sequence_input = layers.Input(shape=(self.seq_length, self.n_features), name="sequence_input")
        x = sequence_input
        inputs = [sequence_input]

        if self.use_embedding and self.n_households is not None:
            hh_input = layers.Input(shape=(1,), name="household_input")
            hh_embed = layers.Embedding(self.n_households, self.embedding_dim)(hh_input)
            hh_embed = layers.Flatten()(hh_embed)
            hh_embed = layers.RepeatVector(self.seq_length)(hh_embed)
            x = layers.Concatenate()([x, hh_embed])
            inputs.append(hh_input)

        if self.use_cluster_embedding and self.n_clusters is not None:
            cl_input = layers.Input(shape=(1,), name="cluster_input")
            cl_embed = layers.Embedding(self.n_clusters, self.cluster_embedding_dim)(cl_input)
            cl_embed = layers.Flatten()(cl_embed)
            cl_embed = layers.RepeatVector(self.seq_length)(cl_embed)
            x = layers.Concatenate()([x, cl_embed])
            inputs.append(cl_input)

        for i, units in enumerate(self.hidden_units):
            return_sequences = self.use_attention or (i < len(self.hidden_units) - 1)
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        if self.use_attention:
            x = TemporalAttention()(x)

        # Add learnable bias per cluster
        if self.use_cluster_embedding and self.n_clusters is not None:
            bias = layers.Embedding(self.n_clusters, 1, name="cluster_bias")(cl_input)
            bias = layers.Flatten()(bias)
            x = layers.Dense(1)(x)
            output = layers.Add(name="bias_corrected_output")([x, bias])
        else:
            output = layers.Dense(1)(x)

        self.model = keras.Model(inputs=inputs, outputs=output)

        def coeff_determination(y_true, y_pred):
            SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae", coeff_determination]
        )
        return self.model

    def _prepare_data(self, X, y=None, fit_scalers=False):
        X_scaled = X.copy()
        if self.scale_features and self.feature_scaler:
            X_reshaped = X.reshape(-1, X.shape[-1])
            if fit_scalers:
                X_scaled = self.feature_scaler.fit_transform(X_reshaped).reshape(X.shape)
            else:
                X_scaled = self.feature_scaler.transform(X_reshaped).reshape(X.shape)

        if y is not None:
            y_scaled = y.copy()
            if self.scale_target and self.target_scaler:
                if fit_scalers:
                    y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
                else:
                    y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).ravel()
            return X_scaled, y_scaled
        return X_scaled

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            household_train=None, household_val=None,
            cluster_train=None, cluster_val=None,
            epochs=50, batch_size=64, early_stopping=True, patience=10, verbose=1):

        if self.model is None:
            self.build_model(
                n_features=X_train.shape[-1],
                n_households=len(np.unique(household_train)) if household_train is not None else None,
                n_clusters=len(np.unique(cluster_train)) if cluster_train is not None else None
            )

        X_train_scaled, y_train_scaled = self._prepare_data(X_train, y_train, fit_scalers=True)
        train_inputs = [X_train_scaled]
        if self.use_embedding:
            train_inputs.append(household_train)
        if self.use_cluster_embedding:
            train_inputs.append(cluster_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_scaled = self._prepare_data(X_val, y_val, fit_scalers=False)
            val_inputs = [X_val_scaled]
            if self.use_embedding:
                val_inputs.append(household_val)
            if self.use_cluster_embedding:
                val_inputs.append(cluster_val)
            validation_data = (val_inputs, y_val_scaled)

        callbacks_list = []
        if early_stopping and validation_data:
            callbacks_list.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True, verbose=verbose))
            callbacks_list.append(keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=patience//2, verbose=verbose))

        self.history = self.model.fit(
            train_inputs, y_train_scaled,
            epochs=epochs, batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list, verbose=verbose
        )
        return self

    def predict(self, X, household_codes=None, cluster_codes=None):
        X_scaled = self._prepare_data(X, fit_scalers=False)
        inputs = [X_scaled]
        if self.use_embedding:
            inputs.append(household_codes)
        if self.use_cluster_embedding:
            inputs.append(cluster_codes)
        y_pred_scaled = self.model.predict(inputs, verbose=0).ravel()
        if self.scale_target and self.target_scaler:
            return self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred_scaled

    def evaluate(self, X, y, household_codes=None, cluster_codes=None):
        y_pred = self.predict(X, household_codes, cluster_codes)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        bias = np.mean(y_pred - y)
        mape = np.mean(np.abs((y - y_pred) / np.clip(y, 1e-6, None))) * 100
        return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "mape": mape}

    def get_model_summary(self):
        return self.model.summary() if self.model else "Model not built yet."
