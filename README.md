# Time-Series-Forecasting-with-LSTM
Predicts future values of a time series using LSTM.
import tensorflow as tf
import numpy as np

# Generate synthetic time series data
time = np.arange(0, 100, 0.1)
data = np.sin(time) + np.random.normal(0, 0.1, len(time))

# Prepare data for LSTM
def create_dataset(series, window_size):
    x, y = [], []
    for i in range(len(series) - window_size):
        x.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(x), np.array(y)

window_size = 10
x, y = create_dataset(data, window_size)
x = np.expand_dims(x, axis=-1)

# Define LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(window_size, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, batch_size=32)
