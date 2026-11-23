
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. GENERATE TWO YEARS OF DAILY ORDER DATA
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

days = 730

dates = pd.date_range(
    start="2024-01-01",
    periods=days,
    freq="D"
)

day_number = np.arange(days)

# Weekly pattern
weekly_pattern = 60 * np.sin(2 * np.pi * day_number / 7)

# Seasonal pattern
seasonal_pattern = 100 * np.sin(
    2 * np.pi * day_number / 365
)

# Gradual increase in orders
trend = 0.15 * day_number

# Festival demand increases
festival_effect = np.zeros(days)

festival_days = [50, 150, 250, 350, 450, 550, 650]

for festival_day in festival_days:
    festival_effect[festival_day:festival_day + 5] = 180

# Random variations
noise = np.random.normal(0, 25, days)

# Calculate daily orders
orders = (
    500
    + weekly_pattern
    + seasonal_pattern
    + trend
    + festival_effect
    + noise
)

orders = np.maximum(orders, 50)

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Orders": orders
})

print("\nFirst five rows of dataset:")
print(df.head())

# Plot original order data
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Orders"])

plt.title("Daily Food Delivery Orders")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. NORMALIZE DATA AND PREPARE SEQUENCES
# --------------------------------------------------

train_end = int(days * 0.70)
val_end = int(days * 0.85)

# Fit scaler only on training data
scaler = MinMaxScaler()

scaler.fit(
    df["Orders"].values[:train_end].reshape(-1, 1)
)

scaled_orders = scaler.transform(
    df["Orders"].values.reshape(-1, 1)
)

LOOK_BACK = 30

X = []
y = []

for i in range(LOOK_BACK, days):

    X.append(
        scaled_orders[i - LOOK_BACK:i]
    )

    y.append(
        scaled_orders[i]
    )

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Target indices corresponding to each sequence
target_indices = np.arange(LOOK_BACK, days)

# Chronological training, validation and test split
train_mask = target_indices < train_end

val_mask = (
    (target_indices >= train_end)
    & (target_indices < val_end)
)

test_mask = target_indices >= val_end

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nDataset shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# --------------------------------------------------
# 3. BUILD SIMPLE RNN MODEL
# --------------------------------------------------

UNITS = 32

def build_rnn_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(LOOK_BACK, 1)
        ),

        tf.keras.layers.SimpleRNN(
            UNITS,
            activation="tanh"
        ),

        tf.keras.layers.Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# --------------------------------------------------
# 4. BUILD LSTM MODEL
# --------------------------------------------------

def build_lstm_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(LOOK_BACK, 1)
        ),

        tf.keras.layers.LSTM(
            UNITS,
            activation="tanh"
        ),

        tf.keras.layers.Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# --------------------------------------------------
# 5. TRAIN SIMPLE RNN
# --------------------------------------------------

print("\nTraining SimpleRNN model...")

rnn_model = build_rnn_model()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

start_time = time.time()

rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

rnn_time = time.time() - start_time

print("\nSimpleRNN training completed.")
print("Training time:", round(rnn_time, 2), "seconds")


# --------------------------------------------------
# 6. TRAIN LSTM
# --------------------------------------------------

print("\nTraining LSTM model...")

lstm_model = build_lstm_model()

start_time = time.time()

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

lstm_time = time.time() - start_time

print("\nLSTM training completed.")
print("Training time:", round(lstm_time, 2), "seconds")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

rnn_predictions = rnn_model.predict(
    X_test,
    verbose=0
)

lstm_predictions = lstm_model.predict(
    X_test,
    verbose=0
)

# Convert predictions back to original order scale
rnn_predictions = scaler.inverse_transform(
    rnn_predictions
).flatten()

lstm_predictions = scaler.inverse_transform(
    lstm_predictions
).flatten()

actual_values = scaler.inverse_transform(
    y_test
).flatten()


# --------------------------------------------------
# 8. CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return mae, rmse, mape


rnn_mae, rnn_rmse, rnn_mape = calculate_metrics(
    actual_values,
    rnn_predictions
)

lstm_mae, lstm_rmse, lstm_mape = calculate_metrics(
    actual_values,
    lstm_predictions
)


# --------------------------------------------------
# 9. CREATE RESULTS TABLE USING PANDAS
# --------------------------------------------------

results = pd.DataFrame({

    "Model": ["SimpleRNN", "LSTM"],

    "MAE": [rnn_mae, lstm_mae],

    "RMSE": [rnn_rmse, lstm_rmse],

    "MAPE (%)": [rnn_mape, lstm_mape],

    "Training Time (seconds)": [
        rnn_time,
        lstm_time
    ],

    "Final Training Loss": [
        rnn_history.history["loss"][-1],
        lstm_history.history["loss"][-1]
    ],

    "Final Validation Loss": [
        rnn_history.history["val_loss"][-1],
        lstm_history.history["val_loss"][-1]
    ]

})

print("\n==========================================")
print("MODEL PERFORMANCE COMPARISON")
print("==========================================")

print(results.round(4).to_string(index=False))


# --------------------------------------------------
# 10. PLOT TRAINING AND VALIDATION LOSS
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    rnn_history.history["loss"],
    label="SimpleRNN Training Loss"
)

plt.plot(
    rnn_history.history["val_loss"],
    label="SimpleRNN Validation Loss"
)

plt.plot(
    lstm_history.history["loss"],
    label="LSTM Training Loss"
)

plt.plot(
    lstm_history.history["val_loss"],
    label="LSTM Validation Loss"
)

plt.title("Training and Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. PLOT ACTUAL VS PREDICTED ORDERS
# --------------------------------------------------

test_dates = dates[target_indices[test_mask]]

plt.figure(figsize=(13, 6))

plt.plot(
    test_dates,
    actual_values,
    label="Actual Orders"
)

plt.plot(
    test_dates,
    rnn_predictions,
    label="SimpleRNN Predictions"
)

plt.plot(
    test_dates,
    lstm_predictions,
    label="LSTM Predictions"
)

plt.title("Actual vs Predicted Daily Orders")
plt.xlabel("Date")
plt.ylabel("Number of Orders")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. COMPARE MAE, RMSE AND MAPE
# --------------------------------------------------

metrics = ["MAE", "RMSE", "MAPE (%)"]

for metric in metrics:

    plt.figure(figsize=(7, 5))

    plt.bar(
        results["Model"],
        results[metric]
    )

    plt.title(metric + " Comparison")
    plt.ylabel(metric)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 13. FINAL RESULT
# --------------------------------------------------

print("\n==========================================")
print("FINAL RESULT")
print("==========================================")

if lstm_rmse < rnn_rmse:

    print("LSTM achieved lower RMSE.")

else:

    print("SimpleRNN achieved lower RMSE.")

if lstm_mae < rnn_mae:

    print("LSTM achieved lower MAE.")

else:

    print("SimpleRNN achieved lower MAE.")

print("\nExperiment 16 completed successfully.")

