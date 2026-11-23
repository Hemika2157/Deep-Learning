
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# --------------------------------------------------
# 1. Generate hourly electricity demand data
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

hours = 365 * 24

date_range = pd.date_range(
    start="2025-01-01",
    periods=hours,
    freq="h"
)

hour_of_day = np.arange(hours) % 24
day_of_year = np.arange(hours) / 24

# Simulated electricity demand
demand = (
    500
    + 80 * np.sin(2 * np.pi * hour_of_day / 24)
    + 50 * np.sin(2 * np.pi * day_of_year / 365)
    + 20 * np.random.randn(hours)
)

# Create Pandas DataFrame
df = pd.DataFrame({
    "Date": date_range,
    "Demand": demand
})

df.set_index("Date", inplace=True)

# Introduce a few missing values for demonstration
df.iloc[100:105, 0] = np.nan

# Handle missing values
df["Demand"] = df["Demand"].interpolate()

print("Dataset Information:")
print(df.head())

print("\nTotal records:", len(df))
print("Missing values:", df["Demand"].isnull().sum())


# --------------------------------------------------
# 2. Plot original electricity demand
# --------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(df.index[:168], df["Demand"].iloc[:168])
plt.title("Hourly Electricity Demand - First 7 Days")
plt.xlabel("Date")
plt.ylabel("Electricity Demand")
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Scale the data
# --------------------------------------------------

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[["Demand"]]
)


# --------------------------------------------------
# 4. Create sliding windows
# --------------------------------------------------

window_size = 24

def create_sequences(data, window):

    X = []
    y = []

    for i in range(len(data) - window):

        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)


X, y = create_sequences(
    scaled_data,
    window_size
)

print("\nSequence Information:")
print("Input shape:", X.shape)
print("Output shape:", y.shape)


# --------------------------------------------------
# 5. Chronological train-test split
# --------------------------------------------------

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTraining sequences:", len(X_train))
print("Testing sequences:", len(X_test))


# --------------------------------------------------
# 6. Build SimpleRNN model
# --------------------------------------------------

model = Sequential([
    SimpleRNN(
        50,
        activation="tanh",
        input_shape=(window_size, 1)
    ),

    Dense(25, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("\nRNN Model Summary:")
model.summary()


# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)


# --------------------------------------------------
# 8. Predict electricity demand
# --------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

# Convert scaled values back to original demand
y_test_actual = scaler.inverse_transform(y_test)
predicted_actual = scaler.inverse_transform(predictions)


# --------------------------------------------------
# 9. Calculate MAE and RMSE
# --------------------------------------------------

mae = mean_absolute_error(
    y_test_actual,
    predicted_actual
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        predicted_actual
    )
)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# --------------------------------------------------
# 10. Plot actual vs predicted demand
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    y_test_actual[:200],
    label="Actual Demand"
)

plt.plot(
    predicted_actual[:200],
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Test Hours")
plt.ylabel("Electricity Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("SimpleRNN Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nSimpleRNN Electricity Demand Prediction Completed!")
