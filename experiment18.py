
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
import time

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# --------------------------------------------------
# 1. INITIAL SETTINGS
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

MAX_WORDS = 10000
MAX_LENGTH = 200
BATCH_SIZE = 64

# --------------------------------------------------
# 2. LOAD IMDB SENTIMENT DATASET
# --------------------------------------------------

print("Loading IMDB dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=MAX_WORDS
)

# Use a smaller dataset to reduce computation
x_train = x_train[:8000]
y_train = y_train[:8000]

x_val = x_test[:2000]
y_val = y_test[:2000]

x_test = x_test[2000:7000]
y_test = y_test[2000:7000]

# Pad sequences to equal length
x_train = pad_sequences(
    x_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

x_val = pad_sequences(
    x_val,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("\nDataset information:")
print("Training samples:", len(x_train))
print("Validation samples:", len(x_val))
print("Testing samples:", len(x_test))
print("Vocabulary size:", MAX_WORDS)

# --------------------------------------------------
# 3. MODEL BUILDER FUNCTION
# --------------------------------------------------

def model_builder(hp, cell_type):

    # Tune embedding size
    embedding_size = hp.Choice(
        "embedding_size",
        values=[32, 64, 96]
    )

    # Tune recurrent units
    units = hp.Choice(
        "units",
        values=[32, 64, 96]
    )

    # Tune number of recurrent layers
    num_layers = hp.Choice(
        "num_layers",
        values=[1, 2]
    )

    # Tune dropout
    dropout_rate = hp.Choice(
        "dropout",
        values=[0.2, 0.4, 0.5]
    )

    # Tune learning rate
    learning_rate = hp.Choice(
        "learning_rate",
        values=[0.001, 0.0005, 0.0001]
    )

    model = models.Sequential()

    model.add(
        layers.Input(shape=(MAX_LENGTH,))
    )

    # Embedding layer
    model.add(
        layers.Embedding(
            input_dim=MAX_WORDS,
            output_dim=embedding_size,
            mask_zero=True
        )
    )

    # Select recurrent cell
    if cell_type == "RNN":
        recurrent_layer = layers.SimpleRNN

    elif cell_type == "LSTM":
        recurrent_layer = layers.LSTM

    else:
        recurrent_layer = layers.GRU

    # Add recurrent layers
    for i in range(num_layers):

        return_sequences = (
            i < num_layers - 1
        )

        model.add(
            recurrent_layer(
                units,
                return_sequences=return_sequences
            )
        )

        model.add(
            layers.Dropout(dropout_rate)
        )

    # Binary sentiment output
    model.add(
        layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 4. HYPERPARAMETER TUNING
# --------------------------------------------------

cell_types = ["RNN", "LSTM", "GRU"]

best_models = {}
best_histories = {}
results = []

for cell_type in cell_types:

    print("\n====================================")
    print("TUNING", cell_type, "MODEL")
    print("====================================")

    # Create a separate builder for each cell type
    def build_model(hp, selected_type=cell_type):

        return model_builder(
            hp,
            selected_type
        )

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective="val_accuracy",
        max_epochs=4,
        factor=3,
        hyperband_iterations=1,
        directory="experiment18_tuning",
        project_name=cell_type,
        overwrite=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # Start hyperparameter search
    search_start = time.time()

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=4,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    search_time = time.time() - search_start

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(
        num_trials=1
    )[0]

    print("\nBest configuration for", cell_type)

    print("Embedding size:", best_hp.get("embedding_size"))
    print("Units:", best_hp.get("units"))
    print("Number of layers:", best_hp.get("num_layers"))
    print("Dropout:", best_hp.get("dropout"))
    print("Learning rate:", best_hp.get("learning_rate"))

    # Build model using best parameters
    best_model = tuner.hypermodel.build(best_hp)

    print("Parameter count:", best_model.count_params())

    # Train best model for final evaluation
    print("\nTraining best", cell_type, "model...")

    start_time = time.time()

    history = best_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=4,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time

    # Evaluate on test dataset
    test_loss, test_accuracy = best_model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("\n", cell_type, "Test Accuracy:",
          round(test_accuracy * 100, 2), "%")

    print("Test Loss:", round(test_loss, 4))
    print("Training Time:", round(training_time, 2), "seconds")
    print("Tuning Time:", round(search_time, 2), "seconds")

    # Save model and history
    best_models[cell_type] = best_model
    best_histories[cell_type] = history

    # Store results
    results.append({

        "Model": cell_type,

        "Best Embedding Size":
            best_hp.get("embedding_size"),

        "Best Units":
            best_hp.get("units"),

        "Best Layers":
            best_hp.get("num_layers"),

        "Best Dropout":
            best_hp.get("dropout"),

        "Best Learning Rate":
            best_hp.get("learning_rate"),

        "Test Accuracy (%)":
            test_accuracy * 100,

        "Test Loss":
            test_loss,

        "Parameters":
            best_model.count_params(),

        "Training Time (seconds)":
            training_time,

        "Tuning Time (seconds)":
            search_time
    })


# --------------------------------------------------
# 5. DISPLAY BEST CONFIGURATION TABLE
# --------------------------------------------------

import pandas as pd

results_df = pd.DataFrame(results)

print("\n====================================")
print("BEST MODEL CONFIGURATIONS")
print("====================================")

print(
    results_df[
        [
            "Model",
            "Best Embedding Size",
            "Best Units",
            "Best Layers",
            "Best Dropout",
            "Best Learning Rate"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 6. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

print("\n====================================")
print("MODEL PERFORMANCE COMPARISON")
print("====================================")

print(
    results_df[
        [
            "Model",
            "Test Accuracy (%)",
            "Test Loss",
            "Parameters",
            "Training Time (seconds)"
        ]
    ].round(3).to_string(index=False)
)


# --------------------------------------------------
# 7. PLOT TRAINING AND VALIDATION ACCURACY
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for cell_type in cell_types:

    history = best_histories[cell_type]

    plt.plot(
        history.history["accuracy"],
        label=cell_type + " Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        linestyle="--",
        label=cell_type + " Validation Accuracy"
    )

plt.title("RNN, LSTM and GRU Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. PLOT TRAINING AND VALIDATION LOSS
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for cell_type in cell_types:

    history = best_histories[cell_type]

    plt.plot(
        history.history["loss"],
        label=cell_type + " Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        linestyle="--",
        label=cell_type + " Validation Loss"
    )

plt.title("Training and Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. PLOT TEST ACCURACY
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Test Accuracy (%)"]
)

plt.title("Test Accuracy Comparison")
plt.xlabel("Recurrent Model")
plt.ylabel("Accuracy (%)")

plt.grid(axis="y")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. PLOT PARAMETER COUNT
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Parameters"]
)

plt.title("Model Parameter Count Comparison")
plt.xlabel("Recurrent Model")
plt.ylabel("Number of Parameters")

plt.grid(axis="y")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. PLOT TRAINING TIME
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Training Time (seconds)"]
)

plt.title("Training Time Comparison")
plt.xlabel("Recurrent Model")
plt.ylabel("Training Time (seconds)")

plt.grid(axis="y")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. IDENTIFY BEST MODEL
# --------------------------------------------------

best_index = results_df["Test Accuracy (%)"].idxmax()

best_model_name = results_df.loc[
    best_index, "Model"
]

best_accuracy = results_df.loc[
    best_index, "Test Accuracy (%)"
]

print("\n====================================")
print("FINAL RESULT")
print("====================================")

print("Best model:", best_model_name)

print(
    "Best test accuracy:",
    round(best_accuracy, 2),
    "%"
)

print("\nExperiment 18 completed successfully.")

