
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import SimpleRNN, Dense

# --------------------------------------------------
# 1. Generate synthetic daily stock prices
# --------------------------------------------------

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

days = 1095

trend = np.linspace(100, 160, days)
seasonal = 8 * np.sin(np.arange(days) * 2 * np.pi / 90)
noise = np.random.normal(0, 2, days)

prices = trend + seasonal + noise

print("Total stock records:", len(prices))

# Scale prices between 0 and 1
minimum = prices.min()
maximum = prices.max()

scaled_prices = (prices - minimum) / (maximum - minimum)

# Chronological split
split_point = int(days * 0.8)


# --------------------------------------------------
# 2. Create sliding-window sequences
# --------------------------------------------------

def create_sequences(data, sequence_length):

    X = []
    y = []
    target_indices = []

    for i in range(len(data) - sequence_length):

        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
        target_indices.append(i + sequence_length)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = X.reshape(-1, sequence_length, 1)

    return X, y, np.array(target_indices)


# --------------------------------------------------
# 3. Build SimpleRNN model
# --------------------------------------------------

def build_model(sequence_length, activation):

    inputs = Input(shape=(sequence_length, 1))

    rnn_layer = SimpleRNN(
        16,
        activation=activation,
        return_sequences=True,
        return_state=True,
        recurrent_initializer="orthogonal"
    )

    sequence_output, final_state = rnn_layer(inputs)

    output_layer = Dense(1)

    prediction = output_layer(final_state)

    model = Model(inputs, prediction)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    return model, rnn_layer, output_layer


# --------------------------------------------------
# 4. Calculate gradients using BPTT
# --------------------------------------------------

def calculate_gradients(
    rnn_layer,
    output_layer,
    input_sequence,
    target,
    activation
):

    # Extract trained Keras RNN weights
    W_xh, W_hh, b_h = rnn_layer.get_weights()

    W_hy, b_y = output_layer.get_weights()

    # Select activation function
    if activation == "tanh":
        activation_function = tf.math.tanh
    else:
        activation_function = tf.nn.relu

    # Initial hidden state
    hidden_state = tf.zeros((1, 16))

    hidden_states = []

    # Record intermediate hidden states
    with tf.GradientTape() as tape:

        for t in range(input_sequence.shape[0]):

            x_t = tf.constant(
                input_sequence[t:t + 1],
                dtype=tf.float32
            )

            # RNN hidden-state calculation
            hidden_state = activation_function(
                tf.matmul(x_t, W_xh)
                + tf.matmul(hidden_state, W_hh)
                + b_h
            )

            hidden_states.append(hidden_state)

        # Final output prediction
        prediction = (
            tf.matmul(hidden_state, W_hy) + b_y
        )

        # Calculate loss
        loss = tf.reduce_mean(
            tf.square(prediction - target)
        )

    # Gradient of loss with respect to each hidden state
    gradients = tape.gradient(loss, hidden_states)

    gradient_norms = []

    for gradient in gradients:

        norm = tf.norm(gradient).numpy()
        gradient_norms.append(norm)

    return np.array(gradient_norms), loss.numpy()


# --------------------------------------------------
# 5. Experiment with different sequence lengths
# --------------------------------------------------

sequence_lengths = [10, 50, 100]
activations = ["tanh", "relu"]

results = {}

for activation in activations:

    for length in sequence_lengths:

        print("\n--------------------------------")
        print("Activation:", activation)
        print("Sequence Length:", length)
        print("--------------------------------")

        # Prepare sequences
        X, y, indices = create_sequences(
            scaled_prices,
            length
        )

        # Chronological train-test split
        train_mask = indices < split_point
        test_mask = indices >= split_point

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        # Build model
        model, rnn_layer, output_layer = build_model(
            length,
            activation
        )

        # Train model
        history = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            verbose=0
        )

        # Calculate gradients for one unseen test sequence
        gradient_norms, loss = calculate_gradients(
            rnn_layer,
            output_layer,
            X_test[0],
            y_test[0],
            activation
        )

        results[(activation, length)] = gradient_norms

        print("Training completed.")
        print("Final training loss:", round(history.history["loss"][-1], 6))
        print("Test sequence loss:", round(loss, 6))
        print("First-step gradient:", gradient_norms[0])
        print("Last-step gradient:", gradient_norms[-1])


# --------------------------------------------------
# 6. Plot gradient norms for tanh and ReLU
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, activation in zip(axes, activations):

    for length in sequence_lengths:

        gradients = results[(activation, length)]

        ax.plot(
            range(1, length + 1),
            gradients + 1e-12,
            label=f"{length} Days"
        )

    ax.set_title(f"Gradient Flow - {activation.upper()}")
    ax.set_xlabel("Time Step (Oldest to Newest)")
    ax.set_ylabel("Gradient Norm (Log Scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Display gradient comparison
# --------------------------------------------------

print("\n========== GRADIENT COMPARISON ==========")

print(
    f"{'Activation':<12}"
    f"{'Sequence':>12}"
    f"{'First Step':>18}"
    f"{'Last Step':>18}"
)

for activation in activations:

    for length in sequence_lengths:

        gradients = results[(activation, length)]

        print(
            f"{activation:<12}"
            f"{length:>12}"
            f"{gradients[0]:>18.8e}"
            f"{gradients[-1]:>18.8e}"
        )


# --------------------------------------------------
# 8. Explain gradient behaviour
# --------------------------------------------------

print("\n========== GRADIENT ANALYSIS ==========")

for activation in activations:

    print("\nActivation:", activation.upper())

    for length in sequence_lengths:

        gradients = results[(activation, length)]

        first = gradients[0]
        last = gradients[-1]

        print(f"\nSequence length: {length} days")
        print("Oldest-step gradient:", f"{first:.8e}")
        print("Newest-step gradient:", f"{last:.8e}")

        if first < last * 0.01:
            print("Possible vanishing gradient at earlier time steps.")
        elif first > last * 100:
            print("Possible exploding gradient at earlier time steps.")
        else:
            print("No extreme gradient difference detected.")

print("\nBPTT Gradient Analysis Completed!")

