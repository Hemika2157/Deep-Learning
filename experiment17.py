
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. INITIAL SETTINGS
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

LEARNING_RATE = 0.5
EPOCHS = 3
BATCH_SIZE = 32
SEQUENCE_LENGTH = 40
STEPS_PER_EPOCH = 40

# --------------------------------------------------
# 2. CHARACTER-LEVEL TEXT CORPUS
# --------------------------------------------------

text = """
The world of science and technology is changing every day.
People use computers to solve problems and discover new ideas.
Learning is a continuous process that requires patience and practice.
A neural network learns patterns from data through repeated training.
Recurrent neural networks process information one step at a time.
They are useful for language modelling and text generation.
A computer can learn to predict the next character in a sentence.
The future of artificial intelligence depends on research and innovation.
Students learn programming by writing and testing simple programs.
Knowledge grows when people explore new ideas and share information.
"""

# Repeat the corpus to create more training examples
text = text.lower() * 100

# Create character vocabulary
vocabulary = sorted(set(text))

char_to_index = {
    char: index
    for index, char in enumerate(vocabulary)
}

index_to_char = {
    index: char
    for char, index in char_to_index.items()
}

vocab_size = len(vocabulary)

print("Total characters:", len(text))
print("Vocabulary size:", vocab_size)
print("Vocabulary:", vocabulary)

# Convert characters into numbers
encoded_text = np.array(
    [char_to_index[char] for char in text],
    dtype=np.int32
)

# --------------------------------------------------
# 3. CREATE INPUT AND TARGET SEQUENCES
# --------------------------------------------------

X = []
y = []

for i in range(
    0,
    len(encoded_text) - SEQUENCE_LENGTH,
    3
):

    X.append(
        encoded_text[i:i + SEQUENCE_LENGTH]
    )

    y.append(
        encoded_text[i + SEQUENCE_LENGTH]
    )

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)

# --------------------------------------------------
# 4. BUILD CHARACTER-LEVEL RNN
# --------------------------------------------------

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH,)
        ),

        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=32
        ),

        tf.keras.layers.SimpleRNN(
            64,
            activation="tanh"
        ),

        tf.keras.layers.Dense(
            vocab_size
        )

    ])

    return model


# Create initial model weights
initial_model = build_model()

initial_model(X[:1])

initial_weights = initial_model.get_weights()


# --------------------------------------------------
# 5. TRAIN MODEL AND RECORD GRADIENTS
# --------------------------------------------------

def train_model(setting):

    print("\n====================================")
    print("Training setting:", setting)
    print("====================================")

    model = build_model()

    # Use identical initial weights for fair comparison
    model.set_weights(initial_weights)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE
    )

    gradient_norms = []
    losses = []

    spike_steps = []
    nan_steps = []

    global_step = 0

    # Training loop
    for epoch in range(EPOCHS):

        print("\nEpoch:", epoch + 1)

        # Shuffle training examples
        indices = np.random.permutation(len(X))

        epoch_losses = []

        for batch_number in range(STEPS_PER_EPOCH):

            global_step += 1

            batch_indices = indices[
                batch_number * BATCH_SIZE:
                (batch_number + 1) * BATCH_SIZE
            ]

            if len(batch_indices) == 0:
                continue

            batch_x = tf.convert_to_tensor(
                X[batch_indices],
                dtype=tf.int32
            )

            batch_y = tf.convert_to_tensor(
                y[batch_indices],
                dtype=tf.int32
            )

            # Forward pass and gradient calculation
            with tf.GradientTape() as tape:

                predictions = model(
                    batch_x,
                    training=True
                )

                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        batch_y,
                        predictions,
                        from_logits=True
                    )
                )

            gradients = tape.gradient(
                loss,
                model.trainable_variables
            )

            # Calculate original global gradient norm
            raw_gradient_norm = tf.linalg.global_norm(
                gradients
            )

            norm_value = float(
                raw_gradient_norm.numpy()
            )

            loss_value = float(loss.numpy())

            # Detect NaN or infinite values
            if (
                not np.isfinite(loss_value)
                or not np.isfinite(norm_value)
            ):

                print(
                    "Non-finite loss or gradient detected at step",
                    global_step
                )

                nan_steps.append(global_step)

                gradient_norms.append(norm_value)
                losses.append(loss_value)

                break

            # Record original gradient norm
            gradient_norms.append(norm_value)
            losses.append(loss_value)

            epoch_losses.append(loss_value)

            # Detect gradient spikes
            if norm_value > 10:

                spike_steps.append(global_step)

                print(
                    "Gradient spike at step:",
                    global_step,
                    "| Norm:",
                    round(norm_value, 4)
                )

            # Apply selected gradient clipping method
            if setting == "clipnorm_1":

                gradients, _ = tf.clip_by_global_norm(
                    gradients,
                    clip_norm=1.0
                )

            elif setting == "clipnorm_0.5":

                gradients, _ = tf.clip_by_global_norm(
                    gradients,
                    clip_norm=0.5
                )

            elif setting == "clipvalue_0.5":

                gradients = [
                    tf.clip_by_value(
                        grad,
                        clip_value_min=-0.5,
                        clip_value_max=0.5
                    )
                    for grad in gradients
                ]

            # No clipping is applied for "no_clipping"

            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables
                )
            )

        if nan_steps:
            print("Stopping because of non-finite values.")
            break

        if epoch_losses:
            print(
                "Average loss:",
                round(np.mean(epoch_losses), 5)
            )

    return {
        "model": model,
        "gradient_norms": gradient_norms,
        "losses": losses,
        "spike_steps": spike_steps,
        "nan_steps": nan_steps
    }


# --------------------------------------------------
# 6. TRAIN WITH DIFFERENT CLIPPING SETTINGS
# --------------------------------------------------

settings = [
    "no_clipping",
    "clipnorm_1",
    "clipvalue_0.5",
    "clipnorm_0.5"
]

results = {}

for setting in settings:

    results[setting] = train_model(setting)


# --------------------------------------------------
# 7. DISPLAY GRADIENT SPIKES AND NAN STEPS
# --------------------------------------------------

print("\n====================================")
print("GRADIENT ANALYSIS")
print("====================================")

for setting in settings:

    result = results[setting]

    print("\nSetting:", setting)

    print(
        "Maximum gradient norm:",
        np.max(result["gradient_norms"])
    )

    print(
        "Gradient spike steps:",
        result["spike_steps"]
    )

    print(
        "NaN / non-finite steps:",
        result["nan_steps"]
    )


# --------------------------------------------------
# 8. PLOT GLOBAL GRADIENT NORMS
# --------------------------------------------------

plt.figure(figsize=(12, 6))

for setting in settings:

    norms = np.array(
        results[setting]["gradient_norms"],
        dtype=float
    )

    # Avoid invalid values in the logarithmic plot
    safe_norms = np.where(
        np.isfinite(norms),
        np.maximum(norms, 1e-8),
        np.nan
    )

    plt.plot(
        safe_norms,
        label=setting
    )

plt.yscale("log")

plt.title("Global Gradient Norm Comparison")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm (Log Scale)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. PLOT TRAINING LOSS
# --------------------------------------------------

plt.figure(figsize=(12, 6))

for setting in settings:

    loss_values = np.array(
        results[setting]["losses"],
        dtype=float
    )

    plt.plot(
        loss_values,
        label=setting
    )

plt.title("Training Loss Comparison")
plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. COMPARE MAXIMUM GRADIENT NORMS
# --------------------------------------------------

maximum_norms = []

for setting in settings:

    norms = np.array(
        results[setting]["gradient_norms"],
        dtype=float
    )

    finite_norms = norms[
        np.isfinite(norms)
    ]

    if len(finite_norms) > 0:
        maximum_norms.append(
            np.max(finite_norms)
        )
    else:
        maximum_norms.append(0)

plt.figure(figsize=(9, 5))

plt.bar(
    settings,
    maximum_norms
)

plt.title("Maximum Gradient Norm by Clipping Method")
plt.xlabel("Clipping Method")
plt.ylabel("Maximum Gradient Norm")

plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. RECOMMEND A CLIPPING THRESHOLD
# --------------------------------------------------

print("\n====================================")
print("FINAL COMPARISON")
print("====================================")

for setting in settings:

    result = results[setting]

    finite_losses = [
        value
        for value in result["losses"]
        if np.isfinite(value)
    ]

    if finite_losses:

        print(
            setting,
            "| Final recorded loss:",
            round(finite_losses[-1], 5),
            "| Maximum gradient:",
            round(
                np.max([
                    value
                    for value in result["gradient_norms"]
                    if np.isfinite(value)
                ]),
                5
            )
        )

    else:

        print(setting, "| No finite loss recorded.")

print("\nRecommended starting threshold: clipnorm = 1.0")

print(
    "Compare the loss curves and gradient norms "
    "to confirm the most stable setting."
)

print("\nExperiment 17 completed.")

