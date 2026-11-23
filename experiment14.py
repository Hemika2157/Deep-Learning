
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense

# --------------------------------------------------
# 1. Prepare five-word input sentence
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

words = ["I", "love", "machine", "learning", "models"]

# Assign a 3-dimensional vector to each word
word_vectors = {
    "I": [0.1, 0.2, 0.3],
    "love": [0.4, 0.5, 0.6],
    "machine": [0.7, 0.8, 0.9],
    "learning": [0.2, 0.4, 0.6],
    "models": [0.3, 0.5, 0.7]
}

X = np.array(
    [[word_vectors[word] for word in words]],
    dtype=np.float32
)

print("Input sentence:", " ".join(words))
print("Input shape:", X.shape)


# --------------------------------------------------
# 2. Build SimpleRNN model
# --------------------------------------------------

inputs = Input(shape=(5, 3))

rnn_layer = SimpleRNN(
    units=3,
    activation="tanh",
    return_sequences=True,
    return_state=True,
    name="simple_rnn"
)

sequence_output, final_state = rnn_layer(inputs)

output_layer = Dense(1, activation="linear", name="output")

outputs = output_layer(sequence_output)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="mse"
)

print("\nModel Summary:")
model.summary()


# --------------------------------------------------
# 3. Extract RNN weights
# --------------------------------------------------

weights = rnn_layer.get_weights()

W_xh = weights[0]   # Input weights
W_hh = weights[1]   # Recurrent weights
b_h = weights[2]    # Hidden bias

W_hy, b_y = output_layer.get_weights()

print("\nExtracted Weights:")
print("\nInput Weights W_xh:\n", W_xh)
print("\nRecurrent Weights W_hh:\n", W_hh)
print("\nHidden Bias:\n", b_h)
print("\nOutput Weights W_hy:\n", W_hy)


# --------------------------------------------------
# 4. Manually calculate hidden states
# --------------------------------------------------

hidden_states = []

h = np.zeros((1, 3), dtype=np.float32)

for t in range(5):

    x_t = X[:, t, :]

    # h(t) = tanh(x(t) W_xh + h(t-1) W_hh + bias)
    h = np.tanh(
        np.dot(x_t, W_xh)
        + np.dot(h, W_hh)
        + b_h
    )

    hidden_states.append(h.copy())

    print(f"\nHidden State h{t + 1}:")
    print(h)


hidden_states = np.concatenate(hidden_states, axis=0)


# --------------------------------------------------
# 5. Verify hidden states against Keras
# --------------------------------------------------

keras_output = Model(
    inputs=inputs,
    outputs=sequence_output
)

keras_states = keras_output.predict(X, verbose=0)[0]

difference = np.max(
    np.abs(hidden_states - keras_states)
)

print("\nManual Hidden States:")
print(hidden_states)

print("\nKeras Hidden States:")
print(keras_states)

print("\nMaximum Difference:", difference)

if np.allclose(hidden_states, keras_states, atol=1e-5):
    print("Verification Successful: Manual and Keras values match.")
else:
    print("Verification Failed.")


# --------------------------------------------------
# 6. Calculate output at each time step
# --------------------------------------------------

predicted_outputs = (
    np.dot(keras_states, W_hy) + b_y
)

print("\nOutput values at each time step:")
for i in range(5):
    print(f"y{i + 1} =", predicted_outputs[i][0])


# --------------------------------------------------
# 7. Draw unfolded RNN architecture
# --------------------------------------------------

G = nx.DiGraph()

# Add input, hidden and output nodes
for t in range(1, 6):

    G.add_node(f"x{t}", layer="input")
    G.add_node(f"h{t}", layer="hidden")
    G.add_node(f"y{t}", layer="output")

# Input to hidden connections
for t in range(1, 6):
    G.add_edge(
        f"x{t}", f"h{t}",
        label="W_xh"
    )

# Recurrent hidden state connections
for t in range(1, 6):
    if t == 1:
        G.add_node("h0", layer="initial")
        G.add_edge("h0", "h1", label="W_hh")
    else:
        G.add_edge(
            f"h{t-1}", f"h{t}",
            label="W_hh"
        )

# Hidden to output connections
for t in range(1, 6):
    G.add_edge(
        f"h{t}", f"y{t}",
        label="W_hy"
    )

# Set node positions
pos = {}

for t in range(1, 6):
    pos[f"x{t}"] = (t, 2)
    pos[f"h{t}"] = (t, 1)
    pos[f"y{t}"] = (t, 0)

pos["h0"] = (0, 1)

plt.figure(figsize=(14, 7))

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[f"x{i}" for i in range(1, 6)],
    node_color="skyblue",
    node_size=1800
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=[f"h{i}" for i in range(1, 6)] + ["h0"],
    node_color="lightgreen",
    node_size=1800
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=[f"y{i}" for i in range(1, 6)],
    node_color="orange",
    node_size=1800
)

# Draw edges and labels
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowsize=20,
    width=1.5,
    connectionstyle="arc3,rad=0.05"
)

nx.draw_networkx_labels(
    G, pos,
    font_size=11,
    font_weight="bold"
)

edge_labels = nx.get_edge_attributes(G, "label")

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=9
)

plt.title("Unfolded SimpleRNN Architecture")
plt.text(3, 2.5, "Input Layer", ha="center")
plt.text(3, 1.5, "Hidden State Layer", ha="center")
plt.text(3, -0.5, "Output Layer", ha="center")

plt.axis("off")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Plot hidden-state values across time
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(3):
    plt.plot(
        range(1, 6),
        hidden_states[:, i],
        marker="o",
        label=f"Hidden Unit {i + 1}"
    )

plt.title("Hidden State Values Across Time Steps")
plt.xlabel("Time Steps")
plt.ylabel("Hidden State Value")
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\nUnfolded RNN Visualization Completed Successfully!")
