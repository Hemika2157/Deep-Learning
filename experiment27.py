
# Experiment 27: Deep Belief Network Representation Learning

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.special import expit
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


# ------------------------------------------------
# 1. LOAD MNIST DATASET
# ------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

print("Loading MNIST dataset...")

# Download MNIST if it is not already cached
mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False
)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)

# Original MNIST has 60,000 training and 10,000 test samples
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

# Use 20,000 images for unsupervised pre-training
X_unlabeled = X_train[:20000]

# Select only 1,000 labelled samples for fine-tuning
rng = np.random.default_rng(42)
labeled_indices = rng.choice(
    len(X_train),
    size=1000,
    replace=False
)

X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]

# Convert labels into one-hot format
y_labeled_onehot = to_categorical(y_labeled, 10)
y_test_onehot = to_categorical(y_test, 10)

print("Unlabelled training samples:", len(X_unlabeled))
print("Labelled training samples:", len(X_labeled))
print("Test samples:", len(X_test))


# ------------------------------------------------
# 2. RBM CLASS
# ------------------------------------------------

class RBM:

    def __init__(
        self,
        n_visible,
        n_hidden,
        learning_rate=0.01
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        self.W = np.random.normal(
            0, 0.01, (n_visible, n_hidden)
        ).astype(np.float32)

        self.visible_bias = np.zeros(
            n_visible, dtype=np.float32
        )

        self.hidden_bias = np.zeros(
            n_hidden, dtype=np.float32
        )

    def hidden_probability(self, visible):

        return expit(
            np.dot(visible, self.W) + self.hidden_bias
        )

    def visible_probability(self, hidden):

        return expit(
            np.dot(hidden, self.W.T) + self.visible_bias
        )

    def train(self, data, epochs=3, batch_size=100):

        for epoch in range(epochs):

            indices = np.random.permutation(len(data))
            shuffled = data[indices]

            total_error = 0
            batches = 0

            for start in range(
                0, len(shuffled), batch_size
            ):

                v0 = shuffled[start:start + batch_size]

                # Positive phase
                ph0 = self.hidden_probability(v0)

                h0 = (
                    np.random.random(ph0.shape) < ph0
                ).astype(np.float32)

                # Negative phase: CD-1
                pv1 = self.visible_probability(h0)

                v1 = (
                    np.random.random(pv1.shape) < pv1
                ).astype(np.float32)

                ph1 = self.hidden_probability(v1)

                # Update weights
                self.W += self.learning_rate * (
                    v0.T @ ph0 - v1.T @ ph1
                ) / len(v0)

                self.visible_bias += self.learning_rate * np.mean(
                    v0 - v1, axis=0
                )

                self.hidden_bias += self.learning_rate * np.mean(
                    ph0 - ph1, axis=0
                )

                total_error += np.mean((v0 - pv1) ** 2)
                batches += 1

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Reconstruction Error: "
                f"{total_error / batches:.4f}"
            )

    def transform(self, data):

        return self.hidden_probability(data)


# ------------------------------------------------
# 3. PRE-TRAIN FIRST RBM: 784 -> 500
# ------------------------------------------------

print("\nTraining RBM 1: 784 -> 500")

rbm1 = RBM(
    n_visible=784,
    n_hidden=500,
    learning_rate=0.01
)

rbm1.train(
    X_unlabeled,
    epochs=3,
    batch_size=100
)

# Transform original input into 500 features
hidden1 = rbm1.transform(X_unlabeled)

print("First hidden representation:", hidden1.shape)


# ------------------------------------------------
# 4. PRE-TRAIN SECOND RBM: 500 -> 250
# ------------------------------------------------

print("\nTraining RBM 2: 500 -> 250")

rbm2 = RBM(
    n_visible=500,
    n_hidden=250,
    learning_rate=0.01
)

rbm2.train(
    hidden1,
    epochs=3,
    batch_size=100
)

# Extract second hidden representation
hidden2 = rbm2.transform(hidden1)

print("Second hidden representation:", hidden2.shape)


# ------------------------------------------------
# 5. VISUALIZE FIRST RBM WEIGHTS
# ------------------------------------------------

plt.figure(figsize=(10, 8))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    # Each hidden neuron has 784 input weights
    weight_image = rbm1.W[:, i].reshape(28, 28)

    plt.imshow(weight_image, cmap='gray')
    plt.axis('off')
    plt.title(f"Filter {i + 1}")

plt.suptitle("First RBM Learned Weights")
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 6. BUILD DBN CLASSIFIER
# ------------------------------------------------

def build_network():

    model = Sequential([
        Input(shape=(784,)),

        Dense(500, activation='sigmoid'),

        Dense(250, activation='sigmoid'),

        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ------------------------------------------------
# 7. DBN PRE-TRAINED NETWORK
# ------------------------------------------------

print("\nBuilding DBN pretrained classifier")

dbn_model = build_network()

# Transfer RBM weights into Keras Dense layers
dbn_model.layers[0].set_weights([
    rbm1.W,
    rbm1.hidden_bias
])

dbn_model.layers[1].set_weights([
    rbm2.W,
    rbm2.hidden_bias
])

print("RBM weights transferred successfully")

print("\nFine-tuning DBN using 1000 labelled samples")

dbn_model.fit(
    X_labeled,
    y_labeled_onehot,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

dbn_loss, dbn_accuracy = dbn_model.evaluate(
    X_test,
    y_test_onehot,
    verbose=0
)


# ------------------------------------------------
# 8. RANDOMLY INITIALIZED NETWORK
# ------------------------------------------------

print("\nTraining randomly initialized network")

random_model = build_network()

random_model.fit(
    X_labeled,
    y_labeled_onehot,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

random_loss, random_accuracy = random_model.evaluate(
    X_test,
    y_test_onehot,
    verbose=0
)


# ------------------------------------------------
# 9. COMPARE CLASSIFICATION ACCURACY
# ------------------------------------------------

print("\nCLASSIFICATION COMPARISON")

print("DBN Pretrained Accuracy:",
      round(dbn_accuracy * 100, 2), "%")

print("Random Network Accuracy:",
      round(random_accuracy * 100, 2), "%")

comparison = [
    dbn_accuracy * 100,
    random_accuracy * 100
]

plt.figure(figsize=(7, 5))

plt.bar(
    ['DBN Pretrained', 'Random Initialization'],
    comparison
)

plt.ylabel("Test Accuracy (%)")
plt.title("DBN vs Random Network Accuracy")
plt.ylim(0, 100)
plt.show()


# ------------------------------------------------
# 10. EXTRACT LEARNED REPRESENTATIONS
# ------------------------------------------------

# Get output of second hidden layer
feature_model = tf.keras.Model(
    inputs=dbn_model.input,
    outputs=dbn_model.layers[1].output
)

features = feature_model.predict(
    X_test[:2000],
    verbose=0
)

labels = y_test[:2000]

print("\nLearned representation shape:", features.shape)


# ------------------------------------------------
# 11. PCA FOR 2-D REPRESENTATION
# ------------------------------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)

features_2d = pca.fit_transform(scaled_features)

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    features_2d[:, 0],
    features_2d[:, 1],
    c=labels,
    cmap='tab10',
    s=8
)

plt.colorbar(scatter, label="Digit Class")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("2-D Visualization of DBN Learned Representations")

plt.show()


# ------------------------------------------------
# 12. FINAL RESULT
# ------------------------------------------------

print("\nEXPERIMENT 27 COMPLETED SUCCESSFULLY")

