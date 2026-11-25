
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.ndimage import zoom
from sklearn.datasets import load_digits

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# 1. LOAD AND PREPARE IMAGE DATA
# --------------------------------------------------

digits = load_digits()

# Resize 8x8 images to 28x28
images = np.array([
    zoom(img, 3.5, order=1) for img in digits.images
])

images = np.clip(images / 16.0, 0, 1)
images = images[:1200].astype("float32")

# Shape: (number of images, height, width, channels)
X = images[..., np.newaxis]

print("Dataset shape:", X.shape)

# --------------------------------------------------
# 2. CONVOLUTIONAL RBM
# --------------------------------------------------

class ConvRBM:
    def __init__(self, filters=16, kernel_size=5, lr=0.001):
        self.filters = filters
        self.kernel_size = kernel_size
        self.lr = lr

        # Shared convolution filters
        self.W = tf.Variable(
            tf.random.normal(
                [kernel_size, kernel_size, 1, filters],
                stddev=0.05
            )
        )

        self.hidden_bias = tf.Variable(tf.zeros([filters]))
        self.visible_bias = tf.Variable(tf.zeros([1]))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def hidden_probability(self, visible):
        output = tf.nn.conv2d(
            visible,
            self.W,
            strides=1,
            padding="VALID"
        )
        return tf.sigmoid(output + self.hidden_bias)

    def sample(self, probability):
        random_values = tf.random.uniform(tf.shape(probability))
        return tf.cast(random_values < probability, tf.float32)

    def visible_probability(self, hidden):
        output_shape = tf.shape(hidden)
        batch_size = output_shape[0]

        # For 28x28 images and 5x5 filters,
        # the reconstructed image is 28x28.
        height = output_shape[1] + self.kernel_size - 1
        width = output_shape[2] + self.kernel_size - 1

        output = tf.nn.conv2d_transpose(
            hidden,
            self.W,
            output_shape=[batch_size, height, width, 1],
            strides=1,
            padding="VALID"
        )

        return tf.sigmoid(output + self.visible_bias)

    def free_energy(self, visible):
        visible_term = tf.reduce_sum(
            visible * self.visible_bias,
            axis=[1, 2, 3]
        )

        hidden_input = tf.nn.conv2d(
            visible,
            self.W,
            strides=1,
            padding="VALID"
        ) + self.hidden_bias

        hidden_term = tf.reduce_sum(
            tf.nn.softplus(hidden_input),
            axis=[1, 2, 3]
        )

        return -visible_term - hidden_term

    def train_batch(self, visible):
        with tf.GradientTape() as tape:

            # Positive phase
            positive_hidden = self.hidden_probability(visible)
            hidden_sample = self.sample(positive_hidden)

            # Negative phase: one Gibbs step
            negative_visible_prob = self.visible_probability(hidden_sample)
            negative_visible = self.sample(negative_visible_prob)

            # Contrastive Divergence loss
            loss = tf.reduce_mean(
                self.free_energy(visible)
                - self.free_energy(negative_visible)
            )

        variables = [self.W, self.hidden_bias, self.visible_bias]

        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return float(loss.numpy())

    def reconstruct(self, visible):
        hidden = self.hidden_probability(visible)
        return self.visible_probability(hidden)

    def generate(self, steps=100):
        # Start with random image pixels
        visible = tf.cast(
            tf.random.uniform([1, 28, 28, 1]) > 0.5,
            tf.float32
        )

        # Gibbs sampling
        for _ in range(steps):
            hidden_prob = self.hidden_probability(visible)
            hidden = self.sample(hidden_prob)

            visible_prob = self.visible_probability(hidden)
            visible = self.sample(visible_prob)

        return visible.numpy()[0, :, :, 0]


# --------------------------------------------------
# 3. TRAIN CONVOLUTIONAL RBM
# --------------------------------------------------

conv_rbm = ConvRBM(filters=16, kernel_size=5, lr=0.001)

epochs = 8
batch_size = 50

conv_losses = []

print("\nTraining Convolutional RBM...")

for epoch in range(epochs):
    np.random.shuffle(X)
    epoch_losses = []

    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]

        loss = conv_rbm.train_batch(
            tf.convert_to_tensor(batch)
        )

        epoch_losses.append(loss)

    average_loss = np.mean(epoch_losses)
    conv_losses.append(average_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")


# --------------------------------------------------
# 4. SINGLE-LAYER RBM FOR COMPARISON
# --------------------------------------------------

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class SimpleRBM:
    def __init__(self, visible=784, hidden=64, lr=0.01):
        self.W = np.random.normal(0, 0.01, (visible, hidden))
        self.vb = np.zeros(visible)
        self.hb = np.zeros(hidden)
        self.lr = lr

    def hidden_probability(self, v):
        return sigmoid(v @ self.W + self.hb)

    def visible_probability(self, h):
        return sigmoid(h @ self.W.T + self.vb)

    def train_batch(self, v):
        # Positive phase
        h_prob = self.hidden_probability(v)
        h_sample = (np.random.rand(*h_prob.shape) < h_prob).astype(float)

        # Negative phase: CD-1
        v_prob = self.visible_probability(h_sample)
        v_sample = (np.random.rand(*v_prob.shape) < v_prob).astype(float)

        h_prob_negative = self.hidden_probability(v_sample)

        # Weight and bias updates
        positive = v.T @ h_prob
        negative = v_sample.T @ h_prob_negative

        self.W += self.lr * (positive - negative) / len(v)
        self.vb += self.lr * np.mean(v - v_sample, axis=0)
        self.hb += self.lr * np.mean(h_prob - h_prob_negative, axis=0)

    def reconstruct(self, v):
        h = self.hidden_probability(v)
        return self.visible_probability(h)


simple_rbm = SimpleRBM()

flat_X = images.reshape(len(images), -1)

print("\nTraining Single-Layer RBM...")

for epoch in range(epochs):
    np.random.shuffle(flat_X)

    for start in range(0, len(flat_X), batch_size):
        batch = flat_X[start:start + batch_size]
        simple_rbm.train_batch(batch)

    print(f"Single RBM Epoch {epoch + 1}/{epochs} completed")


# --------------------------------------------------
# 5. COMPARE RECONSTRUCTION QUALITY
# --------------------------------------------------

test_images = X[:100]

conv_recon = conv_rbm.reconstruct(
    tf.convert_to_tensor(test_images)
).numpy()

simple_recon = simple_rbm.reconstruct(
    test_images.reshape(100, -1)
).reshape(100, 28, 28, 1)

conv_mse = np.mean((test_images - conv_recon) ** 2)
simple_mse = np.mean((test_images - simple_recon) ** 2)

print("\nRECONSTRUCTION COMPARISON")
print("--------------------------------")
print(f"Convolutional RBM MSE : {conv_mse:.5f}")
print(f"Single-layer RBM MSE  : {simple_mse:.5f}")

# --------------------------------------------------
# 6. VISUALISE FILTERS
# --------------------------------------------------

filters = conv_rbm.W.numpy()

plt.figure(figsize=(8, 6))

for i in range(16):
    plt.subplot(4, 4, i + 1)

    current_filter = filters[:, :, 0, i]
    plt.imshow(current_filter, cmap="gray")
    plt.title(f"Filter {i + 1}")
    plt.axis("off")

plt.suptitle("Convolutional RBM Learned Filters")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. VISUALISE HIDDEN UNIT ACTIVATIONS
# --------------------------------------------------

sample_image = tf.convert_to_tensor(X[:1])
activations = conv_rbm.hidden_probability(sample_image).numpy()[0]

plt.figure(figsize=(9, 6))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(activations[:, :, i], cmap="viridis")
    plt.title(f"Hidden {i + 1}")
    plt.axis("off")

plt.suptitle("Hidden Unit Activations")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. DISPLAY ORIGINAL AND RECONSTRUCTED IMAGES
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i, :, :, 0], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(conv_recon[i, :, :, 0], cmap="gray")
    plt.title("CRBM Reconstructed")
    plt.axis("off")

plt.suptitle("Original and Convolutional RBM Reconstructions")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. GENERATE NEW IMAGE SAMPLES
# --------------------------------------------------

plt.figure(figsize=(8, 4))

for i in range(6):
    generated = conv_rbm.generate(steps=100)

    plt.subplot(2, 3, i + 1)
    plt.imshow(generated, cmap="gray")
    plt.title(f"Sample {i + 1}")
    plt.axis("off")

plt.suptitle("Images Generated Using Gibbs Sampling")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. TRAINING LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), conv_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Contrastive Divergence Loss")
plt.title("Convolutional RBM Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nExperiment 28 completed successfully.")

