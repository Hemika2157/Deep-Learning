
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm

# --------------------------------------------------
# 1. INITIALIZATION
# --------------------------------------------------

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

N_SAMPLES = 1000
LATENT_DIM = 2
BATCH_SIZE = 100

# Fashion-MNIST class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]

# --------------------------------------------------
# 2. LOAD FASHION-MNIST DATASET
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Fashion-MNIST loaded successfully.")

# --------------------------------------------------
# 3. REBUILD VAE DECODER
# --------------------------------------------------

decoder_input = layers.Input(shape=(LATENT_DIM,))

x = layers.Dense(128, activation="relu")(decoder_input)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(784, activation="sigmoid")(x)

decoder_output = layers.Reshape((28, 28))(x)

decoder = Model(
    decoder_input,
    decoder_output,
    name="Decoder"
)

# Load trained VAE weights
if not os.path.exists("vae_decoder.weights.h5"):
    raise FileNotFoundError(
        "VAE weights not found. Run Experiment 29 again "
        "after adding the model-saving lines."
    )

decoder.load_weights("vae_decoder.weights.h5")

print("VAE decoder loaded.")

# --------------------------------------------------
# 4. REBUILD DCGAN GENERATOR
# --------------------------------------------------

def build_generator():

    return tf.keras.Sequential([
        layers.Input(shape=(100,)),

        layers.Dense(7 * 7 * 128, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Reshape((7, 7, 128)),

        layers.Conv2DTranspose(
            64, 4, strides=2, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Conv2DTranspose(
            1, 4, strides=2, padding="same", activation="tanh"
        )
    ], name="Generator")


generator = build_generator()

if not os.path.exists("dcgan_generator.weights.h5"):
    raise FileNotFoundError(
        "GAN weights not found. Run Experiment 29 again "
        "after adding the model-saving lines."
    )

generator.load_weights("dcgan_generator.weights.h5")

print("DCGAN generator loaded.")

# --------------------------------------------------
# 5. GENERATE 1000 VAE SAMPLES
# --------------------------------------------------

print("\nGenerating VAE samples...")

vae_samples = []

for start in range(0, N_SAMPLES, BATCH_SIZE):

    size = min(BATCH_SIZE, N_SAMPLES - start)

    latent = tf.random.normal([size, LATENT_DIM])

    generated = decoder.predict(latent, verbose=0)

    vae_samples.append(generated)

vae_samples = np.concatenate(vae_samples, axis=0)

# --------------------------------------------------
# 6. GENERATE 1000 GAN SAMPLES
# --------------------------------------------------

print("Generating GAN samples...")

gan_samples = []

for start in range(0, N_SAMPLES, BATCH_SIZE):

    size = min(BATCH_SIZE, N_SAMPLES - start)

    noise = tf.random.normal([size, 100])

    generated = generator.predict(noise, verbose=0)

    # Convert [-1, 1] into [0, 1]
    generated = (generated + 1.0) / 2.0

    gan_samples.append(generated[:, :, :, 0])

gan_samples = np.concatenate(gan_samples, axis=0)

print("VAE samples:", vae_samples.shape)
print("GAN samples:", gan_samples.shape)

# Save generated samples
np.save("vae_1000_samples.npy", vae_samples)
np.save("gan_1000_samples.npy", gan_samples)

# --------------------------------------------------
# 7. BUILD FASHION-MNIST CLASSIFIER
# --------------------------------------------------

print("\nTraining Fashion-MNIST classifier...")

classifier_input = layers.Input(shape=(28, 28))

x = layers.Reshape((28, 28, 1))(classifier_input)

x = layers.Conv2D(
    32, 3, activation="relu", padding="same"
)(x)

x = layers.MaxPooling2D()(x)

x = layers.Conv2D(
    64, 3, activation="relu", padding="same"
)(x)

x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)

features = layers.Dense(
    128, activation="relu", name="features"
)(x)

output = layers.Dense(
    10, activation="softmax", name="class_output"
)(features)

classifier = Model(
    classifier_input,
    output,
    name="Fashion_Classifier"
)

classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

classifier.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_accuracy = classifier.evaluate(
    x_test, y_test, verbose=0
)

print("Classifier test accuracy:", round(test_accuracy, 4))

# Feature extraction model
feature_model = Model(
    classifier.input,
    classifier.get_layer("features").output
)

# --------------------------------------------------
# 8. CLASSIFIER PREDICTIONS
# --------------------------------------------------

print("\nClassifying generated images...")

real_prob = classifier.predict(
    x_test[:N_SAMPLES], verbose=0
)

vae_prob = classifier.predict(
    vae_samples, verbose=0
)

gan_prob = classifier.predict(
    gan_samples, verbose=0
)

vae_classes = np.argmax(vae_prob, axis=1)
gan_classes = np.argmax(gan_prob, axis=1)

# --------------------------------------------------
# 9. CLASSIFIER-BASED INCEPTION SCORE
# --------------------------------------------------

def calculate_inception_score(probabilities, splits=10):

    scores = []

    parts = np.array_split(probabilities, splits)

    for part in parts:

        marginal = np.mean(part, axis=0)

        kl = part * (
            np.log(part + 1e-10)
            - np.log(marginal + 1e-10)
        )

        kl = np.sum(kl, axis=1)

        score = np.exp(np.mean(kl))

        scores.append(score)

    return np.mean(scores), np.std(scores)


vae_is, vae_is_std = calculate_inception_score(vae_prob)
gan_is, gan_is_std = calculate_inception_score(gan_prob)

print("\nCLASSIFIER-BASED INCEPTION SCORE")
print("VAE:", round(vae_is, 4), "+/-", round(vae_is_std, 4))
print("GAN:", round(gan_is, 4), "+/-", round(gan_is_std, 4))

# --------------------------------------------------
# 10. EXTRACT FEATURES FOR FID
# --------------------------------------------------

print("\nExtracting image features...")

real_features = feature_model.predict(
    x_test[:N_SAMPLES], verbose=0
)

vae_features = feature_model.predict(
    vae_samples, verbose=0
)

gan_features = feature_model.predict(
    gan_samples, verbose=0
)

# --------------------------------------------------
# 11. FRECHET DISTANCE CALCULATION
# --------------------------------------------------

def calculate_fid(real, generated):

    mu1 = np.mean(real, axis=0)
    mu2 = np.mean(generated, axis=0)

    sigma1 = np.cov(real, rowvar=False)
    sigma2 = np.cov(generated, rowvar=False)

    diff = mu1 - mu2

    covariance_mean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real

    fid = (
        np.sum(diff ** 2)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covariance_mean)
    )

    return max(float(fid), 0.0)


vae_fid = calculate_fid(real_features, vae_features)
gan_fid = calculate_fid(real_features, gan_features)

print("\nFRECHET DISTANCE")
print("VAE FID:", round(vae_fid, 4))
print("GAN FID:", round(gan_fid, 4))

# --------------------------------------------------
# 12. CLASS DISTRIBUTION AND MODE COLLAPSE
# --------------------------------------------------

vae_counts = np.bincount(
    vae_classes, minlength=10
)

gan_counts = np.bincount(
    gan_classes, minlength=10
)


def calculate_diversity(features):

    # Average Euclidean distance from feature mean
    center = np.mean(features, axis=0)

    distances = np.linalg.norm(
        features - center, axis=1
    )

    return float(np.mean(distances))


def calculate_entropy(counts):

    probabilities = counts / np.sum(counts)

    return float(
        -np.sum(
            probabilities * np.log2(probabilities + 1e-10)
        )
    )


vae_diversity = calculate_diversity(vae_features)
gan_diversity = calculate_diversity(gan_features)

vae_entropy = calculate_entropy(vae_counts)
gan_entropy = calculate_entropy(gan_counts)

vae_coverage = np.count_nonzero(vae_counts)
gan_coverage = np.count_nonzero(gan_counts)

print("\nCLASS DISTRIBUTION")

for i in range(10):

    print(
        class_names[i],
        "| VAE:", vae_counts[i],
        "| GAN:", gan_counts[i]
    )

print("\nCLASS COVERAGE")
print("VAE:", vae_coverage, "/ 10")
print("GAN:", gan_coverage, "/ 10")

# --------------------------------------------------
# 13. PLOT CLASS DISTRIBUTION
# --------------------------------------------------

x = np.arange(10)
width = 0.35

plt.figure(figsize=(11, 5))

plt.bar(
    x - width / 2,
    vae_counts,
    width,
    label="VAE"
)

plt.bar(
    x + width / 2,
    gan_counts,
    width,
    label="GAN"
)

plt.xticks(x, class_names, rotation=40, ha="right")
plt.ylabel("Number of Generated Images")
plt.title("Generated Image Class Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# --------------------------------------------------
# 14. PCA FEATURE DISTRIBUTION
# --------------------------------------------------

print("\nPreparing PCA visualization...")

all_features = np.vstack([
    real_features,
    vae_features,
    gan_features
])

pca = PCA(n_components=2)

reduced = pca.fit_transform(all_features)

real_2d = reduced[:N_SAMPLES]

vae_2d = reduced[
    N_SAMPLES:2 * N_SAMPLES
]

gan_2d = reduced[
    2 * N_SAMPLES:
]

plt.figure(figsize=(9, 7))

plt.scatter(
    real_2d[:, 0], real_2d[:, 1],
    s=10, alpha=0.35, label="Real"
)

plt.scatter(
    vae_2d[:, 0], vae_2d[:, 1],
    s=10, alpha=0.35, label="VAE"
)

plt.scatter(
    gan_2d[:, 0], gan_2d[:, 1],
    s=10, alpha=0.35, label="GAN"
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Real and Generated Feature Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("pca_comparison.png")
plt.show()

# --------------------------------------------------
# 15. DISPLAY GENERATED IMAGE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for i in range(5):

    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title("Real")
    plt.axis("off")

    plt.subplot(3, 5, i + 6)
    plt.imshow(vae_samples[i], cmap="gray")
    plt.title("VAE")
    plt.axis("off")

    plt.subplot(3, 5, i + 11)
    plt.imshow(gan_samples[i], cmap="gray")
    plt.title("GAN")
    plt.axis("off")

plt.suptitle("Real, VAE and GAN Image Comparison")
plt.tight_layout()
plt.savefig("image_comparison.png")
plt.show()

# --------------------------------------------------
# 16. CREATE METRICS TABLE USING PANDAS
# --------------------------------------------------

results = pd.DataFrame({

    "Metric": [
        "Inception Score",
        "FID",
        "Feature Diversity",
        "Class Entropy",
        "Class Coverage"
    ],

    "VAE": [
        round(vae_is, 4),
        round(vae_fid, 4),
        round(vae_diversity, 4),
        round(vae_entropy, 4),
        vae_coverage
    ],

    "GAN": [
        round(gan_is, 4),
        round(gan_fid, 4),
        round(gan_diversity, 4),
        round(gan_entropy, 4),
        gan_coverage
    ]
})

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv("model_comparison.csv", index=False)

# --------------------------------------------------
# 17. AUTOMATIC MODEL RECOMMENDATION
# --------------------------------------------------

print("\nMODEL RECOMMENDATION")

# Higher IS, diversity, entropy and coverage are preferred.
# Lower FID is preferred.

vae_score = (
    (vae_is > gan_is)
    + (vae_fid < gan_fid)
    + (vae_diversity > gan_diversity)
    + (vae_entropy > gan_entropy)
    + (vae_coverage > gan_coverage)
)

gan_score = (
    (gan_is > vae_is)
    + (gan_fid < vae_fid)
    + (gan_diversity > vae_diversity)
    + (gan_entropy > vae_entropy)
    + (gan_coverage > vae_coverage)
)

if vae_score > gan_score:
    print("VAE performs better on more evaluation metrics.")
    print("VAE is the preliminary recommendation.")

elif gan_score > vae_score:
    print("GAN performs better on more evaluation metrics.")
    print("GAN is the preliminary recommendation.")

else:
    print("Both models have mixed evaluation results.")
    print("Further evaluation is recommended.")

print("\nVAE score:", vae_score, "/ 5")
print("GAN score:", gan_score, "/ 5")

print("\nExperiment 30 completed successfully!")
print("Results saved in model_comparison.csv")

