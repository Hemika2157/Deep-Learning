
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.keras.utils.set_random_seed(42)

# Settings
EPOCHS = 10
BATCH_SIZE = 128
LATENT_DIM = 2

os.makedirs("gan_samples", exist_ok=True)

# -----------------------------------------
# 1. LOAD FASHION-MNIST DATASET
# -----------------------------------------

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train[:20000].astype("float32") / 255.0
x_test = x_test[:2000].astype("float32") / 255.0

print("Training data:", x_train.shape)

# -----------------------------------------
# 2. BUILD VAE ENCODER
# -----------------------------------------

encoder_input = layers.Input(shape=(28, 28))

x = layers.Flatten()(encoder_input)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)


# Reparameterization
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon


z = Sampling()([z_mean, z_log_var])

encoder = Model(
    encoder_input,
    [z_mean, z_log_var, z],
    name="Encoder"
)

# -----------------------------------------
# 3. BUILD VAE DECODER
# -----------------------------------------

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

# -----------------------------------------
# 4. VAE MODEL AND LOSS
# -----------------------------------------

class VAE(Model):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.total_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_tracker,
            self.recon_tracker,
            self.kl_tracker
        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:

            mean, log_var, z = self.encoder(data, training=True)
            reconstructed = self.decoder(z, training=True)

            # Corrected reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        data, reconstructed
                    ),
                    axis=1
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mean) - tf.exp(log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + kl_loss

        gradients = tape.gradient(
            total_loss, self.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        self.total_tracker.update_state(total_loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)

        return {
            "loss": self.total_tracker.result(),
            "reconstruction": self.recon_tracker.result(),
            "kl_loss": self.kl_tracker.result()
        }


vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

print("\nTraining VAE...")

vae_history = vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -----------------------------------------
# 5. GENERATE VAE IMAGES
# -----------------------------------------

random_latent = tf.random.normal([10, LATENT_DIM])

vae_images = decoder.predict(
    random_latent, verbose=0
)

plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(vae_images[i], cmap="gray")
    plt.axis("off")

plt.suptitle("VAE Generated Images")
plt.tight_layout()
plt.savefig("vae_generated.png")
plt.show()

# -----------------------------------------
# 6. 2-D LATENT SPACE GRID
# -----------------------------------------

values = np.linspace(-3, 3, 15)

grid = np.array([
    [x, y]
    for y in values[::-1]
    for x in values
], dtype="float32")

grid_images = decoder.predict(grid, verbose=0)

plt.figure(figsize=(9, 9))

for i, img in enumerate(grid_images):
    plt.subplot(15, 15, i + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")

plt.suptitle("VAE Latent Space Grid")
plt.tight_layout()
plt.savefig("vae_latent_grid.png")
plt.show()

# -----------------------------------------
# 7. BUILD DCGAN GENERATOR
# -----------------------------------------

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


# -----------------------------------------
# 8. BUILD DCGAN DISCRIMINATOR
# -----------------------------------------

def build_discriminator():

    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ], name="Discriminator")


generator = build_generator()
discriminator = build_discriminator()

# -----------------------------------------
# 9. GAN LOSS AND OPTIMIZERS
# -----------------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

g_optimizer = tf.keras.optimizers.Adam(
    0.0002, beta_1=0.5
)

d_optimizer = tf.keras.optimizers.Adam(
    0.0002, beta_1=0.5
)


def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output), fake_output
    )


def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(
        tf.ones_like(real_output), real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output), fake_output
    )

    return real_loss + fake_loss


# -----------------------------------------
# 10. GAN TRAINING STEP
# -----------------------------------------

@tf.function
def train_step(real_images):

    noise = tf.random.normal(
        [tf.shape(real_images)[0], 100]
    )

    with tf.GradientTape() as gt, tf.GradientTape() as dt:

        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    g_gradients = gt.gradient(
        g_loss, generator.trainable_variables
    )

    d_gradients = dt.gradient(
        d_loss, discriminator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )

    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    return g_loss, d_loss


# -----------------------------------------
# 11. SAVE GENERATED GAN IMAGES
# -----------------------------------------

fixed_noise = tf.random.normal([16, 100])


def save_images(epoch):

    images = generator(
        fixed_noise, training=False
    ).numpy()

    images = (images + 1) / 2

    plt.figure(figsize=(6, 6))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.suptitle("GAN Samples - Epoch " + str(epoch))
    plt.tight_layout()

    filename = f"gan_samples/epoch_{epoch}.png"
    plt.savefig(filename)
    plt.close()

    print("Saved:", filename)


# -----------------------------------------
# 12. TRAIN DCGAN
# -----------------------------------------

gan_data = (x_train * 2 - 1)[..., np.newaxis]

dataset = tf.data.Dataset.from_tensor_slices(gan_data)
dataset = dataset.shuffle(10000).batch(
    BATCH_SIZE, drop_remainder=True
)

g_losses = []
d_losses = []

print("\nTraining DCGAN...")

for epoch in range(1, EPOCHS + 1):

    epoch_g = []
    epoch_d = []

    for batch in dataset:

        gl, dl = train_step(batch)

        epoch_g.append(float(gl.numpy()))
        epoch_d.append(float(dl.numpy()))

    g_avg = np.mean(epoch_g)
    d_avg = np.mean(epoch_d)

    g_losses.append(g_avg)
    d_losses.append(d_avg)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Generator Loss: {g_avg:.4f} | "
        f"Discriminator Loss: {d_avg:.4f}"
    )

    if epoch % 2 == 0:
        save_images(epoch)


# -----------------------------------------
# 13. FINAL GAN GENERATED IMAGES
# -----------------------------------------

final_images = generator(
    fixed_noise, training=False
).numpy()

final_images = (final_images + 1) / 2

plt.figure(figsize=(7, 7))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(final_images[i, :, :, 0], cmap="gray")
    plt.axis("off")

plt.suptitle("Final GAN Generated Fashion Images")
plt.tight_layout()
plt.savefig("gan_final_samples.png")
plt.show()

# -----------------------------------------
# 14. GENERATOR AND DISCRIMINATOR LOSS
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(g_losses, marker="o", label="Generator Loss")
plt.plot(d_losses, marker="s", label="Discriminator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DCGAN Loss Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("gan_loss.png")
plt.show()
decoder.save_weights("vae_decoder.weights.h5")
generator.save_weights("dcgan_generator.weights.h5")


# -----------------------------------------
# 15. VAE LOSS GRAPH
# -----------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(
    vae_history.history["loss"],
    label="VAE Total Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("vae_loss.png")
plt.show()

print("\nExperiment 29 completed successfully!")
print("Generated images are saved in:", os.path.abspath("gan_samples"))

