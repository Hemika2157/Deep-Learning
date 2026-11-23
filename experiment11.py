
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# 1. Load CIFAR-10 dataset
# --------------------------------------------------

print("Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

y_train = y_train.flatten()
y_test = y_test.flatten()

# Split original training data into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print("Training images:", len(x_train))
print("Validation images:", len(x_val))
print("Testing images:", len(x_test))


# --------------------------------------------------
# 2. Resize and normalise images
# --------------------------------------------------

# CIFAR-10 images are already 32 x 32.
# Resize to the standard CNN input size of 32 x 32.

x_train = tf.image.resize(x_train, (32, 32)).numpy()
x_val = tf.image.resize(x_val, (32, 32)).numpy()
x_test = tf.image.resize(x_test, (32, 32)).numpy()

# Normalise pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels into one-hot encoded format
y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)


# --------------------------------------------------
# 3. Build CNN model
# --------------------------------------------------

model = Sequential([
    Input(shape=(32, 32, 3)),

    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()


# --------------------------------------------------
# 4. Train CNN model
# --------------------------------------------------

history = model.fit(
    x_train,
    y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=10,
    batch_size=64,
    verbose=1
)


# --------------------------------------------------
# 5. Evaluate model on test data
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test, y_test_cat, verbose=0
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# --------------------------------------------------
# 6. Plot accuracy and loss curves
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Predict test images
# --------------------------------------------------

predictions = model.predict(x_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)


# --------------------------------------------------
# 8. Display confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. Display sample misclassified images
# --------------------------------------------------

wrong_indices = np.where(y_pred != y_test)[0]

print("\nNumber of misclassified images:", len(wrong_indices))

if len(wrong_indices) > 0:

    sample_indices = wrong_indices[:9]

    plt.figure(figsize=(10, 10))

    for i, index in enumerate(sample_indices):

        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[index])

        actual = class_names[y_test[index]]
        predicted = class_names[y_pred[index]]

        plt.title(
            f"Actual: {actual}\nPredicted: {predicted}",
            fontsize=9
        )

        plt.axis("off")

    plt.suptitle("Sample Misclassified Images")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 10. Visualise first convolution layer feature maps
# --------------------------------------------------

first_conv = next(
    layer for layer in model.layers
    if isinstance(layer, Conv2D)
)

feature_model = Model(
    inputs=model.input,
    outputs=first_conv.output
)

sample_image = x_test[0:1]

feature_maps = feature_model.predict(
    sample_image, verbose=0
)

plt.figure(figsize=(12, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
    plt.title(f"Feature Map {i + 1}")
    plt.axis("off")

plt.suptitle("First Convolution Layer Feature Maps")
plt.tight_layout()
plt.show()

print("\nCNN Image Classification Completed Successfully!")

