from pathlib import Path

import numpy as np  # linear algebra
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

# Let us define some paths first
input_path = Path("/valohai/inputs/example-data")

# Path to training images and corresponding labels provided as numpy arrays
kmnist_train_images_path = input_path / "kmnist-train-imgs.npz"
kmnist_train_labels_path = input_path / "kmnist-train-labels.npz"

# Path to the test images and corresponding labels
kmnist_test_images_path = input_path / "kmnist-test-imgs.npz"
kmnist_test_labels_path = input_path / "kmnist-test-labels.npz"

# Load the training data from the corresponding npz files
kmnist_train_images = np.load(kmnist_train_images_path)["arr_0"]
kmnist_train_labels = np.load(kmnist_train_labels_path)["arr_0"]

# Load the test data from the corresponding npz files
kmnist_test_images = np.load(kmnist_test_images_path)["arr_0"]
kmnist_test_labels = np.load(kmnist_test_labels_path)["arr_0"]

print(
    f"Number of training samples: {len(kmnist_train_images)} where each sample is of size: {kmnist_train_images.shape[1:]}"
)
print(
    f"Number of test samples: {len(kmnist_test_images)} where each sample is of size: {kmnist_test_images.shape[1:]}"
)

# A bunch of variables. The variable have the same value as given in the keras example
batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# input shape
input_shape = (img_rows, img_cols, 1)

# Process the train and test data in the exact same manner as done for MNIST
x_train = kmnist_train_images.astype("float32")
x_test = kmnist_test_images.astype("float32")
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# convert class vectors to binary class matrices
y_train = to_categorical(kmnist_train_labels, num_classes)
y_test = to_categorical(kmnist_test_labels, num_classes)

# Build and train the model.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
)

model.save("/valohai/outputs/model.h5")

# Check the test loss and test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1] * 100)
