# Commented out IPython magic to ensure Python compatibility.
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# %matplotlib inline
np.random.seed(1)


def summary(model):
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    result = []
    for layer in model.layers:
        descriptors = [
            layer.__class__.__name__,
            layer.output_shape,
            layer.count_params(),
        ]
        if type(layer) == Conv2D:
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if type(layer) == MaxPooling2D:
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if type(layer) == Dropout:
            descriptors.append(layer.rate)
        if type(layer) == ZeroPadding2D:
            descriptors.append(layer.padding)
        if type(layer) == Dense:
            descriptors.append(layer.activation.__name__)
        result.append(descriptors)
    return result


def comparator(learner, instructor):
    if learner == instructor:
        for a, b in zip(learner, instructor):
            if tuple(a) != tuple(b):
                print(
                    colored("Test failed", attrs=["bold"]),
                    "\n Expected value \n\n",
                    colored(f"{b}", "green"),
                    "\n\n does not match the input value: \n\n",
                    colored(f"{a}", "red"),
                )
                raise AssertionError("Error in test")
        print(colored("All tests passed!", "green"))

    else:
        print(colored("Test failed. Your output is not as expected output.", "red"))


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def load_signs_dataset():
    train_dataset = h5py.File("datasets/train_signs.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_signs.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_happy_dataset():
    train_dataset = h5py.File("datasets/train_happy.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_happy.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

index = 124
plt.imshow(X_train_orig[index])  # display sample training image
plt.show()


def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = tf.keras.Sequential(
        [
            # ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(64, 64, 3)),
            # Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1)),
            # BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis=3),
            # ReLU
            tf.keras.layers.ReLU(),
            # Max Pooling 2D with default parameters
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten layer
            tf.keras.layers.Flatten(),
            # Dense layer with 1 unit for output & 'sigmoid' activation
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


happy_model = happyModel()

output = [
    ["ZeroPadding2D", (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
    ["Conv2D", (None, 64, 64, 32), 4736, "valid", "linear", "GlorotUniform"],
    ["BatchNormalization", (None, 64, 64, 32), 128],
    ["ReLU", (None, 64, 64, 32), 0],
    ["MaxPooling2D", (None, 32, 32, 32), 0, (2, 2), (2, 2), "valid"],
    ["Flatten", (None, 32768), 0],
    ["Dense", (None, 1), 32769, "sigmoid"],
]


happy_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


happy_model.summary()


happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)


happy_model.evaluate(X_test, Y_test)


# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()


# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:, index])))


X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """

    input_img = tf.keras.Input(shape=input_shape)

    # CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(8, (4, 4), strides=(1, 1), padding="SAME")(input_img)

    # RELU
    A1 = tf.keras.layers.ReLU()(Z1)

    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding="SAME")(
        A1
    )

    # CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(16, (2, 2), strides=(1, 1), padding="SAME")(P1)

    # RELU
    A2 = tf.keras.layers.ReLU()(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="SAME")(
        A2
    )

    # FLATTEN
    F = tf.keras.layers.Flatten()(P2)

    # Dense layer
    # 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    outputs = tf.keras.layers.Dense(6, activation="softmax")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


conv_model = convolutional_model((64, 64, 3))
conv_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
conv_model.summary()

output = [
    ["InputLayer", [(None, 64, 64, 3)], 0],
    ["Conv2D", (None, 64, 64, 8), 392, "same", "linear", "GlorotUniform"],
    ["ReLU", (None, 64, 64, 8), 0],
    ["MaxPooling2D", (None, 8, 8, 8), 0, (8, 8), (8, 8), "same"],
    ["Conv2D", (None, 8, 8, 16), 528, "same", "linear", "GlorotUniform"],
    ["ReLU", (None, 8, 8, 16), 0],
    ["MaxPooling2D", (None, 2, 2, 16), 0, (4, 4), (4, 4), "same"],
    ["Flatten", (None, 64), 0],
    ["Dense", (None, 6), 390, "softmax"],
]


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)


history.history


# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on.
df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[["loss", "val_loss"]]
df_loss.rename(columns={"loss": "train", "val_loss": "validation"}, inplace=True)
df_acc = df_loss_acc[["accuracy", "val_accuracy"]]
df_acc.rename(columns={"accuracy": "train", "val_accuracy": "validation"}, inplace=True)
df_loss.plot(title="Model loss", figsize=(12, 8)).set(xlabel="Epoch", ylabel="Loss")
df_acc.plot(title="Model Accuracy", figsize=(12, 8)).set(
    xlabel="Epoch", ylabel="Accuracy"
)
