import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


def plot_images(images, labels=None, label_colors=None):
    grid_size = np.sqrt(images.shape[0]).astype(np.int32)
    for i in range(grid_size ** 2):
        plt.subplot(grid_size, grid_size, i + 1)
        if labels is not None:
            if label_colors is not None:
                plt.title(labels[i], {'color': label_colors[i]})
            else:
                plt.title(labels[i])
        plt.axis('off')
        plt.imshow(images[i], cmap='gray')
    plt.tight_layout()
    plt.show()


def plot_predictions(images, predictions, labels):
    probs = tf.nn.softmax(predictions)  # class probabilities as softmax over logits
    predictions = np.argmax(probs, axis=1)  # most probable class prediction
    correct = predictions == labels  # correct or incorrect predictions
    colors = [("green" if correct[i] else "red") for i in range(correct.shape[0])]  # assign colors for visualization
    info = [f"{predictions[i]} ({100 * np.max(probs[i]):3.2f}%)" for i in range(len(predictions))]  # build titles
    plot_images(images.reshape(-1, 28, 28), info, colors)


# Load mnist data (NumPy arrays)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # norm to [0,1]
x_test = x_test / 255.0  # norm to [0,1]

# # Plot some example images
# indices = np.random.randint(0, x_train.shape[0], size=25)
# example_images = x_train[indices]
# plot_images(example_images)


def sequential_model(learning_rate):
    """Create and compile a model using the Sequential API."""
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(28, 28), name="input_images"),
            keras.layers.Flatten(name="flat_images"),
            keras.layers.Dense(256, activation='relu', name="hidden_1"),
            keras.layers.Dense(128, activation='relu', name="hidden_2"),
            keras.layers.Dense(10, name="output")
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


seq_model = sequential_model(0.001)
print(seq_model.summary())
print("Training...")
seq_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)
print("Testing...")
seq_model.evaluate(x_test, y_test, batch_size=32, verbose=1)

# Get some predictions and plot the corresponding images
num_samples = 25
indices = np.random.randint(0, x_test.shape[0], size=num_samples)
test_imgs = x_test[indices]
test_labels = y_test[indices]
pred = seq_model.predict(test_imgs)
plot_predictions(test_imgs, pred, test_labels)


def functional_model(learning_rate):
    """Create and compile a model using the Functional API."""
    inputs = keras.layers.Input(shape=(28, 28), name='input_layer')
    flatten = keras.layers.Flatten(name='flatten')(inputs)
    l1 = keras.layers.Dense(256, activation='relu', name='hidden_1')(flatten)
    l2 = keras.layers.Dense(128, activation='relu', name='hidden_2')(l1)
    # l2 = keras.layers.Dense(128, activation='relu', name='hidden_2')(l1)
    # l3 = keras.layers.Dense(128, activation='relu', name='hidden_3')(l1)
    # l_2_3 = tf.concat([l2, l3], axis=1)
    outputs = keras.layers.Dense(10, activation='softmax', name='output_layer')(l2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    return model


# func_model = functional_model(0.001)
# print(func_model.summary())
# print("Training...")
# func_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)
# print("Testing...")
# func_model.evaluate(x_test, y_test, batch_size=32, verbose=1)
#
# # Get some predictions and plot the corresponding images
# num_samples = 25
# indices = np.random.randint(0, x_test.shape[0], size=num_samples)
# test_imgs = x_test[indices]
# test_labels = y_test[indices]
# pred = func_model.predict(test_imgs)
# plot_predictions(test_imgs, pred, test_labels)



