import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt


def plot_loss_curve(epochs, losses, save_path=None):
    """Plot the loss curve, which shows loss vs. epoch."""
    # Create separate figure
    plt.figure("Loss curve")
    # Label the axes
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Plot loss curve
    plt.plot(epochs, losses, label="Loss")
    # Attach legend
    plt.legend()
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)
    # Render the plot
    plt.show()


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
    plt.show()


def plot_predictions(images, logits, gt_labels):
    probs = tf.nn.softmax(logits)  # class probabilities as softmax over logits
    predictions = np.argmax(probs, axis=1)  # most probable class prediction
    correct = predictions == gt_labels  # correct or incorrect predictions
    colors = [("green" if correct[i] else "red") for i in range(correct.shape[0])]  # assign colors for visualization
    info = [f"{predictions[i]} ({100 * np.max(probs[i]):3.2f}%)" for i in range(len(predictions))]  # build titles
    plot_images(images.reshape(-1, 28, 28), info, colors)


if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Hyperparameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Keras model
    checkpoint_path = "nets/ffn.ckpt"
    if os.path.isdir(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    else:
        print("Building new model.")
        model = keras.models.Sequential(
            [
                keras.layers.Input((28,28)),
                keras.layers.Flatten(name="input"),
                keras.layers.Dense(units=256, activation='relu', name="hidden_1"),
                keras.layers.Dense(units=128, activation='relu', name="hidden_2"),
                keras.layers.Dense(units=10, name="output")
            ]
        )
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics="sparse_categorical_accuracy")

    # Model summary
    print(model.summary())

    # Setup checkpointing
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor="loss",
                                                 mode="min",
                                                 verbose=1,
                                                 save_best_only=True)

    # Model training
    print("Training.")
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])
    epoch_history = history.epoch
    loss_history = history.history["loss"]
    plot_loss_curve(epoch_history, loss_history)

    # Test model
    print("Testing.")
    model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)

    # Get some predictions and plot the corresponding images
    num_samples = 25
    indices = np.random.randint(0, x_test.shape[0], size=num_samples)
    test_imgs = x_test[indices]
    test_labels = y_test[indices]
    pred = model.predict(test_imgs)
    plot_predictions(test_imgs, pred, test_labels)
