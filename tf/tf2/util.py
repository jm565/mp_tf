import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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
