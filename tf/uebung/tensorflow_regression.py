import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_model(weight, bias, inputs, targets, save_path=None):
    """Plot the trained model against the training inputs and targets."""
    # Create separate figure
    plt.figure("Model evaluation")
    # Label the axes
    plt.xlabel("Inputs")
    plt.ylabel("Targets")
    # Plot the datapoints
    plt.scatter(inputs, targets)
    # Create a red line representing the model
    x0 = 0
    y0 = bias
    x1 = inputs[-1]
    y1 = weight * x1 + bias
    plt.plot([x0, x1], [y0, y1], c='r')
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)
    # Render the plot
    plt.show()


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


# Data
num_datapoints = 13
data = np.arange(0, num_datapoints, dtype=np.float32)
labels = data * 3 + 2 + np.random.normal(0, 1, size=data.shape)
labels = labels.astype(np.float32)

# Plot data
plt.figure()
plt.scatter(data, labels)
plt.xlabel("Inputs")
plt.ylabel("Targets")
plt.savefig("lin_regression_data.png", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()

# Model parameters (static initialization)
w = tf.Variable([0.5])
b = tf.Variable([10.0])

# Actual model
def linear_model(inputs):
    return w * inputs + b

# Plot initial model
plot_model(w, b, data, labels, save_path="lin_regression_model_initial.png")


# loss (mean of squared errors)
def loss_fn(inputs, targets):
    return tf.reduce_mean(tf.square(targets - inputs))
    # return tf.keras.losses.mean_squared_error(targets, inputs)


# gradients (TensorFlow automatic differentiation within GradientTape block)
@tf.function
def gradients(inputs, targets):
    with tf.GradientTape() as tape:
        outputs = linear_model(inputs)
        loss = loss_fn(outputs, targets)
        grads = tape.gradient(loss, [w, b])
    return loss, grads


# optimizer
learning_rate = 0.1
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training loop
loss_history = []
num_epochs = 300
for epoch in range(num_epochs):
    # Compute loss and respective gradients for the model
    loss, grads = gradients(data, labels)
    # Apply a step of the optimizer on the model parameters
    optimizer.apply_gradients(zip(grads, [w, b]))
    # Track the loss for each epoch
    loss_history.append(loss)
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Loss = {loss:.3f}")

# Plot trained model, parameters and loss curve
print(f"Trained parameters: w = {w.numpy()}, b = {b.numpy()}")
plot_model(w, b, data, labels, save_path="lin_regression_model_trained.png")
plot_loss_curve(list(range(num_epochs)), loss_history, save_path="lin_regression_loss_curve.png")
