import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


def plot_model(weight, bias, inputs, targets, name="", show=False, save_path=None):
    """Plot the trained model against the training inputs and targets."""
    # Create separate figure
    plt.figure(name)
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if show:
        # Render the plot
        plt.show()
    else:
        # Or close the current figure
        plt.close()


def plot_loss_curve(epochs, losses, show=False, save_path=None):
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if show:
        # Render the plot
        plt.show()
    else:
        # Or close the current figure
        plt.close()


# Setup
show = False

# Data
num_datapoints = 13
data = np.arange(0, num_datapoints, dtype=np.float32)
labels = data * 3 + 2 + np.random.normal(0, 1, size=data.shape)
labels = labels.astype(np.float32)

# Plot data
plt.figure("Linear data")
plt.scatter(data, labels)
plt.xlabel("Inputs")
plt.ylabel("Targets")
plt.savefig("lin_regression_data.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
if show:
    plt.show()
else:
    plt.close()


# Keras sequential model
def sequential_model(learning_rate):
    """Create and compile a simple linear regression model."""
    # Create model layers
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(1,)),  # shape of a single input
            keras.layers.Dense(units=1)  # fully-connected layer
        ]
    )
    # Configure model
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=keras.losses.mean_squared_error,
                  metrics=[keras.metrics.mean_squared_error])
    return model


# Train function
def train_model(model, inputs, targets, epochs, batch_size):
    """Train the model by feeding it data."""
    # Feed input and target data to the model in chunks of batch_size
    # The model will train for a certain number of epochs and save information in the returned history
    history = model.fit(x=inputs,
                        y=targets,
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    weight = model.get_weights()[0]
    bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of the history.
    epoch_history = history.epoch
    # Gather the model's loss at each epoch
    loss_history = history.history["loss"]
    return weight, bias, epoch_history, loss_history


# Hyperparameters
learning_rate = 0.1
num_epochs = 200
batch_size = len(data)

# Compile model
linear_model = sequential_model(learning_rate)

# Plot initial model and summary
print(linear_model.summary())
initial_vars = [var.numpy() for var in linear_model.trainable_variables]
plot_model(*initial_vars, data, labels,
           name="Initial model", show=show, save_path="lin_regression_model_initial.png")

# Train model
tr_weight, tr_bias, epochs, losses = train_model(linear_model, data, labels, num_epochs, batch_size)

# Plot trained model, parameters and loss curve
print(f"Trained parameters: w = {tr_weight}, b = {tr_bias}")
plot_model(tr_weight, tr_bias, data, labels,
           name="Trained model", show=show, save_path="lin_regression_model_trained.png")
plot_loss_curve(epochs, losses, show=show, save_path="lin_regression_loss_curve.png")
