import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


# Data
num_datapoints = 13
data = np.arange(0, num_datapoints, dtype=np.float32)
labels = data * 3 + 2 + np.random.normal(0, 1, size=data.shape)
labels = labels.astype(np.float32)


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
    return history

# Hyperparameters
learning_rate = 0.1
num_epochs = 200
batch_size = len(data)

# Compile model
linear_model = sequential_model(learning_rate)

# Train model
hist = train_model(linear_model, data, labels, num_epochs, batch_size)
