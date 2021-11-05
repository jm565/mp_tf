import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tf.tf2.util import plot_loss_curve, plot_predictions

if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train / 255.0  # norm to [0,1]
    x_test = x_test / 255.0  # norm to [0,1]

    # Hyperparameters
    use_ckpt = False
    epochs = 10
    learning_rate = 0.001
    batch_size = 32
    n_classes = 10

    # Model parameters
    hidden_units = [256, 128]
    dropout_rates = [0.3, False]

    # Keras model
    checkpoint_path = "nets/ffn.ckpt"
    if use_ckpt and os.path.isdir(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    else:
        print("Building new model.")
        # # Variation 1 (hard-coded)
        # model = keras.models.Sequential(
        #     [
        #         keras.layers.Input((28, 28)),
        #         keras.layers.Flatten(name="input"),
        #         keras.layers.Dense(units=hidden_units[0], activation='relu', name="hidden_1"),
        #         keras.layers.Dropout(rate=dropout_rates[0], name="dropout_1"),
        #         keras.layers.Dense(units=hidden_units[1], activation='relu', name="hidden_2"),
        #         keras.layers.Dropout(rate=dropout_rates[1], name="dropout_2"),
        #         keras.layers.Dense(units=10, name="output")
        #     ]
        # )

        # Variation 2 (dynamic)
        model = keras.models.Sequential()
        model.add(keras.layers.Input((28, 28), name="img_input"))
        model.add(keras.layers.Flatten(name="flat_input"))
        for i, (hidden, dropout) in enumerate(zip(hidden_units, dropout_rates)):
            model.add(keras.layers.Dense(units=hidden, activation="relu", name=f"hidden_{i+1}"))
            if dropout:
                model.add(keras.layers.Dropout(rate=dropout, name=f"dropout_{i+1}"))
        model.add(keras.layers.Dense(units=n_classes, name="output"))

        # Compile model
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
