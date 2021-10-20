import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tf.tf2.util import plot_loss_curve, plot_predictions

if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32')
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # Hyperparameters
    use_ckpt = False
    epochs = 10
    learning_rate = 0.001
    batch_size = 32
    dropout_rate = 0.5

    # Keras model
    checkpoint_path = "nets/cnn.ckpt"
    if use_ckpt and os.path.isdir(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    else:
        print("Building new model.")
        model = keras.models.Sequential(
            [
                keras.layers.Input((28, 28, 1)),
                keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='conv1'),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1'),
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='conv2'),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2'),
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv3'),
                keras.layers.Flatten(name='conv_to_dense'),
                keras.layers.Dense(units=128, name="dense", activation='relu'),
                keras.layers.Dropout(rate=dropout_rate),
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
