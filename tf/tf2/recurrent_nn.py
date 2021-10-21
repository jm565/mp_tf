import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tf.tf2.util import plot_loss_curve, plot_predictions
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell
# from tensorflow.contrib.rnn import static_rnn, static_bidirectional_rnn


if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train / 255.0  # norm to [0,1]
    x_test = x_test / 255.0  # norm to [0,1]
    n_classes = 10
    img_shape = x_train[0].shape
    n_timesteps, n_input = img_shape[0], img_shape[1]  # interpret images as sequences of vectors

    # Hyperparameters
    use_ckpt = False
    epochs = 10
    learning_rate = 0.001
    batch_size = 32
    use_bidirectional_rnn = True
    use_sequence_output = True

    # Model parameters
    hidden_1 = 64
    hidden_2 = 32
    num_fc = 32

    # Keras model
    checkpoint_path = "nets/rnn.ckpt"
    if use_ckpt and os.path.isdir(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    else:
        print("Building new model.")
        # Functional API
        image = keras.layers.Input(img_shape, name="image")
        if use_bidirectional_rnn:
            out = keras.layers.Bidirectional(
                keras.layers.LSTM(hidden_1, return_sequences=use_sequence_output, name="LSTM1"),
                name="BLSTM")(image)
        else:
            out = keras.layers.LSTM(hidden_1, return_sequences=use_sequence_output, name="LSTM1")(image)
        if use_sequence_output:
            out = keras.layers.LSTM(hidden_2, name="LSTM2")(out)
        # if use_sequence_output:
        #     lstm_stack = keras.layers.Reshape(target_shape=(-1,), name="flat_LSTM")(out)
        #     out = keras.layers.Dense(num_fc, name="FC")(lstm_stack)
        output = keras.layers.Dense(n_classes, name="output")(out)
        model = keras.Model(inputs=image, outputs=output)

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
