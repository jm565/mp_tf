import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tf.tf2.util import plot_loss_curve

if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train / 255.0  # norm to [0,1]
    x_test = x_test / 255.0  # norm to [0,1]
    img_shape = x_train[0].shape
    flat_shape = img_shape[0] * img_shape[1]

    # Hyperparameters
    use_ckpt = True
    epochs = 1
    learning_rate = 0.001
    batch_size = 32
    corruption_rate = 0.5

    # Model parameters
    hidden_units = [256, 64, 16]

    # Keras model
    checkpoint_path = "nets/ae.ckpt"
    if use_ckpt and os.path.isdir(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    else:
        print("Building new model.")
        # Functional API
        image = keras.layers.Input(img_shape, name="image_input")
        corruption = keras.layers.Input(img_shape, name="corruption_mask")
        corrupted_img = keras.layers.Multiply(name="corrupted_image")([image, corruption])
        flat_input = keras.layers.Flatten(name="flat_input")(corrupted_img)
        enc_1 = keras.layers.Dense(units=hidden_units[0], activation='relu', name="encoder_1")(flat_input)
        enc_2 = keras.layers.Dense(units=hidden_units[1], activation='relu', name="encoder_2")(enc_1)
        enc_3 = keras.layers.Dense(units=hidden_units[2], activation='relu', name="encoder_3")(enc_2)
        dec_3 = keras.layers.Dense(units=hidden_units[1], activation='relu', name="decoder_3")(enc_3)
        dec_2 = keras.layers.Dense(units=hidden_units[0], activation='relu', name="decoder_2")(dec_3)
        dec_1 = keras.layers.Dense(units=flat_shape, activation='relu', name="decoder_1")(dec_2)
        output = keras.layers.Reshape(target_shape=img_shape, name="output")(dec_1)
        model = keras.Model(inputs=[image, corruption], outputs=output)

        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                      # optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics="mean_squared_error")

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
    corruption_masks = np.random.binomial(1, 1 - corruption_rate, x_train.shape)
    history = model.fit(x=[x_train, corruption_masks], y=x_train,
                        batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])
    epoch_history = history.epoch
    loss_history = history.history["loss"]
    plot_loss_curve(epoch_history, loss_history)

    # Test model
    print("Testing.")
    corruption_masks = np.random.binomial(1, 1 - corruption_rate, x_test.shape)
    model.evaluate(x=[x_test, corruption_masks], y=x_test, batch_size=batch_size, verbose=1)

    # Get some samples and plot the corresponding images
    num_samples = 5
    indices = np.random.randint(0, x_test.shape[0], size=num_samples)
    test_imgs = x_test[indices]
    corruption_masks = np.random.binomial(1, 1 - corruption_rate, test_imgs.shape)

    relevant_layer_names = ["corrupted_image", "encoder_1", "encoder_2", "encoder_3",
                            "decoder_3", "decoder_2", "output"]
    relevant_layers = [model.get_layer(layer_name) for layer_name in relevant_layer_names]
    temp_model = keras.Model(inputs=model.input, outputs=[layer.output for layer in relevant_layers])
    predictions = temp_model([test_imgs, corruption_masks])
    outputs = dict(zip(relevant_layer_names, predictions))

    plt.figure(figsize=(7, num_samples))
    for i in range(num_samples):
        h1_width = int(np.sqrt(hidden_units[0]))
        h2_width = int(np.sqrt(hidden_units[1]))
        h3_width = int(np.sqrt(hidden_units[2]))
        img_in = outputs["corrupted_image"].numpy()[i, :, :]
        img_enc1 = outputs["encoder_1"].numpy()[i, :].reshape(h1_width, h1_width)
        img_enc2 = outputs["encoder_2"].numpy()[i, :].reshape(h2_width, h2_width)
        img_enc3 = outputs["encoder_3"].numpy()[i, :].reshape(h3_width, h3_width)
        img_dec1 = outputs["decoder_3"].numpy()[i, :].reshape(h2_width, h2_width)
        img_dec2 = outputs["decoder_2"].numpy()[i, :].reshape(h1_width, h1_width)
        img_out = outputs["output"].numpy()[i, :, :]
        # plot input image
        plt.subplot(num_samples, 7, 7 * i + 1)
        plt.imshow(img_in, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Input")
        # plot encoder layer 1
        plt.subplot(num_samples, 7, 7 * i + 2)
        plt.imshow(img_enc1, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Enc 1")
        # plot encoder layer 2
        plt.subplot(num_samples, 7, 7 * i + 3)
        plt.imshow(img_enc2, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Enc 2")
        # plot encoder layer 3
        plt.subplot(num_samples, 7, 7 * i + 4)
        plt.imshow(img_enc3, cmap='gray')
        plt.gray()
        plt.axis("off")
        if i == 0:
            plt.title("Enc 3")
        # plot decoder layer 1
        plt.subplot(num_samples, 7, 7 * i + 5)
        plt.imshow(img_dec1, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Dec 1")
        # plot decoder layer 2
        plt.subplot(num_samples, 7, 7 * i + 6)
        plt.imshow(img_dec2, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Dec 2")
        # plot decoder layer 3 (reconstructed image)
        plt.subplot(num_samples, 7, 7 * i + 7)
        plt.imshow(img_out, cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.title("Output")

    # display figure
    # plt.subplots_adjust(hspace=0.5)
    plt.show()
