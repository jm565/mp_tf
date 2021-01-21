import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets.mnist import load_data
from tf.tf2.util import plot_loss_curve, plot_predictions


if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # Hyperparameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 32
    dropout_rate = 0.5
    corruption = 0.3

    keras.layers.GaussianNoise
    keras.layers.Dropout

    # Model parameters (use square numbers for n_hidden for visualization purposes)
    n_hidden_1 = 256
    n_hidden_2 = 64
    n_hidden_3 = 16
    n_input = 28*28  # flattened 28x28 images
    stddev = 0.1
    actf = tf.nn.relu

    # Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    mask = tf.placeholder(tf.float32, [None, n_input])

    # Model
    x_mask = mask * x  # corrupted images
    # Encoder
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev))
    b1 = tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev))
    enc1 = actf(tf.matmul(x_mask, W1) + b1)

    W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev))
    b2 = tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev))
    enc2 = actf(tf.matmul(enc1, W2) + b2)

    W3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev))
    b3 = tf.Variable(tf.random_normal([n_hidden_3], stddev=stddev))
    enc3 = actf(tf.matmul(enc2, W3) + b3)
    # Decoder
    W4 = tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2], stddev=stddev))
    b4 = tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev))
    dec1 = actf(tf.matmul(enc3, W4) + b4)

    W5 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], stddev=stddev))
    b5 = tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev))
    dec2 = actf(tf.matmul(dec1, W5) + b5)

    W6 = tf.Variable(tf.random_normal([n_hidden_1, n_input], stddev=stddev))
    b6 = tf.Variable(tf.random_normal([n_input], stddev=stddev))
    dec3 = actf(tf.matmul(dec2, W6) + b6)

    # Loss function
    loss = tf.reduce_mean(tf.square(x - dec3))

    # Optimizer and train op
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        print("Training...")
        # with tf.contrib.tfprof.ProfileContext('/home/johannes/devel/profile/') as pctx:
        for i in range(train_steps):
            batch_x, _ = mnist.train.next_batch(batch_size)
            mask_np = np.random.binomial(1, 1 - corruption, batch_x.shape)
            # run train op and compute batch loss
            _, batch_loss = sess.run([train_op, loss], feed_dict={x: batch_x, mask: mask_np})
            if (i+1) % 500 == 0:
                print("Train step {}:\n Batch loss = {}".format(i+1, batch_loss))
        print("Optimization finished.")

        # Test model
        mask_np = np.random.binomial(1, 1 - corruption, mnist.test.images.shape)
        test_loss = sess.run(loss, feed_dict={x: mnist.test.images, mask: mask_np})
        print("Test loss = {:.3f}".format(test_loss))

        # Look at some examples (and plot them)
        num_samples = 5
        # take the first num_samples images
        test_images = mnist.test.images[:num_samples, :]
        # compute model layer outputs (including image reconstruction)
        mask_np = np.random.binomial(1, 1 - corruption, test_images.shape)
        imgs_enc1, imgs_enc2, imgs_enc3, imgs_dec1, imgs_dec2, imgs_dec3\
            = sess.run([enc1, enc2, enc3, dec1, dec2, dec3], feed_dict={x: test_images, mask: mask_np})
        # plot images
        test_images = test_images * mask_np
        plt.figure(figsize=(7, num_samples))
        for i in range(num_samples):
            h1_width = int(np.sqrt(n_hidden_1))
            h2_width = int(np.sqrt(n_hidden_2))
            h3_width = int(np.sqrt(n_hidden_3))
            img_in = test_images[i, :].reshape(28, 28)
            img_enc1 = imgs_enc1[i, :].reshape(h1_width, h1_width)
            img_enc2 = imgs_enc2[i, :].reshape(h2_width, h2_width)
            img_enc3 = imgs_enc3[i, :].reshape(h3_width, h3_width)
            img_dec1 = imgs_dec1[i, :].reshape(h2_width, h2_width)
            img_dec2 = imgs_dec2[i, :].reshape(h1_width, h1_width)
            img_dec3 = imgs_dec3[i, :].reshape(28, 28)
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
            plt.imshow(img_dec3, cmap='gray')
            plt.axis("off")
            if i == 0:
                plt.title("Dec 3")

        # display figure
        # plt.subplots_adjust(hspace=0.5)
        plt.show()
