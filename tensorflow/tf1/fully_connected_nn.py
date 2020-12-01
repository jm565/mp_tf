import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    # Import MNIST data
    mnist = read_data_sets("data", one_hot=True)

    # Hyperparameters
    train_steps = 5000
    learning_rate = 0.001
    batch_size = 100

    # Model parameters
    n_hidden_1 = 256
    n_hidden_2 = 256
    n_classes = 10  # 10 classes (numbers 0-9)
    n_input = 28*28  # flattened 28x28 images

    # Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Model
    # First layer
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], 0.0, 0.1))
    b1 = tf.Variable(tf.random_normal([n_hidden_1], 0.0, 0.1))
    layer1 = tf.matmul(x, W1) + b1
    layer1 = tf.sigmoid(layer1)
    # Second layer
    W2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0.0, 0.1))
    b2 = tf.Variable(tf.random_normal([n_hidden_2], 0.0, 0.1))
    layer2 = tf.matmul(layer1, W2) + b2
    layer2 = tf.sigmoid(layer2)
    # Output layer
    W3 = tf.Variable(tf.random_normal([n_hidden_2, n_classes], 0.0, 0.1))
    b3 = tf.Variable(tf.random_normal([n_classes], 0.0, 0.1))
    logits = tf.matmul(layer2, W3) + b3

    # Prediction and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    # Loss function
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

    # Optimizer and train op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        print("Training...")
        for i in range(train_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # run train op and compute batch loss
            _, batch_loss = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
            if (i+1) % 500 == 0:
                print("Train step {}:\n Batch loss = {}".format(i+1, batch_loss))
        print("Optimization finished.")

        # Test model
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy = {:.3f}".format(test_acc))

        # Look at some examples (and plot them)
        print("Some test examples")
        num_samples = 9  # has to be a square number for visualization purposes
        # take the first num_samples images and labels
        test_images = mnist.test.images[:num_samples, :]
        test_labels = mnist.test.labels[:num_samples, :]
        # compute model predictions
        predictions = sess.run(prediction, feed_dict={x: test_images, y: test_labels})
        for i in range(num_samples):
            # get actual class
            label = np.argmax(test_labels[i, :])
            pred = np.argmax(predictions[i, :])
            status = "CORRECT" if label == pred else "ERROR"
            print("{:7} - Label: {}, Prediction: {}".format(status, label, pred))

            # plot image
            img = test_images[i, :].reshape(28, 28)
            plt.subplot(np.sqrt(num_samples), np.sqrt(num_samples), i+1)
            plt.imshow(img)
            plt.gray()
            plt.axis("off")
            plt.title("Label: {}, Pred: {}".format(label, pred))

        # display figure
        plt.subplots_adjust(hspace=0.5)
        plt.show()
