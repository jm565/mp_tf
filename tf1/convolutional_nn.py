import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    # Import MNIST data
    mnist = read_data_sets("data", one_hot=True)

    # Hyperparameters
    train_steps = 1000
    learning_rate = 0.001
    batch_size = 100

    # Model parameters
    n_hidden_1 = 16
    n_hidden_2 = 32
    n_hidden_3 = 64
    n_hidden_4 = 512
    n_classes = 10  # 10 classes (numbers 0-9)
    n_input = 28*28  # flattened 28x28 images

    # Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    dropout_prob = tf.placeholder(tf.float32)

    # Model
    # Reshape input images for convolution
    input_images = tf.reshape(x, [-1, 28, 28, 1])
    # Conv & Pool 1
    W1 = tf.Variable(tf.random_normal([4, 4, 1, n_hidden_1], 0.0, 0.1))  # 4x4 filter
    b1 = tf.Variable(tf.random_normal([n_hidden_1], 0.0, 0.1))
    conv1 = tf.nn.conv2d(input_images, filter=W1, strides=[1, 1, 1, 1], padding="SAME")  # shape=(?, 28, 28, n_hidden_1)
    conv1 = tf.nn.relu(conv1 + b1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # shape=(?, 14, 14, n_hidden_1)
    # Conv & Pool 2
    W2 = tf.Variable(tf.random_normal([3, 3, n_hidden_1, n_hidden_2], 0.0, 0.1))  # 3x3 filter
    b2 = tf.Variable(tf.random_normal([n_hidden_2], 0.0, 0.1))
    conv2 = tf.nn.conv2d(pool1, filter=W2, strides=[1, 1, 1, 1], padding="SAME")  # shape=(?, 14, 14, n_hidden_2)
    conv2 = tf.nn.relu(conv2 + b2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # shape=(?, 7, 7, n_hidden_2)
    # Conv 3
    W3 = tf.Variable(tf.random_normal([2, 2, n_hidden_2, n_hidden_3], 0.0, 0.1))  # 2x2 filter
    b3 = tf.Variable(tf.random_normal([n_hidden_3], 0.0, 0.1))
    conv3 = tf.nn.conv2d(pool2, filter=W3, strides=[1, 1, 1, 1], padding="SAME")  # shape=(?, 7, 7, n_hidden_3)
    conv3 = tf.nn.relu(conv3 + b3)
    # Fully Connected with dropout
    conv3 = tf.reshape(conv3, [-1, 7*7*n_hidden_3])  # shape=(?, 7 * 7 * n_hidden_3)
    W4 = tf.Variable(tf.random_normal([7 * 7 * n_hidden_3, n_hidden_4], 0.0, 0.1))
    b4 = tf.Variable(tf.random_normal([n_hidden_4], 0.0, 0.1))
    fc = tf.nn.relu(tf.matmul(conv3, W4) + b4)
    fc = tf.nn.dropout(fc, 1.0 - dropout_prob)
    # Output
    W5 = tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0.0, 0.1))
    b5 = tf.Variable(tf.random_normal([n_classes], 0.0, 0.1))
    logits = tf.matmul(fc, W5) + b5

    # Prediction and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    # Loss function
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

    # Optimizer and train op
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        print("Training...")
        for i in range(train_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # run train op and compute batch loss
            _, batch_loss = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y, dropout_prob: 0.5})
            if (i+1) % 100 == 0:
                print("Train step {}:\n Batch loss = {}".format(i+1, batch_loss))
        print("Optimization finished.")

        # Test model
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout_prob: 0.0})
        print("Test accuracy = {:.3f}".format(test_acc))

        # Look at some examples (and plot them)
        print("Some test examples")
        num_samples = 9  # has to be a square number for visualization purposes
        # take the first num_samples images and labels
        test_images = mnist.test.images[:num_samples, :]
        test_labels = mnist.test.labels[:num_samples, :]
        # compute model predictions
        predictions = sess.run(prediction, feed_dict={x: test_images, y: test_labels, dropout_prob: 0.0})

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
