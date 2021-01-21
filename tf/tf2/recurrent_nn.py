import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell
from tensorflow.contrib.rnn import static_rnn, static_bidirectional_rnn
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def output_on_last_timestep(rnn_outputs, number_classes):
    rnn_cell_size = rnn_outputs[-1].get_shape().as_list()[1]  # = n_hidden_1 (2 * n_hidden_ when bidirectional)
    W1 = tf.Variable(tf.random_normal([rnn_cell_size, number_classes], 0.0, 0.1))
    b1 = tf.Variable(tf.random_normal([number_classes], 0.0, 0.1))
    out = tf.matmul(rnn_outputs[-1], W1) + b1
    return out


def output_on_all_timesteps(rnn_outputs, fc_size, number_classes):
    fc_inputs = tf.stack(rnn_outputs, axis=1)
    number_timesteps = fc_inputs.get_shape().as_list()[1]  # = n_timesteps
    rnn_cell_size = fc_inputs.get_shape().as_list()[2]  # = n_hidden_1 (2 * n_hidden when bidirectional)
    fc_inputs = tf.reshape(fc_inputs, [-1, number_timesteps * rnn_cell_size])
    W1 = tf.Variable(tf.random_normal([number_timesteps * rnn_cell_size, fc_size], 0.0, 0.1))
    b1 = tf.Variable(tf.random_normal([fc_size], 0.0, 0.1))
    fc = tf.nn.relu(tf.matmul(fc_inputs, W1) + b1)
    # Output layer
    W2 = tf.Variable(tf.random_normal([fc_size, number_classes], 0.0, 0.1))
    b2 = tf.Variable(tf.random_normal([number_classes], 0.0, 0.1))
    out = tf.matmul(fc, W2) + b2
    return out


if __name__ == "__main__":
    # Import MNIST data
    mnist = read_data_sets("data", one_hot=True)

    # Hyperparameters
    train_steps = 5000
    learning_rate = 0.001
    batch_size = 100

    # Model parameters
    n_hidden_1 = 64
    n_hidden_2 = 512
    n_classes = 10  # 10 classes (numbers 0-9)
    n_input, n_timesteps = 28, 28  # interpret 28x28 images as 28 sequences of length 28
    use_bidirectional_rnn = True
    output_on_last_timestep_only = False

    # Graph input
    x = tf.placeholder(tf.float32, [None, n_input * n_timesteps])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Model
    # Reshape input images for recurrent neural network
    input_images = tf.reshape(x, [-1, n_timesteps, n_input])
    input_images = tf.unstack(input_images, axis=1)
    # Recurrent layer (uni- or bidirectional)
    if not use_bidirectional_rnn:
        rnn_cell = LSTMCell(n_hidden_1)
        outputs, _ = static_rnn(rnn_cell, input_images, dtype=tf.float32)
    else:
        rnn_cell_fw = LSTMCell(n_hidden_1)
        rnn_cell_bw = LSTMCell(n_hidden_1)
        outputs, _, _ = static_bidirectional_rnn(rnn_cell_fw, rnn_cell_bw, input_images, dtype=tf.float32)
    # Output layer (on last or all timesteps)
    if output_on_last_timestep_only:
        logits = output_on_last_timestep(outputs, n_classes)
    else:
        logits = output_on_all_timesteps(outputs, n_hidden_2, n_classes)

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
