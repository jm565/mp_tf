import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    # Import MNIST data
    mnist = read_data_sets("data", one_hot=True)

    # Limit the data
    xtrain, ytrain = mnist.train.next_batch(30000)
    xtest, ytest = mnist.test.next_batch(500)

    # Placeholders for graph input
    xtr = tf.placeholder(tf.float32, [None, 28 * 28])  # training input
    ytr = tf.placeholder(tf.float32, [None, 10])  # training label
    xte = tf.placeholder(tf.float32, [28 * 28])  # testing input (1 image at a time)

    K = 5  # how many neighbors

    # model
    # distances = -tf.reduce_sum(tf.abs(xtr - xte), axis=1)  # 1-Norm
    # # the negative above enables top_k to get the lowest distances
    # values, indices = tf.nn.top_k(distances, k=K, sorted=False)
    distances = tf.reduce_sum(tf.abs(xtr - xte), axis=1)  # 1-Norm
    indices = tf.argsort(distances, direction='ASCENDING')[:K]

    # nearest neighbor classes
    nearest_neighbors = [tf.argmax(ytr[indices[i]], 0) for i in range(K)]

    # this will return the unique neighbors with their respective counts
    labels, idx, count = tf.unique_with_counts(nearest_neighbors)

    # predictions
    pred = labels[tf.argmax(count, 0)]

    with tf.Session() as sess:
        correct_predicted = 0.0

        for i in range(xtest.shape[0]):
            # return the predicted value
            predicted_value = sess.run(pred, feed_dict={xtr: xtrain, ytr: ytrain, xte: xtest[i, :]})

            print("Test {}: Prediction = {}, True Class = {}".format(i, predicted_value, np.argmax(ytest[i])))

            if predicted_value == np.argmax(ytest[i]):
                correct_predicted += 1.0
        print("Nearest Neighbor (k={}) Accuracy is: {:.3f}".format(K, correct_predicted / len(xtest)))
