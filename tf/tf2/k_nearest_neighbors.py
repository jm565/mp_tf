import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
import numpy as np


if __name__ == "__main__":
    # Import MNIST data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train / 255.0  # norm to [0,1]
    x_test = x_test / 255.0  # norm to [0,1]

    # Limit the data
    limit_data = True
    if limit_data:
        limit_train = 20000
        limit_test = 300
        x_train, y_train = x_train[:limit_train], y_train[:limit_train]
        x_test, y_test = x_test[:limit_test], y_test[:limit_test]

    # Flatten and cast data
    x_train = x_train.reshape(-1, 28*28).astype(np.float32)
    y_train = y_train.astype(np.int32)
    x_test = x_test.reshape(-1, 28*28).astype(np.float32)
    y_test = y_test.astype(np.int32)

    # How many neighbors to consider
    k = 5

    # Test loop
    correct_predicted = 0.0
    for i in range(x_test.shape[0]):
        # Model
        distances = tf.reduce_sum(tf.abs(x_train - x_test[i, :]), axis=1)  # 1-Norm
        indices = tf.argsort(distances, direction='ASCENDING')[:k]  # top-K indices (closest distances)

        # nearest neighbor classes
        nearest_neighbors = [y_train[indices[i]] for i in range(k)]

        # this will return the unique neighbors with their respective counts
        labels, idx, count = tf.unique_with_counts(nearest_neighbors)

        # predictions
        pred = labels[tf.argmax(count, 0)]
        pred = pred.numpy().astype(np.int32)
        print(f"Test image {i}: Prediction = {pred}, True Class = {y_test[i]}")

        if pred == y_test[i]:
            correct_predicted += 1.0

    accuracy = correct_predicted / x_test.shape[0]
    print(f"Nearest Neighbor (k={k}) accuracy is: {accuracy * 100:.2f}%")
