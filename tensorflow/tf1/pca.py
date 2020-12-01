import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    # model parameters
    number_classes = 3
    points_per_class = 300
    reduce_dims = 1

    # data generation

    # # Linear data set (simple point blob)
    # x = tf.random_normal([points_per_class], -1, 1)
    # y = tf.random_normal([points_per_class], 2, 2)
    # data_x = x + y
    # data_y = -x + y
    # data = tf.stack([data_x, data_y], axis=1)
    # classes = tf.ones([points_per_class])

    # Non-linear data set (three concentric circles)
    # -> basic PCA won't work well since it assumes the directions of variation are all straight lines
    # -> Kernel PCA would do the trick (not implemented here)
    data_list = []
    class_list = []
    for i in range(number_classes):
        theta = tf.random_normal([points_per_class], 0, 3.14)
        r = tf.random_normal([points_per_class], i*2, 0.1)
        data_x = r * tf.cos(theta)
        data_y = r * tf.sin(theta)
        data_points = tf.stack([data_x, data_y], axis=1) + [2.0, 2.0]
        classes = tf.ones([points_per_class]) * i
        data_list.append(data_points)
        class_list.append(classes)
    data = tf.concat(data_list, axis=0)
    classes = tf.concat(class_list, axis=0)
    print(classes.get_shape())

    # normalize data
    data_mean = tf.reduce_mean(data, axis=0)
    data_normalized = data - data_mean

    # Covariance matrix
    def tf_cov(matrix, bias=False):
        mean = tf.reduce_mean(matrix, axis=0, keepdims=True)
        mm = tf.matmul(tf.transpose(mean), mean)
        xx = tf.matmul(tf.transpose(matrix), matrix)
        if bias:
            xx = xx / tf.to_float(tf.shape(matrix)[0])
        cov_matrix = xx - mm
        return cov_matrix

    def tf_cov2(matrix):
        m = tf.matmul(tf.transpose(matrix), matrix)
        return m / tf.to_float(tf.shape(matrix)[0])

    c = tf_cov(data_normalized, bias=True)
    # c = tf_cov2(data_normalized)

    # Eigenvectors & Eigenvalues
    evals, evecs = tf.self_adjoint_eig(c)
    # Sort Eigenvectors according to descending Eigenvalues
    indices = tf.argsort(evals, direction='DESCENDING')
    evecs = tf.gather(evecs, indices, axis=1)

    # Reduce dimensionality
    if reduce_dims > 0:
        # evecs = tf.slice(evecs, [0, 0], [evecs.get_shape()[0], evecs.get_shape()[1] - reduce_dims])
        evecs = evecs[:, :-reduce_dims]

    # New data matrix
    x = tf.matmul(data_normalized, evecs)
    # Original data
    y = tf.matmul(x, tf.transpose(evecs)) + data_mean
    pca = y

    with tf.Session() as sess:
        # Compute PCA
        data_points, data_classes, pca_data_points = sess.run([data, classes, pca])

        # Plot results
        plt.figure(1, (8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original dataset")
        plt.scatter(data_points[:, 0], data_points[:, 1], c=data_classes, s=25, alpha=0.5)
        plt.subplot(1, 2, 2)
        plt.title("Dataset after PCA")
        plt.scatter(pca_data_points[:, 0], pca_data_points[:, 1], c=data_classes, s=25, alpha=0.5)
        plt.show()
