import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition

if __name__ == "__main__":
    # model parameters
    number_classes = 3
    points_per_class = 300
    reduce_dims = 1

    # data generation

    # Linear data set (simple point blob)
    x = np.random.normal(-1, 1, [points_per_class])
    y = np.random.normal(2, 3, [points_per_class])
    data_x = x + y
    data_y = -x + y
    data = np.stack([data_x, data_y], axis=1)
    classes = np.ones([points_per_class])

    # Non-linear data set (three concentric circles)
    # -> basic PCA won't work well since it assumes the directions of variation are all straight lines
    # -> Kernel PCA would do the trick (not implemented here)
    # data_list = []
    # class_list = []
    # for i in range(number_classes):
    #     theta = tf.random_normal([points_per_class], 0, 3.14)
    #     r = tf.random_normal([points_per_class], i*2, 0.1)
    #     data_x = r * tf.cos(theta)
    #     data_y = r * tf.sin(theta)
    #     data_points = tf.stack([data_x, data_y], axis=1) + [2.0, 2.0]
    #     classes = tf.ones([points_per_class]) * i
    #     data_list.append(data_points)
    #     class_list.append(classes)
    # data = tf.concat(data_list, axis=0)
    # classes = tf.concat(class_list, axis=0)

    # pca = decomposition.PCA(n_components=2)
    # pca.fit(data)
    # data_reduced = pca.transform(data)
    # pca_data = pca.inverse_transform(data_reduced)

    # Normalize data
    data_mean = data.mean(axis=0)
    data_normalized = data - data_mean

    # Covariance matrix
    c = np.cov(data_normalized, rowvar=False)

    # Eigenvectors & Eigenvalues

    evals, evecs = np.linalg.eigh(c)
    # Sort Eigenvectors according to descending Eigenvalues
    indices = np.argsort(evals)[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    # Reduce dimensionality
    if reduce_dims > 0:
        evecs = evecs[:, :-reduce_dims]

    # New data matrix
    x = np.matmul(data_normalized, evecs)
    # Original data
    y = np.matmul(x, np.transpose(evecs)) + data_mean
    pca = y

    # # SCIKITLEARN
    # p = decomposition.PCA(n_components=1)
    # p.fit(data)
    # p_data = p.transform(data)
    # p_original = np.matmul(x, np.transpose(evecs)) + p.mean_

    # Plot results
    plt.figure(1, (8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original dataset")
    plt.scatter(data[:, 0], data[:, 1], c=classes, s=25, alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.title("Dataset after PCA")
    plt.scatter(pca[:, 0], pca[:, 1], c=classes, s=25, alpha=0.5)
    plt.show()
