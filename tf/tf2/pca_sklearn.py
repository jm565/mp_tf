import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition


def plot_data(data, colors, title=None, show=False):
    if title:
        plt.title(title)
    try:
        plt.scatter(data[:, 0], data[:, 1], c=colors, s=25, alpha=0.5)
    except IndexError:
        plt.scatter(data[:, 0], np.zeros(data.shape[0]), c=colors, s=25, alpha=0.5)
    if show:
        plt.show()


if __name__ == "__main__":
    # Parameters
    number_classes = 3
    points_per_class = 300

    # Data
    # Linear data set (simple point blob)
    x = np.random.normal(-1, 0.5, [points_per_class])
    y = np.random.normal(2, 2, [points_per_class])
    data_x = x + y
    data_y = -x + y
    data = np.stack([data_x, data_y], axis=1).astype(np.float32)
    classes = np.ones([points_per_class])

    # # Non-linear data set (three concentric circles)
    # # -> basic PCA won't work well since it assumes the directions of variation are all straight lines
    # # -> Kernel PCA would do the trick (not implemented here)
    # data_list = []
    # class_list = []
    # for i in range(number_classes):
    #     theta = np.random.normal(0, 3.14, [points_per_class])
    #     r = np.random.normal(i*2, 0.1, [points_per_class])
    #     data_x = r * np.cos(theta)
    #     data_y = r * np.sin(theta)
    #     data_points = np.stack([data_x, data_y], axis=1) + [2.0, 2.0]
    #     data_points = data_points.astype(np.float32)
    #     classes = np.ones([points_per_class]) * i
    #     data_list.append(data_points)
    #     class_list.append(classes)
    # data = np.concatenate(data_list, axis=0)
    # classes = np.concatenate(class_list, axis=0)

    # MANUAL NUMPY
    reduce_dims = 1
    # Normalize data
    data_mean = data.mean(axis=0)
    data_normalized = data - data_mean
    # Covariance matrix
    c = np.cov(data_normalized, rowvar=False)
    # Eigenvalues & Eigenvectors
    eigen_vals, eigen_vecs = np.linalg.eigh(c)

    explained_variances = []
    for i in range(len(eigen_vals)):
        explained_variances.append(eigen_vals[i] / np.sum(eigen_vals))

    print(np.sum(explained_variances))
    print(explained_variances)

    sumEig = np.sum(eigen_vals)
    eigVar = np.sort(eigen_vals)[::-1]
    eigVar = eigVar / sumEig
    cumVar = np.cumsum(eigVar)
    print(cumVar)
    print(eigVar)

    # Sort Eigenvectors according to descending Eigenvalues
    indices = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[indices]
    eigen_vecs = eigen_vecs[:, indices]
    # Reduce dimensionality
    if reduce_dims > 0:
        eigen_vecs = eigen_vecs[:, :-reduce_dims]
    # New data matrix
    x = np.matmul(data_normalized, eigen_vecs)
    # Original space
    pca = np.matmul(x, np.transpose(eigen_vecs)) + data_mean

    # # SCIKITLEARN (Linear PCA)
    # reduce_dims = 1
    # p = decomposition.PCA(n_components=data.shape[-1] - reduce_dims)
    # p.fit(data)
    # x = p.transform(data)
    # # pca = x
    # pca = p.inverse_transform(x)

    # # SCIKITLEARN (Kernal PCA)
    # reduce_dims = 0
    # p = decomposition.KernelPCA(n_components=data.shape[-1] - reduce_dims, kernel="rbf", gamma=0.3)
    # p.fit(data)
    # x = p.transform(data)
    # pca = x

    # Plot results
    plt.figure(1, (8, 4))
    plt.subplot(1, 2, 1)
    plot_data(data, classes, title="Original dataset", show=False)
    plt.subplot(1, 2, 2)
    plot_data(pca, classes, title="Dataset after PCA", show=False)
    plt.show()
