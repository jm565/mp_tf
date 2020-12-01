import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition


def plot_data(data, colors, show=False):
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=25, alpha=0.5)
    if show:
        plt.show()


if __name__ == "__main__":
    # Parameters
    number_classes = 3
    points_per_class = 300
    reduce_dims = 1

    # Data
    # Linear data set (simple point blob)
    x = np.random.normal(-1, 1, [points_per_class])
    y = np.random.normal(2, 3, [points_per_class])
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

    # # MANUAL NUMPY
    # # Normalize data
    # data_mean = data.mean(axis=0)
    # data_normalized = data - data_mean
    # # Covariance matrix
    # c = np.cov(data_normalized, rowvar=False)
    # # Eigenvalues & Eigenvectors
    # eigen_vals, eigen_vecs = np.linalg.eigh(c)
    # # Sort Eigenvectors according to descending Eigenvalues
    # indices = np.argsort(eigen_vals)[::-1]
    # eigen_vals = eigen_vals[indices]
    # eigen_vecs = eigen_vecs[:, indices]
    # # Reduce dimensionality
    # if reduce_dims > 0:
    #     eigen_vecs = eigen_vecs[:, :-reduce_dims]
    # # New data matrix
    # x = np.matmul(data_normalized, eigen_vecs)
    # print(x.shape)
    # # Original data
    # pca = np.matmul(x, np.transpose(eigen_vecs)) + data_mean

    # # SCIKITLEARN
    # p = decomposition.PCA(n_components=data.shape[-1] - reduce_dims)
    # p.fit(data)
    # x = p.transform(data)
    # pca = p.inverse_transform(x)

    # SCIKITLEARN (Kernal PCA)
    p = decomposition.KernelPCA(n_components=data.shape[-1] - reduce_dims, kernel="linear", fit_inverse_transform=True)
    p.fit(data)
    x = p.transform(data)
    pca = p.inverse_transform(x)

    # Plot results
    plt.figure(1, (8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original dataset")
    plot_data(data, classes, show=False)
    plt.subplot(1, 2, 2)
    plt.title("Dataset after PCA")
    plot_data(pca, classes, show=False)
    plt.show()
