import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_data_and_centroids(data, centroids, colors):
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=25, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c="black", linewidth=2)
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_points = 300
    num_clusters = 3
    num_interations = 100

    # data generation
    points_a = np.random.normal(0.0, 0.5, [num_points // 3, 2])  # [num_points // 3, 2]
    points_b = np.random.normal(2.0, 0.5, [num_points // 3, 2])  # [num_points // 3, 2]
    points_c = np.random.normal(4.0, 0.5, [num_points // 3, 2]) + [0, -4]  # [num_points // 3, 2]
    points = np.concatenate([points_a, points_b, points_c], axis=0)  # [num_points, 2]
    points = points.astype(np.float32)

    # sample centroids from points
    centroids = tf.Variable(tf.random.shuffle(points)[:num_clusters])  # [num_clusters, 2]

    # plot initialization
    assignments = np.zeros([num_points])
    plot_data_and_centroids(points, centroids, assignments)

    # training loop
    for step in range(num_interations):
        # prepare for broadcasting
        points_expanded = tf.expand_dims(points, 0)  # [1, num_points, 2]
        centroids_expanded = tf.expand_dims(centroids, 1)  # [num_clusters, 1, 2]
        # compute distances (each points vs. each centroid)
        distances = tf.square(points_expanded - centroids_expanded)  # [num_clusters, num_points, 2]
        distances = tf.reduce_sum(distances, 2)  # [num_clusters, num_points]
        # determine cluster assignments
        assignments = tf.argmin(distances, 0)  # [num_points]

        # compare clusters with cluster assignments and calculate means across points assigned to each cluster
        means = []
        for c in range(num_clusters):
            cluster_points = tf.gather(points, tf.reshape(tf.where(tf.equal(assignments, c)), [-1]))
            cluster_mean = tf.reduce_mean(cluster_points, axis=0)
            means.append(cluster_mean)

        # update centroids
        centroids.assign(means)

    # plot points with final centroids
    print(f"Centroids:\n{centroids.numpy()}")
    plot_data_and_centroids(points, centroids, assignments)
