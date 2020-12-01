import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    points_n = 300
    clusters_n = 3
    iteration_n = 100

    # data generation
    points_a = tf.random.normal([points_n // 3, 2], 0.0, 0.5)  # [100, 2]
    points_b = tf.random.normal([points_n // 3, 2], 2.0, 0.5)  # [100, 2]
    points_c = tf.random.normal([points_n // 3, 2], 4.0, 0.5) + [0, -4]  # [100, 2]
    points = tf.concat([points_a, points_b, points_c], axis=0)  # [300, 2]

    # sample centroids from points
    # centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, 2]))
    centroids = tf.Variable(tf.random.shuffle(points)[:clusters_n])  # [C, 2]

    # prepare for broadcasting
    points_expanded = tf.expand_dims(points, 0)  # [1, 300, 2]
    centroids_expanded = tf.expand_dims(centroids, 1)  # [C, 1, 2]
    # compute distances and determine cluster assignments
    distances = tf.reduce_sum(tf.square(points_expanded - centroids_expanded), 2)  # [C, 300, 2] -> [C, 300]
    assignments = tf.argmin(distances, 0)  # [300]

    # compare clusters with cluster assignments and calculate means across points assigned to each cluster
    means = []
    for c in range(clusters_n):
        cluster_points = tf.gather(points, tf.reshape(tf.where(tf.equal(assignments, c)), [-1]))
        cluster_mean = tf.reduce_mean(cluster_points, axis=0)
        means.append(cluster_mean)

    # update centroids
    update_centroids = tf.assign(centroids, means)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # training loop
        for step in range(iteration_n):
            _, centroid_values, points_values, assignment_values = sess.run(
                [update_centroids, centroids, points, assignments])

        print("Centroids\n{}".format(centroid_values))

        # plot points with final centroids
        plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=25, alpha=0.5)
        plt.plot(centroid_values[:, 0], centroid_values[:, 1],
                 marker='x', markersize=15, markeredgecolor="black", linestyle="None")
        plt.show()
