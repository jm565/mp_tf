import matplotlib.pyplot as plt
import numpy as np

# Plot einer Liste von y-Werte (x-Werte implizit über Indizes)
plt.figure("Plotten einer Liste von y-Werte")
plt.plot([-1, -4.5, 16, 23, 15, 59])

# Plot von (x,y)-Punkten mittels zwei Listen
xs = [0, 2, 4, 6, 8]
ys = [-2, -1, 0, -1, -2]
plt.figure("Plotten von (x,y)-Punkten mittels zwei Listen")
plt.plot(xs, ys)

# Plot Formartierung
plt.figure("Plot-Formatierung über Format-Parameter '^r'")
plt.plot([-1, -4.5, 16, 23, 15, 59], '^r')

plt.figure("Plot-Formatierung über Format-Parameter 'x:k'")
plt.plot([-1, -4.5, 16, 23, 15, 59], 'x:k')

plt.figure("Plot-Formatierung über Argumente, mit Legende")
plt.plot(xs, ys, color='green', marker='o', linestyle='dashed', linewidth=2.5, markersize=12, label='line xy')
plt.legend()

# Scatter plot
num_points = 300
points_a = np.random.normal(0.0, 0.5, size=(num_points // 3, 2))                     # [100, 2]
points_b = np.random.normal(2.0, 0.5, size=(num_points // 3, 2))                     # [100, 2]
points_c = np.random.normal(4.0, 0.5, size=(num_points // 3, 2)) - np.array([0, 4])  # [100, 2]
points = np.concatenate([points_a, points_b, points_c], axis=0)                      # [300, 2]
centroid_a = np.mean(points_a, axis=0)                                               # [2]
centroid_b = np.mean(points_b, axis=0)                                               # [2]
centroid_c = np.mean(points_c, axis=0)                                               # [2]
centroids = np.stack([centroid_a, centroid_b, centroid_c], axis=0)                   # [3, 2]
colors = np.repeat([0, 1, 2], num_points // 3)                                       # [300]
plt.figure("Scatter Plot")
plt.scatter(points[:, 0], points[:, 1], c=colors, s=25, alpha=0.5)
plt.plot(centroids[:, 0], centroids[:, 1], marker='x', markersize=18, markeredgecolor="black", linestyle="None")

# Mehrere subplots
plt.figure("Mehrere Plots")
colors = ['green', 'red', 'yellow', 'blue']
plt.suptitle("Multiple subplots")
for i, c in enumerate(colors):
    plt.subplot(2, 2, i+1)
    plt.plot(xs, ys, color=c)
    plt.title(f"Subplot {i+1}/4")
plt.subplots_adjust(hspace=0.4)


# Bilder anzeigen
def samplemat(dims):
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa


plt.figure("Bild anzeigen")
plt.imshow(samplemat((15, 15)))
plt.gray()

# Display the figures
plt.show()
