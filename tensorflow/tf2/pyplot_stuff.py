import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

plt.figure(1)
plt.plot([-1, -4.5, 16, 23, 15, 59])

xs = [0, 2, 4, 6, 8]
ys = [-2, -1, 0, -1, -2]
plt.figure(2)
plt.plot(xs, ys)

plt.figure(3)
plt.plot(xs, ys, color='green', marker='o', linestyle='dashed', linewidth=2.5, markersize=12, label='line xy')
plt.legend()

plt.figure(4)
plt.plot([-1, -4.5, 16, 23, 15, 59], '^r')

plt.figure(5)
plt.plot([-1, -4.5, 16, 23, 15, 59], 'x:k')

plt.figure(6)
colors = ['green', 'red', 'yellow', 'blue']
plt.suptitle("Multiple subfigures")
for i, c in enumerate(colors):
    plt.subplot(2, 2, i+1)
    plt.plot(xs, ys, color=c)
    plt.title("Subplot {}/4".format(i+1))
plt.subplots_adjust(hspace=0.4)

plt.figure(7)
def samplemat(dims):
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa
plt.imshow(samplemat((15, 15)))
plt.gray()

plt.figure(8)
mnist = input_data.read_data_sets("data", one_hot=True)
# mnist = input_data.read_data_sets("data/fashion", one_hot=True,
#                                   source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
imgs = mnist.train.images[15:21, :].reshape(-1, 28, 28)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i], cmap="gray")
    plt.axis("off")
plt.show()
