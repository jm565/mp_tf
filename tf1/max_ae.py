import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

mnist = input_data.read_data_sets("data", one_hot=True)

tf.reset_default_graph()

num_inputs = 784  # 28x28 pixels
num_hid1 = 392
num_output = num_inputs
lr = 0.001
actf = tf.nn.relu

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# w1 = tf.Variable(tf.zeros([num_inputs, num_hid1]))
# w2 = tf.Variable(tf.zeros([num_hid1, num_output]))
w1 = tf.Variable(tf.random_normal([num_inputs, num_hid1], stddev=0.1))
w2 = tf.Variable(tf.random_normal([num_hid1, num_output], stddev=0.1))

# b1 = tf.Variable(tf.zeros(num_hid1))
# b2 = tf.Variable(tf.zeros(num_output))
b1 = tf.Variable(tf.random_normal([num_hid1], stddev=0.1))
b2 = tf.Variable(tf.random_normal([num_output], stddev=0.1))

hid_layer = actf(tf.matmul(X, w1) + b1)
output_layer = actf(tf.matmul(hid_layer, w2) + b2)

loss = tf.reduce_mean(tf.square(output_layer - X))

optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

train_steps = 5000
batch_size = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([train, loss], feed_dict={X: X_batch})

        if (i + 1) % 500 == 0:
            print("Train step {}:\n Batch loss = {}".format(i + 1, batch_loss))

    test_loss = sess.run(loss, {X: mnist.test.images})
    print("Test loss: {:.2f}".format(test_loss))
