import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# # Eager execution
# print(f"TF in eager execution: {tf.executing_eagerly()}")

# # Konstanten und Standardoperationen
# print("-------------------------------------------")
# c1 = tf.constant(3.0)  # Skalar
# c2 = tf.constant(5.0)  # Skalar
# c3 = c1 + c2  # oder tf.add(c1, c2)
# c4 = c1 * c2  # oder tf.multiply(c1, c2)
# print(c1, c2, c3, c4, sep="\n")
# x = tf.constant([[8, 8], [6, 6]])  # Konstante 2x2 matrix
# print(x)
# x = tf.cast(x, dtype=tf.float32)  # Datentyp verändern
# print(x)

# # Tensor-Initialisierungen
# print("-------------------------------------------")
# shape = (2, 3)
# value = 6.0
# print(tf.constant(list(range(1, 9)), shape=[4, 2]))  # Liste auf 4x2 Matrix
# print(tf.constant(value, shape=shape))  # Konstanter Wert auf 2x3 Matrix
# print(tf.ones(shape))  # Einsen
# print(tf.zeros(shape))  # Nullen
# print(tf.range(2, 10, 2))  # gleichverteilte Werte aus [start, limit), analog zu range()
# print(tf.linspace(0, 5, num=20))  # 'num' viele gleichverteilte Werte aus [start, stop]
# print(tf.random.uniform(shape, -value, value))  # Zufallswerte aus Gleichverteilung in [minval, maxval)
# print(tf.random.normal(shape, 0.0, 1.0))  # Zufallswerte aus Normalverteilung N(mean, stddev)

# # Variablen
# print("-------------------------------------------")
# x = tf.Variable([[1, 2, 3]], name="x")
# print(x)
# x.assign(x + tf.ones_like(x))  # Variable neuen Wert zuweisen
# print(x)
#
# y = tf.ones([3, 3], dtype=tf.int32)
# z = tf.matmul(x, y)  # Matrixmultiplikation

# # NumPy Kompatibilität
# print("-------------------------------------------")
# ndarray = np.ones([1, 3])
# tensor = tf.multiply(ndarray, 5)
# array = np.add(tensor, -3)
# print(tensor)
# print(array, type(array))
# print(tensor.numpy(), type(tensor.numpy()))

# Tensor-Transformationen
# print("-------------------------------------------")
# tensor = tf.constant(2.0, shape=[28, 28])
# t1 = tf.reshape(tensor, [14, 56])
# t2 = tf.reshape(tensor, [-1, 2])  # -1 entspricht "restlicher" shape
# t3 = tf.reshape(tensor, [28, 14, 2])
# t4 = tf.reshape(tensor, [28, 1, 28])
# for t in [tensor, t1, t2, t3, t4]:
#     print(t.shape)

# print("---------------------------")
# tensor = tf.constant(2.0, shape=[28, 28])
# t_expand = tf.expand_dims(tensor, axis=1)
# print(tensor.shape)
# print(t_expand.shape)

# print("---------------------------")
# tensor = tf.constant(2.0, shape=[28, 1, 1, 28, 1])
# t_squeeze = tf.squeeze(tensor, 1)
# t_squeeze_2 = tf.squeeze(t_squeeze)
# print(tensor.shape)
# print(t_squeeze.shape)
# print(t_squeeze_2.shape)

# print("---------------------------")
# tensor = tf.constant(list(range(12)), shape=[3, 4])
# t_transpose = tf.transpose(tensor)  # perm=None entspricht [n-1, n-2, ..., 0]
# print(tensor.shape)
# print(t_transpose.shape)
# tensor = tf.constant(list(range(24)), shape=[2, 3, 4])
# t_transpose = tf.transpose(tensor, [1, 2, 0])
# print(tensor.shape)
# print(t_transpose.shape)

# print("---------------------------")
# t1 = tf.constant(list(range(12)), shape=[3, 4])
# t2 = tf.constant(list(range(8)), shape=[2, 4])
# t_concat = tf.concat([t1, t2], axis=0)
# t_unstack = tf.unstack(t_concat, axis=1)
# t_stack = tf.stack(t_unstack, axis=0)
# # print(t1)
# # print(t2)
# print(t_concat)
# for t in t_unstack:
#     print(t)
# print(t_stack)
