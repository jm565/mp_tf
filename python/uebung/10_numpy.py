import numpy as np

print("---------------------------")
values = [1, 2, 3, 4, 5, 6, 7, 8]
x = np.array(values)
print(x)
print(type(x))

print("---------------------------")
y = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(y.ndim)
print(y.shape)

print("---------------------------")
print(x[0])
print(x[2:-1])
print(x[:4])
print(x[::2])

print("---------------------------")
z = np.array([
    [11, 12, 13, 14, 15],
    [21, 22, 23, 24, 25],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45],
    [51, 52, 53, 54, 55]])
print(z[2, 2])
print(z[3:, :-1])
print(z[:, 3])
print(z[::3, ::-1])

print("---------------------------")
shape = (2, 2)
value = 5
print(np.array(value))  # direkt
print(np.ones(shape))  # Einsen
print(np.zeros(shape))  # Nullen
print(np.full(shape, value))  # Array voll mit 'value'
print(np.arange(2, 10, 2))  # gleichverteilte Werte aus [start, stop), analog zu range()
print(np.linspace(0, 5, num=20, endpoint=False))  # 'num' viele gleichverteilte Werte aus [start, stop])
print(np.random.rand(*shape))  # Zufaellige floats aus [0,1]
print(np.random.randint(-5, 5, shape))  # Zufaellige integer aus [start, stop)

print("---------------------------")
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a + 10)
print(a ** 2)
b = np.full((2, 3), 3)
print(a - b)
print(a * b)

print("---------------------------")
c = np.transpose(a)
print(np.matmul(a, c))
