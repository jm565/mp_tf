def fib(n):
    a, b = 0, 1
    for i in range(n-2):
        a, b = b, a + b
    return b


def fib_list(n):
    res = [0, 1]
    for i in range(n-2):
        res.append(res[-1] + res[-2])
    return res
