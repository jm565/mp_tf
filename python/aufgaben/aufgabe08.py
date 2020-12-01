"""
Schreiben Sie ein Python Programm 'fibonacci.py' in dem Sie zwei Funktionen 'fib' und 'fib_list'
definieren. Die Funktion 'fib' soll die n-te Fibonacci Zahl berechnen und zurückgeben. Die Funktion
'fib_list' soll die ersten n Fibonacci Zahlen in einer Liste zurückgeben. Anschließend sollen die beiden
Funktionen aus einer anderen Python Datei angesprochen und auf n=10 angewendet werden.
Hinweis: Die ersten beiden Zahlen der Fibonacci-Folge sind 0, 1
"""

import fibonacci as fib

n = 10
print(f"Die {n}-te Fibonacci Zahl: {fib.fib(n)}")
print(f"Die ersten {n} Fibonacci Zahlen: {fib.fib_list(n)}")
