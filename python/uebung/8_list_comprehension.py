######################
# Listen-Abstraktion #
######################

# Einfache Form
print("---------------------------")
print([x ** 2 for x in range(10)])

liste_int = [1, 5, 10, 16]
print([x + 3 for x in liste_int])

string = "Monty Python was great"
print([len(s) for s in string.split()])

# Erweiterte Form mit if-Anweisung
print("---------------------------")
liste1 = [5, 6, 12, 17]
liste2 = [2, 5, 6]
print([x for x in liste1 if x in liste2])

# Pythagoreische Tripel
# Hinweis: y startet ab x und z startet ab y um Permutationen der Tripel zu eliminieren und gleichzeitig Zeit zu sparen
print([(x, y, z) for x in range(1, 20) for y in range(x, 20) for z in range(y, 20) if x**2 + y**2 == z**2])

# Komplexeres Beispiel (Primzahlen nach Sieb des Eratosthenes)
# Hinweis: Da mindestens ein Primfaktor einer zusammengesetzten Zahl immer kleiner gleich der Wurzel der Zahl sein muss,
#          reicht es aus, nur die Vielfachen von Zahlen zu streichen, die kleiner oder gleich der Wurzel von n sind.
# Hinweis: Ebenso genügt es beim Streichen der Vielfachen, mit dem Quadrat der Primzahl zu beginnen,
#          da alle kleineren Vielfachen bereits markiert sind.
print("---------------------------")
from math import sqrt
n = 100
no_primes = [j for i in range(2, int(sqrt(n)) + 1) for j in range(i ** 2, n, i)]
primes = [x for x in range(2, n) if x not in no_primes]
print(no_primes)
print(primes)

# Mengen-Abstraktion
no_primes = {j for i in range(2, int(sqrt(n)) + 1) for j in range(i ** 2, n, i)}
primes = {x for x in range(2, n) if x not in no_primes}
print(no_primes)
print(primes)
