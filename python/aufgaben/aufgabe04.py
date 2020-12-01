"""
Gegeben sei die Liste
liste = [-1, 1, 2, 4, -5, 9, 12]
Schreiben Sie ein Python-Programm, das sowohl die Summe und das Produkt der in der Liste enthaltenen
Zahlen berechnet, als auch das Maximum und Minimum über alle Elemente ermittelt.
"""

liste = [-1, 1, 2, 4, -5, 9, 12]
summe = 0
produkt = 1
minimum = liste[0]
maximum = liste[0]
for element in liste:
    summe += element
    produkt *= element
    if element < minimum:
        minimum = element
    if element > maximum:
        maximum = element
print(f"Liste: {liste}")
print(f"Summe der Listenelemente: {summe}")
print(f"Produkt der Listenelemente: {produkt}")
print(f"Maximales Element der Liste: {maximum}")
print(f"Minimales Element der Liste: {minimum}")
