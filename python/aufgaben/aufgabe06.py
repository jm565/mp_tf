"""
Gegeben seien folgende Listen von x- bzw. y-Koordinaten
x_coords = [-2, -1, 0, 1, 2]
y_coords = [-5, -3, -1, 1, 3, 5]
Schreiben Sie ein Python-Programm, das eine Liste aller möglichen Punkte aus den gegebenen Koordinaten
erzeugt, wobei nur Punkte im 1. Quadranten mit aufgenommen werden sollen. Nutzen Sie dazu die Listen-Abstraktion.
Speichern Sie anschließend die Liste in eine Text-Datei.
"""

x_coords = [-2, -1, 0, 1, 2]
y_coords = [-5, -3, -1, 1, 3, 5]

points_in_q1 = [(x, y) for x in x_coords for y in y_coords if x >= 0 and y >= 0]

print(f"Punkte im 1. Quadranten: {points_in_q1}")

print("Schreibe nach 'aufgabe06.txt' ...")
with open("aufgabe06.txt", "w") as f:
    for p in points_in_q1:
        f.write(str(p))
        f.write("\n")
print("Fertig.")
