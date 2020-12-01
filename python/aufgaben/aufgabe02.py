"""
In einem Programm sollen die x- und y-Koordinate eines Punktes vom Benutzer eingegeben werden.
Anschließend soll das Programm entscheiden, in welchem Quadranten der Punkt liegt.
Hinweis: Kombiniere input() und float(), um eine Gleitkommazahl einzulesen.
"""

print("Bestimmung des Quadranten für (x,y)-Koordinate")
x = float(input("x = "))
y = float(input("y = "))
if x >= 0:
    if y >= 0:
        quadrant = 1
    else:
        quadrant = 4
else:
    if y >= 0:
        quadrant = 2
    else:
        quadrant = 3
print(f"Der Punkt ({x},{y}) liegt im {quadrant}. Quadranten.")


