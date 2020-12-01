"""
Gegeben sei eine Liste von Punkten mit x-y-Koordinaten:
punkte = [(0, 1), (1, 2), (1, 3), (2, 3), (2, 5), (4, 3), (5, 3), (5, 4), (5, 5)]
Es soll ein Python-Programm erstellt werden, das entscheidet, ob diese Punkte innerhalb eines Kreises liegen.
Schreiben Sie eine allgemeingültige Funktion, die beliebige Kreise und Punktlisten
verarbeiten kann. Beispielhaft soll ein Kreis mit den Mittelpunktskoordinaten (3,3) und einem Radius von 2 betrachtet
werden. Als Ausgabe soll die Funktion eine Liste liefern, die neben den Punktkoordinaten
den Eintrag 'True' oder 'False' enthält, je nachdem ob der Punkt innerhalb des Kreises liegt oder nicht.
"""


def punkte_im_kreis(punkt_liste, mp_x, mp_y, r):
    """
    Bestimmt ob die Punkte aus `punkt_liste` im Kreis mit Mittelpunkt `(mp_x, mp_y)` und Radius `r` liegen.

    :param punkt_liste: Liste von Punkten.
    :param mp_x: x-Koordinate vom Mittelpunkt des Kreises.
    :param mp_y: y-Koordinate vom Mittelpunkt des Kreises.
    :param r: Durchmesser des Kreises.
    :return: Liste mit den Punkten aus `punkt_liste` mit zusätzlichem Boolean, ob der Punkt im Kreis liegt oder nicht.
    """

    res = []
    for punkt in punkt_liste:
        x = punkt[0]
        y = punkt[1]
        in_circle = True if (x - mp_x)**2 + (y - mp_y)**2 <= r**2 else False
        res.append((x, y, in_circle))
    return res


punkte = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]
x_k = 3
y_k = 3
r_k = 2

p = punkte_im_kreis(punkte, x_k, y_k, r_k)
print(f"Kreis mit Mittelpunkt ({x_k},{y_k}) und Radius {r_k}.")
print(f"Erweiterte Punkteliste: {p}")
