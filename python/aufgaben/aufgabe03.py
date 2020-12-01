"""
Gegeben sei die Liste
liste = [12, 22, "Monty", 34.5, "Python", 45.1, 212, 0.2].
Schreiben Sie ein Python-Programm, das die Elemente dieser Liste in drei verschiedene Listen für
Ganzzahlen (int), Gleitkommazahlen (float) und Zeichenketten (str) einfügt.
Die Originalliste soll dabei erhalten bleiben. Geben Sie anschließend die Listen aus.
Hinweis: Nutzen sie type() für die Typabfrage.
"""

liste = [12, 22, "Monty", 34.5, "Python", 45.1, 212, 0.2]
liste_int = []
liste_float = []
liste_string = []
for element in liste:
    if type(element) == int:
        liste_int.append(element)
    elif type(element) == float:
        liste_float.append(element)
    elif type(element) == str:
        liste_string.append(element)
    else:
        print(f"Ich kann mit dem Datentyp {type(element)} nichts anfangen.")
print(f"Originalliste: {liste}")
print(f"Ganzzahlliste: {liste_int}")
print(f"Gleitkommaliste: {liste_float}")
print(f"Stringliste: {liste_string}")
