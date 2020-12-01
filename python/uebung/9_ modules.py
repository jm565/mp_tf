###################
# Modularisierung #
###################

# Importieren eines Moduls
print("---------------------------")
import math
print(math.sin(math.pi), math.cos(math.pi))

# Selektives Importieren
print("---------------------------")
from math import sin, cos, pi
print(sin(pi), cos(pi))

# Umbenennen des Namensraumes
print("---------------------------")
import math as mathe
from math import cos as cosinus
print(mathe.sin(mathe.pi), cosinus(mathe.pi))

# Lade das Modul modularisierung.py und führe die Funktion func aus
print("---------------------------")
import modularisierung
print("Zugriff auf das Modul modularisierung.py:")
modularisierung.func()

