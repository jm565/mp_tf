###########################
# Sequentielle Datentypen #
###########################

# Strings
print("---------------------------")
string = "Monty Python"  # Base String
print(string)
s_concat = "Monty" + " Python"  # String Konkatenation
print(s_concat)
s_repeat = "Python" * 3  # String Wiederholung
print(s_repeat)
s_index = string[-1]  # String Indexing
print(s_index)
# string[4] = "Y"  # Strings unveränderlich (ERROR)
s_slice = string[5:9]  # String Slicing
print(s_slice)
s_len = len(string)  # String Länge
print(s_len)
s_split = string.split()  # String Split
print(s_split)

# Listen
print("---------------------------")
liste = ["Python", 4, False, 0.1]  # Base Liste
print(liste)
list_copy = liste[:]  # Kopie einer Liste
list_reference = liste  # Referenz auf eine Liste
liste[0] = 8  # Ursprungsliste verändern
print(liste)
print(list_copy)
print(list_reference)
list_of_lists = [[1, 2, 3], [4, 5, 6]]  # Liste von Listen
print(list_of_lists)
a = 4
b = "hello"
in_list = a in liste  # Objekt in Liste enthalten?
not_in_list = b not in liste  # Objekt nicht in Liste enthalten?
print(in_list)
print(not_in_list)

tupel = ("Python", 4, False, 0.1)
# tupel[0] = 8  # Tupel unveränderlich (ERROR)


# Listenmanipulation
print("---------------------------")
liste = [1, 2, 3]
print(liste)
liste.append([4, 5])  # Einzelnes Element anhängen
print(liste)
liste.extend([4, 5])  # Alle Elemente eines iterierbaren Objektes anhängen
print(liste)
liste.extend("Hello")
print(liste)
liste.remove(3)  # Element entfernen
print(liste)
liste_index = liste.index("H")  # Index eines Elementes suchen
print(liste_index)
liste.insert(2, "i")  # Element bei Index einfügen
print(liste)
