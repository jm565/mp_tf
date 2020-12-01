################
# Dictionaries #
################

print("---------------------------")
dic = {"a": 1, "b": 2, "c": 3}  # Base dictionary (Schlüssel-Werte-Paare)
print(dic)
dic_len = len(dic)  # Länge des dictionary
print(dic_len)
print(dic.keys())  # Keys (Schlüssel)
print(dic.values())  # Values (zugehörigen Werte)
print(dic["b"])  # Zugriff über Schlüssel
del dic["b"]  # Löschen eines Paares via Schlüssel
print(dic)
in_dict = "a" in dic  # Element in Schlüsseln enthalten?
not_in_dict = 3 not in dic  # Element nicht in Schlüsseln enthalten?
print(in_dict)
print(not_in_dict)
# dic = {"b": 10, "a": 20, [1]: 30}  # Nur unveränderliche Schlüssel (ERROR)
dict_copy = dic.copy()  # Kopie eines dictionary
dic["a"] = 5  # Zugehörigen Wert eines Schlüssels überschreiben
print(dic)
print(dict_copy)
dic.update({"a": 2, "b": 4, "c": 6})  # Dictionary updaten
print("Dict - Update: ", dic)
