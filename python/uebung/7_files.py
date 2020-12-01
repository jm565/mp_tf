###########
# Dateien #
###########

# Text aus Datei zeilenweise lesen
print("---------------------------")
txtfile = open("input.txt", "r", encoding="utf-8")  # Benötigt "input.txt" im aktuellen Verzeichnis!
for line in txtfile:
    print(line.rstrip())  # .rstrip() entfernt white spaces am rechten Ende eines Strings
txtfile.close()


# Komplettes Einlesen
print("---------------------------")
with open("input.txt", encoding="utf-8") as inp:
    text_list = inp.readlines()
with open("input.txt", encoding="utf-8") as inp:
    text_string = inp.read()
print(text_list)
print(text_string)


# Schreiben in Datei
print("---------------------------")
in_file = open("input.txt", "r", encoding="utf-8")
out_file = open("output.txt", "w", encoding="utf-8")
i = 1
for line in in_file:
    out = f"{str(i)}: {line.rstrip()}"
    print(out)
    out_file.write(out + "\n")
    i += 1
in_file.close()
out_file.close()
