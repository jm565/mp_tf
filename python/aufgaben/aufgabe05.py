"""
Schreiben Sie eine Funktion 'palindrom', die feststellt, ob eine übergebene Zeichenkette ein Palindrom
ist oder nicht. Vom Benutzer soll anschließend eine Zeichenkette eingegeben werden und das Python-Programm
soll ausgeben ob es sich um ein Palindrom handelt oder nicht.
Hinweis: Ein Palindrom ist ein String, der von vorn und von hinten gelesen gleich ist.
"""


##############
# Variante 1 #
##############
def palindrom(s):
    for i in range(len(s)):
        if s[i] != s[-1 - i]:
            return False
    return True


##############
# Variante 2 #
##############
def palindrom_list(s):
    l1 = list(s)
    l2 = l1[:]
    l2.reverse()
    is_palindrom = True if l1 == l2 else False
    return is_palindrom


##############
# Variante 3 #
##############
def palindrom_slice(s):
    is_p = True if s == s[::-1] else False
    return is_p


print("Bestimmung ob es sich bei der Zeichenkette s um ein Palindrom handelt.")
inp = input("s = ")
inp = inp.lower()

is_palindrom = palindrom(inp)
is_palindrom2 = palindrom_list(inp)
is_palindrom3 = palindrom_slice(inp)

# Check ob die Rückgaben übereinstimmen
assert is_palindrom == is_palindrom2
assert is_palindrom2 == is_palindrom3

if is_palindrom:
    print(f"{inp} ist ein Palindrom!")
else:
    print(f"{inp} ist kein Palindrom!")
