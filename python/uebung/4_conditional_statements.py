########################
# Bedingte Anweisungen #
########################

# standard if-elif-else statement
print("---------------------------")
person = input("Nationality: ")
if person.lower() == "german":
    print("Hallo.")
    print("Wie gehts dir?")
elif person.lower() == "french":
    print("Salut.")
elif person.lower() == "spanish":
    print("Hola.")
else:
    print("We default to english.\nHello.")


# short if-else statement
print("---------------------------")
num_1 = float(input("Number 1: "))
num_2 = float(input("Number 2: "))
num_max = num_1 if (num_1 > num_2) else num_2
print(f"The maximum of {num_1} and {num_2} is {num_max}")
