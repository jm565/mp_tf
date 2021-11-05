#############
# Schleifen #
#############

# while-loop (Summe der ersten n Integer)
print("---------------------------")
n = 20
i = 1
sum_n = 0
while i <= n:
    print(i, end=' ')
    sum_n += i
    i += 1  # i = i + 1
else:
    print("\nWe are done.")
print(f"Sum = {sum_n}")

# continue-break-statements (Zahlenratespiel)
print("---------------------------")
import random

to_be_guessed = random.randint(1, n)
guess = 0
print(f"Errate meine Zahl von 1 bis {n}!\nMit '0' kannst du aufhören.")
while guess != to_be_guessed:
    guess = int(input("Neuer Versuch: "))
    if 0 < guess <= n:
        if guess > to_be_guessed:
            print("Zu groß.")
        elif guess < to_be_guessed:
            print("Zu klein.")
    elif guess == 0:
        print("Schade, dass du aufgibst!")
        break
    else:
        print(f"Nur Zahlen von 1 bis {n}!")
        continue
else:
    print(f"Gratuliere, die gesuchte Zahl war {to_be_guessed}!")


# Iterieren über Listen
print("---------------------------")
languages = ["C", "C++", "Java", "Python"]
for lang in languages:
    print(lang)
else:
    print("No more languages.")


# for-loops
print("---------------------------")
sum_n = 0
for i in range(1, n+1):
    sum_n += i
print(f"Sum of first {n} integers = {sum_n}")

for i in range(len(languages)):
    print(i, languages[i])

for i, language in enumerate(languages):
    print(i, language)
