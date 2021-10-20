##############
# Funktionen #
##############

# Ohne Rückgabewert
def print_km_to_miles(km):
    """Prints the km distance and respective distance in miles"""
    print(f"{km:5.1f} km ~= {km / 1.61:5.1f} miles")


print("---------------------------")
print_km_to_miles(20)


# Mit Rückgabewert
def km_to_miles(km):
    """Returns the km distance in miles."""
    return km / 1.61


print("---------------------------")
for distance in [1, 10, 80, 200]:
    print(f"{distance:5.1f} km ~= {km_to_miles(distance):5.1f} miles")


# Optionale Parameter und mehrere Rückgabewerte
def func_with_defaults(a, b, c=0, d=0, e=1):
    res1 = (a + b + c + d) * e  # lokale Definition von res1
    res2 = (a + b + c + d) / e  # lokale Definition von res2
    return res1, res2


func_with_defaults(1, 1, e=10)
# print(res1, res2)  # Error (diese Variablen stehen global nicht zur Verfügung)
var1, var2 = func_with_defaults(1, 1, e=10)
print(var1, var2)


print("---------------------------")
print(func_with_defaults(1, 1))
print(func_with_defaults(1, 1, 9, 9, 2))
print(func_with_defaults(5, 5, e=10))


# Globale und lokale Objekte
global_var = 0
def func(integer):
    global global_var
    local_var = integer ** 2
    print(global_var)
    global_var += local_var  # benötigt 'global' keyword


func(10)
func(5)
print(global_var)
# print(local_var)  # Error


# Beliebige Anzahl von Paramtern
def mean(*values):
    sum_values = 0
    for val in values:
        sum_values += val
    return sum_values / len(values)


def print_kws(**key_vals):
    print(key_vals)


print("---------------------------")
print(mean(1, 2, 3, 4, 5, 6, 7, 8, 9))
print_kws(de="german", en="english", fr="french")


# Semantik von "*"
def func_1(*params):
    print(params)


print("---------------------------")
# Paramter werden in ein Tupel gepackt
func_1("My", "answer", "is", 42)
liste = ["My", "answer", "is", 42]
# liste wird als einzelner Parameter interpretiert und in ein Tupel gepackt
func_1(liste)
# liste wird entpackt und die Parameter in ein Tupel gepackt
func_1(*liste)


def func_2(a, b, c, d):
    print(a, b, c, d)


# liste wird entpackt
func_2(*liste)


# Semantik von "**"
def func_3(**key_vals):
    print(key_vals)


def func_4(de, en, fr):
    print(de, en, fr)


print("---------------------------")
# Key-Value Paare werden zusammengefasst
func_3(de="German", en="English", fr="French")
dictionary = dict(de="German", en="English", fr="French")
# Key-Value Paare werden aus dictionary entpackt
func_3(**dictionary)
# Dictionary wird entpackt und Key-Value Paare als Schlüsselwortparameter verwendet
func_4(**dictionary)


# Kombinationen
def f(a, b, m, n, x, y):
    print(a, b, m, n, x, y)


print("---------------------------")
liste = ["l1", "l2"]
dic = {"x": "d1", "y": "d2"}
f("a", "b", *liste, **dic)


def g(a, b, *arguments, **keywords):
    print(f"a = {a}")
    print(f"b = {b}")
    for arg in arguments:
        print(f"arg: {arg}")
    for kw in keywords:
        print(f"{kw}: {keywords[kw]}")


print("---------------------------")
g(1, 2, 3, 4, 5, 6, type="int", func="g")


# Funktionen in Funktionen
def f():
    def g():
        print("Hello, it's 'g'.")
        print("Thanks for calling me.")

    print("This is 'f'.")
    print("We'll call 'g' now.")
    g()


print("---------------------------")
f()


# Funktionen als Parameter
def g():
    print("Hello, it's 'g'.")
    print("Thanks for calling me.")


def f(func):
    print("This is 'f'.")
    print("We'll call '{}' now.".format(func.__name__))
    func()


print("---------------------------")
f(g)


# Funktionen als Rückgabe
def f(x):
    def g(y):
        return y + x

    return g


print("---------------------------")
f1 = f(1)
f5 = f(5)
print(f1(10))
print(f5(10))
