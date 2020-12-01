########################
# Formatierte Ausgaben #
########################

# Kommaseparierte Liste
print("---------------------------")

a, b, c = 1, 2.2, "test"
print(a, a+b, c, "d")
print(a, a+b, c, "d", sep=",")
print(a, a+b, c, "d", end=" | ")
print(a, a+b, c, "d")


# String-Konkatenation
print("---------------------------")
concat = str(a) + "," + str(a+b) + "," + c + ",d"
print(concat)


# String-Modulo-Operator
print("---------------------------")
n = 356.08977
m = 101.454
print("n = %10.2f, m = %10.2f" % (n, m))
print("n = %10.2e, m = %10.2E" % (n, m))


# String-Methode "format"
print("---------------------------")
a = 30.89
b = 11
print("Erstes Argument: {}, zweites: {}".format(a, b))
print("Erstes Argument: {0}, zweites: {1}".format(a, b))
print("Erstes Argument: {1}, zweites: {0}".format(a, b))
print("Erstes Argument: {1:5d}, zweites: {0:5.1f}".format(a, b))
print("precisions: {0:5.2f} or {0:5.1f}".format(10.169))
print("Die Hauptstadt von {land:s} ist {stadt:s}".format(land="Deutschland", stadt="Berlin"))

# Format-Strings
print("---------------------------")
x = 30.8
p = 3.141592
print("x = {:.2f}, p = {:.2f}".format(x, p))
print(f"x = {x:.2f}, p = {p:.2f}")
