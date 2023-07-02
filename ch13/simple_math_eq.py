from sympy import *

a,b,c = symbols('a b c', Positive = True, Real = True)
x1 = symbols('\Delta_t', Real = True)

x1 = (-b + sqrt(b ** 2 - 4*a*c))/(2*a)
pprint(x1)