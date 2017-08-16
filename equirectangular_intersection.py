#!/usr/bin/env python3

from sympy import *

o_x = Symbol('o_x', real=True)
o_y = Symbol('o_y', real=True)
o_z = Symbol('o_z', real=True)
d_x = Symbol('d_x', real=True)
d_y = Symbol('d_y', real=True)
d_z = Symbol('d_z', real=True)
t = Symbol('t', real=True, positive=True)

c = Symbol('c', real=True)


x = o_x + t*d_x
y = o_y + t*d_y
z = o_z + t*d_z

cone = Eq((x**2+y**2)/c**2, z**2)

init_printing()
pprint(cone)

print("Line equation:\n")
print("x =", x)
print("y =", y)
print("z =", z)

print("\nCone Equation\n")
pprint(cone)

pprint("\tt solution\n")

pprint(simplify(solve(cone, t)[0]))
pprint(simplify(solve(cone, t)[1]))
# print(simplify(solve(cone, t)[0]))
# print(latex(simplify(solve(cone, t)[0])))
