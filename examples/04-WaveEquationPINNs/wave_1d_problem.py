from mtc.problem import *
from cfg import p

import numpy as np

[x, t], [u] = p.add_neural_network(
    name="wave1d",
    inputs=["x", "t"],
    outputs=["u"],
)

L = float(np.pi)
geo = p.Line1D("geom", 0, L)

interior = p.add_interior_subdomain("interior", geom=geo, params={t: (0, 2 * L)})

initial_t0 = p.add_interior_subdomain("initial_t0", geom=geo, params={t: (0, 0)})

boundary = p.add_boundary_subdomain("boundary", geom=geo, params={t: (0, 2 * L)})


c = 1.0
wave_eq = Eq(u.diff(t, 2), (c**2 * u.diff(x)).diff(x))

from sympy import sin

p.set_constraints(
    {
        "initial_u": enforce(equation=Eq(u, sin(x)), on_domain=initial_t0),
        "initial_u_t": enforce(equation=Eq(u.diff(t), sin(x)), on_domain=initial_t0),
        "boundary": enforce(equation=Eq(u, 0), on_domain=boundary),
        "wave_equation": enforce(equation=wave_eq, on_domain=interior),
    }
)
