from cfg import *

[x, y, rot], [u, pp, v] = p.add_neural_network(
    name="NN", inputs=["x", "y", "rot"], outputs=["u", "p", "v"]
)

# geometry
import numpy as np

lines = [[0, 0], [1, 0], [1, 1]]

from sympy import Number, Symbol, Heaviside, atan, sin, cos, sqrt


# Naca implementation modified from https://stackoverflow.com/questions/31815041/plotting-a-naca-4-series-airfoil
# https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
def camber_line(x, m, p, c):
    cl = []
    for xi in x:
        cond_1 = Heaviside(xi, 0) * Heaviside((c * p) - xi, 0)
        cond_2 = Heaviside(-xi, 0) + Heaviside(xi - (c * p), 0)
        v_1 = m * (xi / p**2) * (2.0 * p - (xi / c))
        v_2 = m * ((c - xi) / (1 - p) ** 2) * (1.0 + (xi / c) - 2.0 * p)
        cl.append(cond_1 * v_1 + cond_2 * v_2)
    return cl


def dyc_over_dx(x, m, p, c):
    dd = []
    for xi in x:
        cond_1 = Heaviside(xi) * Heaviside((c * p) - xi)
        cond_2 = Heaviside(-xi) + Heaviside(xi - (c * p))
        v_1 = ((2.0 * m) / p**2) * (p - xi / c)
        v_2 = (2.0 * m) / (1 - p**2) * (p - xi / c)
        dd.append(atan(cond_1 * v_1 + cond_2 * v_2))
    return dd


def thickness(x, t, c):
    th = []
    for xi in x:
        term1 = 0.2969 * (sqrt(xi / c))
        term2 = -0.1260 * (xi / c)
        term3 = -0.3516 * (xi / c) ** 2
        term4 = 0.2843 * (xi / c) ** 3
        term5 = -0.1015 * (xi / c) ** 4
        th.append(5 * t * c * (term1 + term2 + term3 + term4 + term5))
    return th


def naca4(x, m, p, t, c=1):
    th = dyc_over_dx(x, m, p, c)
    yt = thickness(x, t, c)
    yc = camber_line(x, m, p, c)
    line = []
    for (xi, thi, yti, yci) in zip(x, th, yt, yc):
        line.append((xi - yti * sin(thi), yci + yti * cos(thi)))
    x.reverse()
    th.reverse()
    yt.reverse()
    yc.reverse()
    for (xi, thi, yti, yci) in zip(x, th, yt, yc):
        line.append((xi + yti * sin(thi), yci - yti * cos(thi)))
    return line


m = 0.02
ppp = 0.4
t = 0.12
c = 1.0

# make naca geometry
xx = [x for x in np.linspace(0, 0.2, 20 // 4)] + [x for x in np.linspace(0.2, 1.0, 10)][
    1:
]  # higher res in front
line = naca4(xx, m, ppp, t, c)[:-1]

# # lines = [x for x in np.linspace(0, 0.2, 20)] + [x for x in np.linspace(0.2, 1.0, 10)][1:]

params = {rot: (0, -np.pi / 6)}

tri = p.Polygon("poly", line, rotate=(rot, "z"), params=params)
# tri = p.Rectangle("poly", (0,0), (1,1), rotate=(rot, "z"),params=params)


channel_length = 15.0 / 2
channel_height = 10.0 / 2
a = 0.3

# inlet = Line((-channel_length*a, -channel_height/2), (-channel_length*a, channel_height/2), normal=1)
# outlet = Line((channel_length*(1-a), -channel_height/2), (channel_length*(1-a), channel_height/2), normal=1)
channel_rect = p.Rectangle(
    "channel_rect",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
)
channel = p.Channel2D(
    "channel",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
)


domain_geom = p.GeometryDifference("dg", channel, tri)

interior = p.add_interior_subdomain(
    "interior", geom=domain_geom, compute_sdf_derivatives=True, params=params
)
top_bot = p.add_boundary_subdomain(
    "top_bot", geom=domain_geom, criteria=Eq(Abs(y), channel_height / 2), params=params
)
inlet = p.add_boundary_subdomain(
    "inlet", geom=channel_rect, criteria=Eq(x, -channel_length * a), params=params
)

outlet = p.add_boundary_subdomain(
    "outlet", geom=channel_rect, criteria=Eq(x, channel_length * (1 - a)), params=params
)

airfoil_bdry = p.add_boundary_subdomain("airfoil_bdry", geom=tri)

lower_rec = p.Rectangle(
    "lower_rec",
    (-channel_length * a, -channel_height / 2),
    (-1, channel_height / 2),
    params=params,
)
lower_rec = p.add_boundary_subdomain(
    "lower_rec", geom=lower_rec, criteria=Eq(x, -channel_length * a) | Eq(x, -1)
)

lower_rec2 = p.Rectangle(
    "lower_rec2",
    (-channel_length * a, -channel_height / 2),
    (2, channel_height / 2),
    params=params,
)
lower_rec2 = p.add_boundary_subdomain(
    "lower_rec2", geom=lower_rec2, criteria=Eq(x, -channel_length * a) | Eq(x, 2)
)
inlet_outlet = p.Rectangle(
    "inlet_outlet",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
    params=params,
)
inlet_outlet = p.add_boundary_subdomain(
    "inlet_outlet",
    geom=inlet_outlet,
    criteria=Eq(x, -channel_length * a) | Eq(x, channel_length * (1 - a)),
)

import sympy as sp

normal_x = sp.Symbol("normal_x")
normal_y = sp.Symbol("normal_y")

# p.add_constraint(f"no_flux_1",
#                  enforce(equation=Eq(Integral((normal_x*u+normal_y*v), x, y), 0),
#                          on_domain=lower_rec))
# p.add_constraint(f"no_flux_2",
#                  enforce(equation=Eq(Integral((normal_x*u+normal_y*v), x, y), 0),
#                          on_domain=lower_rec2))
# p.add_constraint(f"noflux_inlet_outlet",
#                  enforce(equation=Eq(Integral((normal_x*u+normal_y*v), x, y), 0),
#                          on_domain=inlet_outlet))


sdf = sympy.Function("sdf")(x, y, rot)
from sympy import sqrt, Min, Abs

# Zero Equation
nu = (
    sqrt((u.diff(y) + v.diff(x)) ** 2 + 2 * u.diff(x) ** 2 + 2 * v.diff(y) ** 2)
    * Min(0.045, 0.419 * sdf) ** 2
    + 6.25e-6
)
nu = p.add_submodel("nu", nu)

# N-S Momentum equations
m_x = (
    -1.0 * nu * u.diff(x).diff(x)
    - 1.0 * nu * u.diff(y).diff(y)
    + 1.0 * u * u.diff(x)
    + 1.0 * v * u.diff(y)
    - 1.0 * nu.diff(x) * u.diff(x)
    - 1.0 * nu.diff(y) * u.diff(y)
    + pp.diff(x)
)
momentum_x = Eq(m_x, 0)

m_y = (
    -1.0 * nu * v.diff(x).diff(x)
    - 1.0 * nu * v.diff(y).diff(y)
    + 1.0 * u * v.diff(x)
    + 1.0 * v * v.diff(y)
    - 1.0 * nu.diff(x) * v.diff(x)
    - 1.0 * nu.diff(y) * v.diff(y)
    + pp.diff(y)
)
momentum_y = Eq(m_y, 0)

continuity_eq = Eq(u.diff(x) + v.diff(y), 0)

p.add_constraint(
    "interior_continuity", enforce(equation=continuity_eq, on_domain=interior)
)
p.add_constraint(
    "interior_momentum_x", enforce(equation=momentum_x, on_domain=interior)
)
p.add_constraint(
    "interior_momentum_y", enforce(equation=momentum_y, on_domain=interior)
)

p.add_constraint("airfoil_bdry_u", enforce(equation=Eq(u, 0), on_domain=airfoil_bdry))
p.add_constraint("airfoil_bdry_v", enforce(equation=Eq(v, 0), on_domain=airfoil_bdry))

p.add_constraint("top_bot_u", enforce(equation=Eq(u, 1), on_domain=top_bot))
p.add_constraint("top_bot_v", enforce(equation=Eq(v, 0), on_domain=top_bot))

p.add_constraint("inlet_u", enforce(equation=Eq(u, 1), on_domain=inlet))
p.add_constraint("inlet_v", enforce(equation=Eq(v, 0), on_domain=inlet))

p.add_constraint("outlet_p", enforce(equation=Eq(pp, 0), on_domain=outlet))
