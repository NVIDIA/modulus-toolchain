from cfg import *
import sympy as sp

# HEAT

[x, y], [T] = p.add_neural_network(
    name="NN_Tsolid", inputs=["x", "y"], outputs=["Tsolid"]
)

geo_solid = p.Rectangle("rect", (0, 0), (1, 1))
interior_solid = p.add_interior_subdomain("interior_solid", geom=geo_solid)
bdry_solid = p.add_boundary_subdomain("bdry_solid", geom=geo_solid, criteria=y > 0)
bdry_heat_src = p.add_boundary_subdomain(
    "bdry_heat_src", geom=geo_solid, criteria=Eq(y, 0) & (Abs(x - 0.5) < 0.1)
)

Dsolid = 0.0625
p.add_constraint(
    "diffusion_solid",
    enforce(
        equation=Eq(Dsolid * (T.diff(x, x) + T.diff(y, y)), 0), on_domain=interior_solid
    ),
)
p.add_constraint("diffusion_solid_bc", enforce(equation=Eq(T, 0), on_domain=bdry_solid))

ny = sp.Symbol("normal_y")
p.add_constraint(
    "heat_source", enforce(equation=Eq(ny * T.diff(y), 1), on_domain=bdry_heat_src)
)


# Air flow
a_end = 5
rect = p.Rectangle("rect_air", (-1, 0), (a_end, 2))

geo_air = p.GeometryDifference("gd0", rect, geo_solid)
interior_air = p.add_interior_subdomain(
    "interior_air", geom=geo_air, compute_sdf_derivatives=True
)
inlet = p.add_boundary_subdomain("inlet", geom=geo_air, criteria=Eq(x, -1))
outlet = p.add_boundary_subdomain("outlet", geom=geo_air, criteria=Eq(x, a_end))
noslip = p.add_boundary_subdomain(
    "noslip", geom=geo_air, criteria=(x > -1) & (x < a_end)
)

p.add_constraint(
    "continuity", enforce(equation=Eq(ny * T.diff(y), 1), on_domain=interior_air)
)

p.add_constraint("inlet", enforce(equation=Eq(ny * T.diff(y), 1), on_domain=inlet))
p.add_constraint("outlet", enforce(equation=Eq(ny * T.diff(y), 1), on_domain=outlet))
p.add_constraint("noslip", enforce(equation=Eq(ny * T.diff(y), 1), on_domain=noslip))
