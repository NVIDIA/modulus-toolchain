from cfg import *

[x], [ua, ub] = p.add_neural_network(name="NN", inputs=["x"], outputs=["ua", "ub"])

DA = 1
DB = 1 / 100

geom = p.Line1D("geomA", -1, 0)
interior = p.add_interior_subdomain("interiorA", geom=geom)

middle = p.add_boundary_subdomain("middle", geom=geom, criteria=Eq(x, 0))

bdry = p.add_boundary_subdomain("bdryA", geom=geom, criteria=Eq(x, -1))
diff_eq = Eq(DA * ua.diff(x, 2) + 1, 0)
p.add_constraint("diffusionA", enforce(equation=diff_eq, on_domain=interior))
p.add_constraint("bdryA", enforce(equation=Eq(ua, 0), on_domain=bdry))

geom = p.Line1D("geomB", 0, 1)
interior = p.add_interior_subdomain("interiorB", geom=geom)
bdry = p.add_boundary_subdomain("bdryB", geom=geom, criteria=Eq(x, 1))
diff_eq = Eq(DB * ub.diff(x, 2) - 1 / 2, 0)
p.add_constraint("diffusionB", enforce(equation=diff_eq, on_domain=interior))
p.add_constraint("bdryB", enforce(equation=Eq(ub, 0), on_domain=bdry))

p.add_constraint("uAuB", enforce(equation=Eq(ub, ua), on_domain=middle))
p.add_constraint(
    "GraduAuB", enforce(equation=Eq(DB * ub.diff(x), DA * ua.diff(x)), on_domain=middle)
)

p.set_model(
    "f",
    [
        {"func": ua, "on": x <= 0},
        {"func": ub, "on": x > 0},
    ],
)
