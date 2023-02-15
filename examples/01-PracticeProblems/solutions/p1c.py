from cfg import *

[x], [ua, ub] = p.add_neural_network(name="NN", inputs=["x"], outputs=["ua", "ub"])

geom = p.Line1D("geomA", -1,0)
interior = p.add_interior_subdomain("interiorA", geom=geom)

middle = p.add_boundary_subdomain("middle", geom=geom, criteria=Eq(x,0))

bdry = p.add_boundary_subdomain("bdryA", geom=geom, criteria=Eq(x,-1))
diff_eq = Eq(ua.diff(x,2) + 1, 0)
p.add_constraint("diffusionA", enforce(equation=diff_eq, on_domain=interior))
p.add_constraint("middleA", enforce(equation=Eq(ua,0), on_domain=middle))
p.add_constraint("bdryA", enforce(equation=Eq(ua.diff(x),0), on_domain=bdry))

geom = p.Line1D("geomB", 0,1)
interior = p.add_interior_subdomain("interiorB", geom=geom)
bdry = p.add_boundary_subdomain("bdryB", geom=geom, criteria=Eq(x,1))
diff_eq = Eq(ub.diff(x,2) - 1, 0)
p.add_constraint("diffusionB", enforce(equation=diff_eq, on_domain=interior))
p.add_constraint("middleB", enforce(equation=Eq(ub,0), on_domain=middle))
p.add_constraint("bdryB", enforce(equation=Eq(ub.diff(x),0), on_domain=bdry))

p.set_model(
    "f",
    [
        {"func": ua, "on": x<=0},
        {"func": ub, "on": x>0},
    ],
)