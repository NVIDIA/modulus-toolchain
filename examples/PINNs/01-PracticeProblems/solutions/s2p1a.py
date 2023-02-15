from cfg import *

[x, y], [u] = p.add_neural_network(name="NN", inputs=["x", "y"], outputs=["u"])

w,h = 2,1
r1 = p.Rectangle("r1", (0,0), (w,h))
ch1 = p.Channel2D("ch1", (0,0), (w,h))

inlet = p.add_boundary_subdomain("inlet", geom=r1, criteria=Eq(x,0))
outlet = p.add_boundary_subdomain("outlet", geom=r1, criteria=Eq(x,w))

r_interior = p.add_interior_subdomain("r_interior", geom=r1)
ch_interior = p.add_interior_subdomain("ch_interior", geom=ch1)

noslip = p.add_boundary_subdomain("no_slip", geom=ch1)

# p.add_constraint("c1", enforce(equation=Eq(x,0), on_domain=r_interior))
p.add_constraint("b1", enforce(equation=Eq(x,0), on_domain=inlet))
p.add_constraint("b2", enforce(equation=Eq(x,0), on_domain=outlet))
p.add_constraint("b3", enforce(equation=Eq(x,0), on_domain=noslip))
p.add_constraint("c2", enforce(equation=Eq(x,0), on_domain=ch_interior))
