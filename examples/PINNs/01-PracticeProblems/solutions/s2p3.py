from cfg import *

[x, y, oh], [u, v, pp] = p.add_neural_network(name="NN", inputs=["x", "y", "oh"], outputs=["u", "v", "p"])

w,h = 2,1
r1 = p.Rectangle("r1", (0,0), (w,h))
ch1 = p.Channel2D("ch1", (0,0), (w,h))

params = {oh: (0,0.95)}

# subr = p.Rectangle("subr1", (w/2-.1,0), (w/2+.1,h*.8))
subr = p.Rectangle("subr1", (w/2-.1,0), (w/2+.1,oh), params=params)
ch1 = p.GeometryDifference("gd", ch1, subr)

inlet = p.add_boundary_subdomain("inlet", geom=r1, criteria=Eq(x,0), params=params)
outlet = p.add_boundary_subdomain("outlet", geom=r1, criteria=Eq(x,w), params=params)

r_interior = p.add_interior_subdomain("r_interior", geom=r1, params=params)
ch_interior = p.add_interior_subdomain("ch_interior", geom=ch1, params=params)

noslip = p.add_boundary_subdomain("no_slip", geom=ch1, params=params)


# p.add_constraint("c1", enforce(equation=Eq(x,0), on_domain=r_interior))
p.add_constraint("inlet_u", enforce(equation=Eq(u,1), on_domain=inlet))
p.add_constraint("inlet_v", enforce(equation=Eq(v,0), on_domain=inlet))

p.add_constraint("outlet_p", enforce(equation=Eq(pp,0), on_domain=outlet))

p.add_constraint("noslip_u", enforce(equation=Eq(u,0), on_domain=noslip))
p.add_constraint("noslip_v", enforce(equation=Eq(v,0), on_domain=noslip))

nu = 0.02
p.add_constraint("continuity", enforce(equation=Eq(u.diff(x)+v.diff(y),0), 
                                       on_domain=ch_interior))

p.add_constraint("momentum_x", enforce(equation=Eq(u*u.diff(x)+v.diff(y)+pp.diff(x),
                                                   nu*(u.diff(x,x) + u.diff(y,y))), 
                                       on_domain=ch_interior))

p.add_constraint("momentum_y", enforce(equation=Eq(u*u.diff(x)+v.diff(y)+pp.diff(y),
                                                   nu*(v.diff(x,x) + v.diff(y,y))), 
                                       on_domain=ch_interior))

# Adding mass balance
import sympy as sp
nx = sp.Symbol("normal_x")
ny = sp.Symbol("normal_y")

int_rect = p.Rectangle("int_rect", (0,0), (w/2,h))
# int_rect = p.GeometryDifference("gd1", ch1, int_rect)
int_rect = p.GeometryDifference("gd1", ch1, int_rect)


inlet_outlet = p.add_boundary_subdomain("inlet_outlet", geom=r1, criteria=Eq(x,0) | Eq(x,w), params=params)

int_constrict = p.add_boundary_subdomain("int_constrict", geom=int_rect, 
                                         criteria=Eq(x,0) | Eq(x,w/2), params=params)

p.add_constraint("mass_balance_io", enforce(equation=Eq(Integral(nx*u+ny*v, x,y),0), 
                                         on_domain=inlet_outlet))

p.add_constraint("mass_balance_middle", enforce(equation=Eq(Integral(nx*u+ny*v, x,y),0), 
                                         on_domain=int_constrict))
