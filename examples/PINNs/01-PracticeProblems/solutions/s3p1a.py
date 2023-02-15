from cfg import *

[x, y], [T] = p.add_neural_network(name="NN_Tsolid", inputs=["x", "y"], outputs=[ "Tsolid"])

geo_solid = p.Rectangle("rect", (0,0), (1,1))
interior_solid = p.add_interior_subdomain("interior_solid", geom=geo_solid)
bdry_solid = p.add_boundary_subdomain("bdry_solid", geom=geo_solid, criteria=y>0)
bdry_heat_src = p.add_boundary_subdomain("bdry_heat_src", geom=geo_solid, 
                                         criteria=Eq(y,0) & (Abs(x-0.5)<0.1) )

p.add_constraint("diffusion_solid", enforce(equation=Eq(x,0), on_domain=interior_solid))
p.add_constraint("diffusion_solid_bc", enforce(equation=Eq(x,0), on_domain=bdry_solid))
p.add_constraint("heat_source", enforce(equation=Eq(x,0), on_domain=bdry_heat_src))
