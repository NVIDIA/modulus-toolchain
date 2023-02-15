from cfg import *
import numpy as np
from sympy import exp, sin

# Vel is also an input  
[x, t, k], [u] = p.add_neural_network(name="wave1d", inputs=["x", "t", "k"], outputs=["u"])

# Geometry + domains
L = float(np.pi)
geo = p.Line1D("geom", 0, L)
interior = p.add_interior_subdomain("interior", geom=geo, params={t:(0,2*L), k:(1,10)})
initial_t0 = p.add_interior_subdomain("initial_t0", geom=geo, params={t:(0,0), k:(1,10)}) 
boundary = p.add_boundary_subdomain("boundary", geom=geo, params={t:(0,2*L), k:(1,10)})

# Generate velocity
vel = 1.0 + 1.0 / (1.0 + exp(-2*k*(x-1.5)))

# PDE constraints
wave_eq = Eq(u.diff(t, 2) - vel**2 * u.diff(x, 2), 0)
p.add_constraint("wave_equation", enforce(equation=wave_eq, on_domain=interior))

# Initial conditions
initial_p = Eq(u, sin(x))
initial_dp_dt = Eq(u.diff(t), sin(x))
p.add_constraint("initial_p", enforce(equation=initial_p, on_domain=initial_t0))
p.add_constraint("initial_dp_dt", enforce(equation=initial_dp_dt, on_domain=initial_t0))

# Boundary conditions
boundary_cond = Eq(u, 0)
p.add_constraint("boundary", enforce(equation=boundary_cond, on_domain=boundary))