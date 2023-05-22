from cfg import *
import numpy as np
from sympy import exp, sin

# Create NN to predict pressure (x,t) -> p
[x, t], [u] = p.add_neural_network(name="pressure", inputs=["x", "t"], outputs=["u"])

# Create NN to predict velocity (x,t) -> vel
[x], [vel] = p.add_neural_network(name="velocity", inputs=["x"], outputs=["vel"])

# Geometry
L = float(np.pi)
geo = p.Line1D("geom", 0, L)

# Domains
interior = p.add_interior_subdomain("interior", geom=geo, params={t:(0,2*L)})
initial_t0 = p.add_interior_subdomain("initial_t0", geom=geo, params={t:(0,0)}) 
boundary = p.add_boundary_subdomain("boundary", geom=geo, params={t:(0,2*L)})

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

# Prior knowledge on velocity
vel_space_invariant = Eq(vel.diff(x, 1), 0)
p.add_constraint("vel_space_invariant", enforce(equation=vel_space_invariant, on_domain=interior))

# Data constraints
# hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel/data_fd_resampled-1000.hdf5"
hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel-derv_constraint/data_fd_resampled-subx10-subt100.hdf5"
p.add_data_constraint(name="observed_data", model=u, data_fname=hdf_fname)

