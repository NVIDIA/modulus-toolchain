from cfg import *
import numpy as np
from sympy import exp, sin

################ Neural Network defintion ################
# Create NN to predict pressure (x,t) -> p
[x, t], [u] = p.add_neural_network(name="pressure", inputs=["x", "t"], outputs=["u"])

# Create NN to predict velocity (x) -> vel
[x], [vel] = p.add_neural_network(name="velocity", inputs=["x"], outputs=["vel"])

######################### Geometry #######################
# Geometry + domains
L = float(np.pi)
geo = p.Line1D("geom", 0, L)
interior = p.add_interior_subdomain("interior", geom=geo, params={t:(0,2*L)})
initial_t0 = p.add_interior_subdomain("initial_t0", geom=geo, params={t:(0,0)}) 
boundary = p.add_boundary_subdomain("boundary", geom=geo, params={t:(0,2*L)})

# Old method 
# Bounds for interior 1
# x1=0.5
# x2=1.5

# # Bounds for interior 2
# x3=1.7
# x4=3

# # Define interior 1 and 2
# interior1 = p.add_interior_subdomain("interior1", geom=geo, criteria=And(x > x1, x < x2), params={t:(0,2*L)})
# interior2 = p.add_interior_subdomain("interior2", geom=geo, criteria=And(x > x3, x < x4), params={t:(0,2*L)})
## 

# Interface position
# x_int = 1.565
# eps = 0.02
interior_layers = p.add_interior_subdomain("layers", geom=geo, criteria=Or( (x < 1.5), (x > 1.7) ), params={t:(0,2*L)})

#################### PDEs + constraints ##################
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

vel_space_invariant = Eq(vel.diff(x, 1), 0)
# p.add_constraint("vel_space_invariant1", enforce(equation=vel_space_invariant, on_domain=interior1))
# p.add_constraint("vel_space_invariant2", enforce(equation=vel_space_invariant, on_domain=interior2))
p.add_constraint("vel_space_invariant", enforce(equation=vel_space_invariant, on_domain=interior_layers))

# 5. Data constraints
hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel-2layers-deriv_constraint/data_fd_resampled-subx10-subt100.hdf5"
p.add_data_constraint(name="observed_data", model=u, data_fname=hdf_fname)