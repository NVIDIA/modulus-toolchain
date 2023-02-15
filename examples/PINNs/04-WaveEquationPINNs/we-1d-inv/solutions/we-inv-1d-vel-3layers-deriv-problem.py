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

# Interfaces
x_int1 = 1.045
x_int2 = 2.095
eps = 0.2
b1 = 0.8
b2 = 1.2
b3 = 1.9
b4 = 2.3

band1 = p.Line1D("band1", b1, b2)
band2 = p.Line1D("band2", b3, b4)
diff_temp = p.GeometryDifference("diff_temp", geo, band1)
diff = p.GeometryDifference("diff", diff_temp, band2)
interior_layers = p.add_interior_subdomain("layers", geom=diff, params={t:(0, 2*L)})

# interior_layers = p.add_interior_subdomain("layers", geom=geo, criteria=Or( (x < b1), (x  x_int1+eps), (x < x_int2-eps), (x > x_int2+eps) ), params={t:(0,2*L)})

# # Bounds for interior 1
# x1=0.0
# x2=1.03

# # Bounds for interior 2
# x3=1.1
# x4=2.0

# # Bounds for interior 3
# x5=2.2
# x6=3.1

# # Define interior 1 and 2
# interior1 = p.add_interior_subdomain("interior1", geom=geo, criteria=And(x > x1, x < x2), params={t:(0,2*L)})
# interior2 = p.add_interior_subdomain("interior2", geom=geo, criteria=And(x > x3, x < x4), params={t:(0,2*L)})
# interior3 = p.add_interior_subdomain("interior3", geom=geo, criteria=And(x > x5, x < x6), params={t:(0,2*L)})

#################### PDEs + constraints ##################
# 1. PDE constraints
wave_eq = Eq(u.diff(t, 2) - vel**2 * u.diff(x, 2), 0)
p.add_constraint("wave_equation", enforce(equation=wave_eq, on_domain=interior))

# 2. Initial conditions
initial_p = Eq(u, sin(x))
initial_dp_dt = Eq(u.diff(t), sin(x))
p.add_constraint("initial_p", enforce(equation=initial_p, on_domain=initial_t0))
p.add_constraint("initial_dp_dt", enforce(equation=initial_dp_dt, on_domain=initial_t0))

# 3. Boundary conditions
boundary_cond = Eq(u, 0)
p.add_constraint("boundary", enforce(equation=boundary_cond, on_domain=boundary))

# 4. Velocity constraints
vel_space_invariant = Eq(vel.diff(x, 1), 0)
# p.add_constraint("vel_space_invariant1", enforce(equation=vel_space_invariant, on_domain=interior1))
# p.add_constraint("vel_space_invariant2", enforce(equation=vel_space_invariant, on_domain=interior2))
# p.add_constraint("vel_space_invariant3", enforce(equation=vel_space_invariant, on_domain=interior3))
p.add_constraint("vel_space_invariant", enforce(equation=vel_space_invariant, on_domain=interior_layers))

# 5. Data constraints
hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel-3layers/data_fd_resampled-subx10-subt100.hdf5"
p.add_data_constraint(name="observed_data", model=u, data_fname=hdf_fname)