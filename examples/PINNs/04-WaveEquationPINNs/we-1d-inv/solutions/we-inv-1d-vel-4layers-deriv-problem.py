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

# # Bounds for interior 0
# x1=0.0
# x2=0.7

# # Bounds for interior 1
# x3=0.8
# x4=1.5

# # Bounds for interior 2
# x5=1.6
# x6=2.3

# # Bounds for interior 3
# x7=2.4
# x8=3.1

# # Define interior 1 and 2
# interior1 = p.add_interior_subdomain("interior1", geom=geo, criteria=And(x > x1, x < x2), params={t:(0,2*L)})
# interior2 = p.add_interior_subdomain("interior2", geom=geo, criteria=And(x > x3, x < x4), params={t:(0,2*L)})
# interior3 = p.add_interior_subdomain("interior3", geom=geo, criteria=And(x > x5, x < x6), params={t:(0,2*L)})
# interior4 = p.add_interior_subdomain("interior4", geom=geo, criteria=And(x > x7, x < x8), params={t:(0,2*L)})

# Interfaces
x_int1 = 1.045
x_int2 = 2.095
b1 = 0.65
b2 = 0.9
b3 = 1.4
b4 = 1.7
b5 = 2.2
b6 = 2.7

band1 = p.Line1D("band1", b1, b2)
band2 = p.Line1D("band2", b3, b4)
band3 = p.Line1D("band3", b5, b6)
diff_temp1 = p.GeometryDifference("diff_temp1", geo, band1)
diff_temp2 = p.GeometryDifference("diff_temp2", diff_temp1, band2)
diff = p.GeometryDifference("diff", diff_temp2, band3)
interior_layers = p.add_interior_subdomain("layers", geom=diff, params={t:(0, 2*L)})

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

# Velocity constraints
vel_space_invariant = Eq(vel.diff(x, 1), 0)
p.add_constraint("vel_space_invariant", enforce(equation=vel_space_invariant, on_domain=interior_layers))

# 5. Data constraints
hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel-4layers/data_fd_resampled-subx10-subt100.hdf5"
# hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-inv-1d-vel-4layers/data_fd_resampled-5.hdf5"
p.add_data_constraint(name="observed_data", model=u, data_fname=hdf_fname)