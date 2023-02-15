from cfg import *
import numpy as np
from sympy import exp, sin, cos, DiracDelta

##########################################################
####################### Neural networks ##################
##########################################################
# Create NN to predict pressure (x,y,t) -> p
[x, y, t], [pressure] = p.add_neural_network(name="pressure", inputs=["x", "y", "t"], outputs=["pressure"])

##########################################################
######################### Constraints ####################
##########################################################
# 3. Data constraints (snapshots taken from t in [0.25, 2.0]
hdf_fname="/mount/workspace_test/temp/mtc-repo/examples/we-fwd-2d-cv-snapshots/data_fd_wave-2d-snapshots-n10-ti-0.2-tf-0.6.hdf5"
p.add_data_constraint(name="observed_data", model=pressure, data_fname=hdf_fname)



