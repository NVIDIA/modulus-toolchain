# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cfg import *

[x, y], sOuts = p.add_neural_network(name="NN", inputs=["x", "y"], outputs=["u", "v"])

u, v = sOuts

e_xx = p.add_submodel("epsilon_xx", u.diff(x))
e_yy = p.add_submodel("epsilon_yy", v.diff(y))
e_xy = p.add_submodel("epsilon_xy", 0.50 * u.diff(y) + 0.50 * v.diff(x))

# https://www.mathworks.com/matlabcentral/fileexchange/70183-elastic-constitutive-law-plane-stress

E = 1  # 2e9
nu = 0.3
C = E / (1 + nu**2)

sigma_xx = p.add_submodel("sigma_xx", C * (e_xx + nu * e_yy))
sigma_yy = p.add_submodel("sigma_yy", C * (nu * e_xx + e_yy))
sigma_xy = p.add_submodel("sigma_xy", C * (1 - nu) * e_xy)


from sympy import Symbol

n_x, n_y, n_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")

traction_x = n_x * sigma_xx + n_y * sigma_xy
traction_y = n_x * sigma_xy + n_y * sigma_yy

# Geometry
r_sz = 200
g_inner = p.Circle("inner_c", (0, 0), 10)
g_outer = p.Rectangle("outer_c", (-r_sz, -r_sz), (r_sz, r_sz))

geom = p.GeometryDifference("geom", g_outer, g_inner)

pressure = p.add_submodel(
    "traction_dot_normal", n_x * (traction_x) + n_y * (traction_y)
)


inner_solid = p.add_interior_subdomain("inner_solid", geom=geom)

equilibrium_x = Eq(sigma_xx.diff(x) + sigma_xy.diff(y), 0)
equilibrium_y = Eq(sigma_xy.diff(x) + sigma_yy.diff(y), 0)

p.add_constraint(
    "equilibrium_x", enforce(equation=equilibrium_x, on_domain=inner_solid)
)
p.add_constraint(
    "equilibrium_y", enforce(equation=equilibrium_y, on_domain=inner_solid)
)

inner_mesh_bdry = p.add_boundary_subdomain("inner_mesh", geom=g_inner)
# p.add_constraint("inner_traction", enforce(equation=Eq(pressure, 0),
#                                             on_domain=inner_mesh_bdry))

p.add_constraint(
    "inner_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=inner_mesh_bdry)
)

p.add_constraint(
    "inner_traction_y", enforce(equation=Eq(traction_y, 0), on_domain=inner_mesh_bdry)
)

top_bdry = p.add_boundary_subdomain("top_bdry", geom=g_outer, criteria=Eq(y, r_sz))
p.add_constraint(
    "top_traction_y", enforce(equation=Eq(traction_y, 1), on_domain=top_bdry)
)
p.add_constraint(
    "top_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=top_bdry)
)

bottom_bdry = p.add_boundary_subdomain(
    "bottom_bdry", geom=g_outer, criteria=Eq(y, -r_sz)
)
p.add_constraint(
    "bottom_traction_y", enforce(equation=Eq(traction_y, -1), on_domain=bottom_bdry)
)
p.add_constraint(
    "bottom_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=bottom_bdry)
)

left_right_bdry = p.add_boundary_subdomain(
    "left_right_bdry", geom=g_outer, criteria=Or(Eq(x, r_sz), Eq(x, -r_sz))
)
p.add_constraint(
    "left_right_traction_x",
    enforce(equation=Eq(traction_x, 0), on_domain=left_right_bdry),
)
p.add_constraint(
    "left_right_traction_y",
    enforce(equation=Eq(traction_y, 0), on_domain=left_right_bdry),
)
