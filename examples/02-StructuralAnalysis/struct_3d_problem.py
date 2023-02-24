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

[x, y, z, R], sOuts = p.add_neural_network(
    name="NN", inputs=["x", "y", "z", "R"], outputs=["u", "v", "w"]
)

u, v, w = sOuts
params = {R: (10, 20)}

e_xx = p.add_submodel("epsilon_xx", u.diff(x))
e_yy = p.add_submodel("epsilon_yy", v.diff(y))
e_zz = p.add_submodel("epsilon_zz", w.diff(z))

e_xy = p.add_submodel("epsilon_xy", 0.50 * u.diff(y) + 0.50 * v.diff(x))
e_yz = p.add_submodel("epsilon_yz", 0.50 * v.diff(z) + 0.50 * w.diff(y))
e_zx = p.add_submodel("epsilon_zx", 0.50 * w.diff(x) + 0.50 * u.diff(z))


E = 1  # 2e9
nu = 0.3
C = E / (1 + nu**2)
mu = E / ((1 + nu) * 2)

C11 = (E * (1 - nu)) / ((1 + nu) * (1 - 2 * nu))
C12 = (E * nu) / ((1 + nu) * (1 - 2 * nu))

sigma_xx = p.add_submodel("sigma_xx", C11 * e_xx + C12 * e_yy + C12 * e_zz)
sigma_yy = p.add_submodel("sigma_yy", C12 * e_xx + C11 * e_yy + C12 * e_zz)
sigma_zz = p.add_submodel("sigma_zz", C12 * e_xx + C12 * e_yy + C11 * e_zz)
sigma_yz = p.add_submodel("sigma_yz", mu * 2 * e_yz)
sigma_zx = p.add_submodel("sigma_zx", mu * 2 * e_zx)
sigma_xy = p.add_submodel("sigma_xy", mu * 2 * e_xy)

# refer to : https://en.wikiversity.org/wiki/Elasticity/Constitutive_relations

from sympy import Symbol

n_x, n_y, n_z = Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")

traction_x = n_x * sigma_xx + n_y * sigma_xy + n_z * sigma_zx
traction_y = n_x * sigma_xy + n_y * sigma_yy + n_z * sigma_yz
traction_z = n_x * sigma_zx + n_y * sigma_yz + n_z * sigma_zz

# Geometry
r_sz = 200 * 1.0
g_inner = p.Sphere("inner_c", (0, 0, 0), R, params=params)
g_outer = p.Box("outer_c", (-r_sz, -r_sz, -r_sz), (r_sz, r_sz, r_sz))

geom = p.GeometryDifference("geom", g_outer, g_inner)

pressure = p.add_submodel(
    "traction_dot_normal", n_x * (traction_x) + n_y * (traction_y) + n_z * (traction_z)
)

# ----- Dirichlet
rect = p.Rectangle("rect", (-r_sz, -r_sz), (r_sz, r_sz))
bdry = p.add_boundary_subdomain(
    "bdry", geom=rect, params={z: float(r_sz), **params}, criteria=Eq(y, r_sz)
)
p.add_constraint("bdry_w0", enforce(equation=Eq(w, 0), on_domain=bdry))
# ------

inner_solid = p.add_interior_subdomain("inner_solid", geom=geom, params=params)

equilibrium_x = Eq(sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_zx.diff(z), 0)
equilibrium_y = Eq(sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z), 0)
equilibrium_z = Eq(sigma_zx.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z), 0)

p.add_constraint(
    "equilibrium_x", enforce(equation=equilibrium_x, on_domain=inner_solid)
)
p.add_constraint(
    "equilibrium_y", enforce(equation=equilibrium_y, on_domain=inner_solid)
)
p.add_constraint(
    "equilibrium_z", enforce(equation=equilibrium_z, on_domain=inner_solid)
)

inner_mesh_bdry = p.add_boundary_subdomain("inner_mesh", geom=g_inner, params=params)
# p.add_constraint("inner_traction", enforce(equation=Eq(pressure, 0),
#                                             on_domain=inner_mesh_bdry))

p.add_constraint(
    "inner_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=inner_mesh_bdry)
)

p.add_constraint(
    "inner_traction_y", enforce(equation=Eq(traction_y, 0), on_domain=inner_mesh_bdry)
)
p.add_constraint(
    "inner_traction_z", enforce(equation=Eq(traction_z, 0), on_domain=inner_mesh_bdry)
)

top_bdry = p.add_boundary_subdomain(
    "top_bdry", geom=g_outer, criteria=Eq(y, r_sz), params=params
)


p.add_constraint(
    "top_traction_y", enforce(equation=Eq(traction_y, 1), on_domain=top_bdry)
)
p.add_constraint(
    "top_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=top_bdry)
)
p.add_constraint(
    "top_traction_z", enforce(equation=Eq(traction_z, 0), on_domain=top_bdry)
)

bottom_bdry = p.add_boundary_subdomain(
    "bottom_bdry", geom=g_outer, criteria=Eq(y, -r_sz), params=params
)
p.add_constraint(
    "bottom_traction_y", enforce(equation=Eq(traction_y, -1), on_domain=bottom_bdry)
)
p.add_constraint(
    "bottom_traction_x", enforce(equation=Eq(traction_x, 0), on_domain=bottom_bdry)
)
p.add_constraint(
    "bottom_traction_z", enforce(equation=Eq(traction_z, 0), on_domain=bottom_bdry)
)

left_right_bdry = p.add_boundary_subdomain(
    "left_right_bdry",
    geom=g_outer,
    criteria=Or(Eq(x, r_sz), Eq(x, -r_sz), Eq(z, r_sz), Eq(z, -r_sz)),
    params=params,
)
p.add_constraint(
    "left_right_traction_x",
    enforce(equation=Eq(traction_x, 0), on_domain=left_right_bdry),
)
p.add_constraint(
    "left_right_traction_y",
    enforce(equation=Eq(traction_y, 0), on_domain=left_right_bdry),
)
p.add_constraint(
    "left_right_traction_z",
    enforce(equation=Eq(traction_z, 0), on_domain=left_right_bdry),
)
