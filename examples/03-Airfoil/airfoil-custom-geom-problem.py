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

[x, y, rot], [u, pp, v] = p.add_neural_network(
    name="NN", inputs=["x", "y", "rot"], outputs=["u", "p", "v"]
)

# geometry
import numpy as np


params = {rot: (0, -np.pi / 6)}
# params = {rot: float(0.0)}

channel_length = 15.0 / 2
channel_height = 10.0 / 2
a = 0.3

channel_rect = p.Rectangle(
    "channel_rect",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
)
channel = p.Channel2D(
    "channel",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
)

import os

module_path = os.getcwd()
module_name = "CustomAirfoilGeom"
domain_geom = p.CustomGeometry(
    "geom",
    module_path,
    module_name,
    "AirfoilInChannel",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
)

# domain_geom = p.GeometryDifference("dg", channel, tri)

interior = p.add_interior_subdomain(
    "interior", geom=domain_geom, compute_sdf_derivatives=True, params=params
)
top_bot = p.add_boundary_subdomain(
    "top_bot", geom=channel, criteria=Eq(Abs(y), channel_height / 2), params=params
)
inlet = p.add_boundary_subdomain(
    "inlet", geom=channel_rect, criteria=Eq(x, -channel_length * a), params=params
)

outlet = p.add_boundary_subdomain(
    "outlet", geom=channel_rect, criteria=Eq(x, channel_length * (1 - a)), params=params
)

airfoil_bdry = p.add_boundary_subdomain("airfoil_bdry", geom=domain_geom, params=params)

lower_rec = p.Rectangle(
    "lower_rec",
    (-channel_length * a, -channel_height / 2),
    (-1, channel_height / 2),
    params=params,
)
lower_rec = p.add_boundary_subdomain(
    "lower_rec", geom=lower_rec, criteria=Eq(x, -channel_length * a) | Eq(x, -1)
)

lower_rec2 = p.Rectangle(
    "lower_rec2",
    (-channel_length * a, -channel_height / 2),
    (2, channel_height / 2),
    params=params,
)
lower_rec2 = p.add_boundary_subdomain(
    "lower_rec2",
    geom=lower_rec2,
    criteria=Eq(x, -channel_length * a) | Eq(x, 2),
    params=params,
)
inlet_outlet = p.Rectangle(
    "inlet_outlet",
    (-channel_length * a, -channel_height / 2),
    (channel_length * (1 - a), channel_height / 2),
    params=params,
)
inlet_outlet = p.add_boundary_subdomain(
    "inlet_outlet",
    geom=inlet_outlet,
    criteria=Eq(x, -channel_length * a) | Eq(x, channel_length * (1 - a)),
    params=params,
)

import sympy as sp

normal_x = sp.Symbol("normal_x")
normal_y = sp.Symbol("normal_y")

p.add_constraint(
    f"no_flux_1",
    enforce(
        equation=Eq(Integral((normal_x * u + normal_y * v), x, y), 0),
        on_domain=lower_rec,
    ),
)
p.add_constraint(
    f"no_flux_2",
    enforce(
        equation=Eq(Integral((normal_x * u + normal_y * v), x, y), 0),
        on_domain=lower_rec2,
    ),
)
p.add_constraint(
    f"noflux_inlet_outlet",
    enforce(
        equation=Eq(Integral((normal_x * u + normal_y * v), x, y), 0),
        on_domain=inlet_outlet,
    ),
)


sdf = sympy.Function("sdf")(x, y, rot)
from sympy import sqrt, Min, Abs

# Zero Equation
nu = (
    sqrt((u.diff(y) + v.diff(x)) ** 2 + 2 * u.diff(x) ** 2 + 2 * v.diff(y) ** 2)
    * Min(0.045, 0.419 * sdf) ** 2
    + 6.25e-6
)
nu = p.add_submodel("nu", nu)

# N-S Momentum equations
m_x = (
    -1.0 * nu * u.diff(x).diff(x)
    - 1.0 * nu * u.diff(y).diff(y)
    + 1.0 * u * u.diff(x)
    + 1.0 * v * u.diff(y)
    - 1.0 * nu.diff(x) * u.diff(x)
    - 1.0 * nu.diff(y) * u.diff(y)
    + pp.diff(x)
)
momentum_x = Eq(m_x, 0)

m_y = (
    -1.0 * nu * v.diff(x).diff(x)
    - 1.0 * nu * v.diff(y).diff(y)
    + 1.0 * u * v.diff(x)
    + 1.0 * v * v.diff(y)
    - 1.0 * nu.diff(x) * v.diff(x)
    - 1.0 * nu.diff(y) * v.diff(y)
    + pp.diff(y)
)
momentum_y = Eq(m_y, 0)

continuity_eq = Eq(u.diff(x) + v.diff(y), 0)

p.add_constraint(
    "interior_continuity", enforce(equation=continuity_eq, on_domain=interior)
)
p.add_constraint(
    "interior_momentum_x", enforce(equation=momentum_x, on_domain=interior)
)
p.add_constraint(
    "interior_momentum_y", enforce(equation=momentum_y, on_domain=interior)
)

p.add_constraint("airfoil_bdry_u", enforce(equation=Eq(u, 0), on_domain=airfoil_bdry))
p.add_constraint("airfoil_bdry_v", enforce(equation=Eq(v, 0), on_domain=airfoil_bdry))

p.add_constraint("top_bot_u", enforce(equation=Eq(u, 1), on_domain=top_bot))
p.add_constraint("top_bot_v", enforce(equation=Eq(v, 0), on_domain=top_bot))

p.add_constraint("inlet_u", enforce(equation=Eq(u, 1), on_domain=inlet))
p.add_constraint("inlet_v", enforce(equation=Eq(v, 0), on_domain=inlet))

p.add_constraint("outlet_p", enforce(equation=Eq(pp, 0), on_domain=outlet))
