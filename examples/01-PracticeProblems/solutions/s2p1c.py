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

[x, y], [u, v, pp] = p.add_neural_network(
    name="NN", inputs=["x", "y"], outputs=["u", "v", "p"]
)

w, h = 2, 1
r1 = p.Rectangle("r1", (0, 0), (w, h))
ch1 = p.Channel2D("ch1", (0, 0), (w, h))

inlet = p.add_boundary_subdomain("inlet", geom=r1, criteria=Eq(x, 0))
outlet = p.add_boundary_subdomain("outlet", geom=r1, criteria=Eq(x, w))

r_interior = p.add_interior_subdomain("r_interior", geom=r1)
ch_interior = p.add_interior_subdomain("ch_interior", geom=ch1)

noslip = p.add_boundary_subdomain("no_slip", geom=ch1)

# p.add_constraint("c1", enforce(equation=Eq(x,0), on_domain=r_interior))
p.add_constraint("inlet_u", enforce(equation=Eq(u, 1), on_domain=inlet))
p.add_constraint("inlet_v", enforce(equation=Eq(v, 0), on_domain=inlet))

p.add_constraint("outlet_p", enforce(equation=Eq(pp, 0), on_domain=outlet))

p.add_constraint("noslip_u", enforce(equation=Eq(u, 0), on_domain=noslip))
p.add_constraint("noslip_v", enforce(equation=Eq(v, 0), on_domain=noslip))

nu = 0.02
p.add_constraint(
    "continuity", enforce(equation=Eq(u.diff(x) + v.diff(y), 0), on_domain=ch_interior)
)

p.add_constraint(
    "momentum_x",
    enforce(
        equation=Eq(
            u * u.diff(x) + v.diff(y) + pp.diff(x), nu * (u.diff(x, x) + u.diff(y, y))
        ),
        on_domain=ch_interior,
    ),
)

p.add_constraint(
    "momentum_y",
    enforce(
        equation=Eq(
            u * u.diff(x) + v.diff(y) + pp.diff(y), nu * (v.diff(x, x) + v.diff(y, y))
        ),
        on_domain=ch_interior,
    ),
)

# Adding mass balance
import sympy as sp

nx = sp.Symbol("normal_x")
ny = sp.Symbol("normal_y")

inlet_outlet = p.add_boundary_subdomain(
    "inlet_outlet", geom=r1, criteria=Eq(x, 0) | Eq(x, w)
)
p.add_constraint(
    "mass_balance",
    enforce(equation=Eq(Integral(nx * u + ny * v, x, y), 0), on_domain=inlet_outlet),
)
