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

# Suggested structure
#
# 1. Define problem variables and unknown functions; e.g.,
#    [x, y], [u] = p.add_neural_network(name="NN", inputs=["x", "y"], outputs=["u"])
#
# 2. Define geometries; e.g.,
#    geo = p.Line1D("geom", 0,1)
#
# 3. Define sub-domains (geometries plus non-geometric variables goe here); e.g.
#
# interior = p.add_interior_subdomain("interior",
#                                     geom=geo,
#                                     params={t:(0,2*L)})
# boundary = p.add_boundary_subdomain("boundary",
#                                     geom=geo,
#                                     params={t:(0,2*L)})
#
# 4. (Optionally) Sub-models; e.g,
#    g_air = p.add_submodel("g_air", u_air * zf + airT)
#
#
# 5. Define constraints; e.g.,
#
#     wave_eq = Eq(u.diff(t, 2), (c**2 * u.diff(x)).diff(x))
#     p.add_constraint("wave_equation", enforce(equation=wave_eq, on_domain=interior))
#     p.add_data_constraint(name="calculated", model=u, data_fname=hdf_fname)
#
#
# 6. (optionally) Define piecewise-models (collecting submodels)
#
# p.set_model(
#     "T",
#     [
#         {"func": g_metal, "on": And(y > mny, y < mxy)},
#         {"func": g_air, "on": ~And(y > mny, y < mxy)},
#     ],
# )
