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

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.primitives_1d import Line1D, Point1D
from modulus.geometry.primitives_2d import Rectangle, Circle, Polygon, Line, Channel2D
from modulus.geometry.primitives_3d import Box, Sphere, Cylinder

from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
    IntegralBoundaryConstraint
)

from modulus.domain.inferencer import PointwiseInferencer

from modulus.geometry.parameterization import Parameterization, Parameter

from modulus.key import Key
from modulus.node import Node

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig

import numpy as np
import os

from modulus.geometry.tessellation import Tessellation
###############
from sympy import Symbol, Eq, Or, And, Function
import sympy

import torch


class PDParam:
    def __init__(self, geom, param_ranges=None):
        self._geom = geom
        self._params = param_ranges

    def sample_interior(self, *args, **kwargs):
        d = self._geom.sample_interior(*args, **kwargs)
        shp = d["x"].shape
        nr = shp[0]

        for param, rng in self._params.items():
            data = np.random.rand(nr).reshape(shp)
            delta = rng[1] - rng[0]
            d[param] = data * delta + rng[0]

        return d

    def sample_boundary(self, *args, **kwargs):
        d = self._geom.sample_boundary(*args, **kwargs)

        for k, v in d.items():
            d[k] = np.vstack([v, v, v])

        shp = d["x"].shape
        nr = shp[0]

        for param, rng in self._params.items():
            data = np.random.rand(nr).reshape(shp)
            delta = rng[1] - rng[0]
            d[param] = data * delta + rng[0]

        return d

{% if no_modulus_main %}
from modulus.hydra.utils import compose
cfg = compose(config_path="conf", config_name="config")
cfg.network_dir = "{{conf_path}}/outputs"
def run(cfg=cfg) -> None:
{% else %}
@modulus.main(config_path="conf", config_name="config")
def run(cfg) -> None:
{% endif %}