from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D, Point1D
from modulus.sym.geometry.primitives_2d import Rectangle, Circle, Polygon, Line, Channel2D
from modulus.sym.geometry.primitives_3d import Box, Sphere, Cylinder

from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
    IntegralBoundaryConstraint
)

from modulus.sym.domain.inferencer import PointwiseInferencer

from modulus.sym.geometry.parameterization import Parameterization, Parameter

from modulus.sym.key import Key
from modulus.sym.node import Node

import modulus
from modulus.sym.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.sym.hydra.config import ModulusConfig

import numpy as np
import os

# from modulus.sym.geometry.tessellation import Tessellation
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
from modulus.sym.hydra.utils import compose
cfg = compose(config_path="conf", config_name="config")
cfg.network_dir = "{{conf_path}}/outputs"
def run(cfg=cfg) -> None:
{% else %}
@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg) -> None:
{% endif %}