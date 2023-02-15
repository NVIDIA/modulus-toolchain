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
from modulus.geometry.primitives_1d import Line1D
from modulus.geometry.primitives_2d import Rectangle, Circle

from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)

from modulus.domain.inferencer import PointwiseInferencer

from modulus.key import Key
from modulus.node import Node

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig

import numpy as np

###############
from sympy import Symbol, Eq, Or, And, Function

allowed_NN_types = {}  # {"fully_connected": None}

global _cfg

import torch

torch.set_num_threads(1)


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


class InteriorSubdomain:
    def __init__(self, geom, criteria):
        self._geom = geom
        self._criteria = criteria

    def sample(self, *args, **kwargs):
        self._geom.sample_interior(*args, **kwargs)


class BoundarySubdomain:
    def __init__(self, geom, criteria):
        self._geom = geom
        self._criteria = criteria

    def sample(self, *args, **kwargs):
        self._geom.sample_boundary(*args, **kwargs)

    def get_variables(self):
        d = self._geom.sample_boundary(1)
        return {v: Symbol(v) for v in d.keys()}


def enforce(equation=None, on_domain=None):
    return {"equation": equation, "on_domain": on_domain}


# @modulus.main(config_path="conf", config_name="config")
# def _run(cfg: ModulusConfig) -> None:
#     global _cfg
#     _cfg = cfg

#     global allowed_NN_types
#     allowed_NN_types = {"fully_connected": cfg.arch.fully_connected}
#     cfg["save_filetypes"] = "np"
#     cfg["network_dir"] = "outputs/problem"
#     print(to_yaml(cfg))


# _run()


class Problem:
    def __init__(self, name="problem", cfg=None):

        global _cfg
        _cfg = cfg
        global allowed_NN_types
        allowed_NN_types = {"fully_connected": cfg.arch.fully_connected}

        self._problem_name = name
        self._vars = {}  # str -> sympy.Symbol
        self._nns = {}
        self._nn_outs = set()

        self.domain = Domain()
        self._nodes = []

        self._model = []
        self._submodels = {}
        self._constraints = {}
        self._data_constraints = {}

        ##########
        # import omegaconf

        # cfg = omegaconf.OmegaConf.load("config.yaml")

        # import os

        # global _cfg
        # _cfg = cfg

        # global allowed_NN_types
        # allowed_NN_types = {"fully_connected": cfg["arch"]["fully_connected"]}
        # cfg["save_filetypes"] = "np"
        # cfg["network_dir"] = os.path.join("outputs", self._problem_name)
        # print(to_yaml(cfg))

    def get_infer_fn(self):
        # load NNs
        slv = Solver(_cfg, self.domain)
        slv._eval()

        # create Inference object
        invals = {str(v): np.array([0]).reshape(-1, 1) for v in self._vars.keys()}
        inferencer = PointwiseInferencer(
            invar=invals,
            output_names=[submodel for submodel in self._submodels],
            nodes=self._nodes,
            batch_size=256 * 4 * 4 * 4,
        )
        self.domain.add_inferencer(inferencer)
        # create inference function
        def infer_fn(*args, **kargs):
            from modulus.domain.constraint import Constraint

            invals = {str(v): kargs[v].reshape(-1, 1) for v in self._vars.keys()}

            invar0 = invals
            invar = Constraint._set_device(
                invar0, requires_grad=False, device=inferencer.device
            )
            pred_outvar = inferencer.forward(invar)

            result = {}
            for submodel in self._submodels:
                ret = pred_outvar[submodel].cpu().detach().numpy()
                ret_val = np.array([v for v in ret[:, 0]])
                result[submodel] = ret_val

            # now build the main model
            model = self._model
            main_result = ret_val.copy().reshape(-1, 1)
            invars, invals = [], []
            for varn, varval in invar0.items():
                invars.append(Symbol(varn))
                invals.append(varval)

            for smodel in model["conditions"]:
                func = smodel["func"]
                cond = smodel["on"]

                submodel_result = result[str(func)].reshape(-1, 1)
                from sympy import lambdify

                sel = lambdify(invars, cond)(*invals)
                main_result[sel] = submodel_result[sel]

            result[model["name"]] = main_result

            return result

        return infer_fn

    def train(self):
        self.pprint()

        slv = Solver(_cfg, self.domain)
        # start solver
        slv.solve()

    def set_model(self, name, func_on_list):
        self._model = {"name": name, "conditions": func_on_list}

    def add_submodel(self, name, expr):
        self._submodels[name] = expr
        self._nodes += [Node.from_sympy(expr, str(name))]

        return Function(name)(*self._vars.values())

    def add_neural_network(self, name="", nn_type="", inputs=[], outputs=[]):
        assert len(name) > 0, "must provide a network name"

        assert name not in self._nns, f"{name} NN already exists"

        assert len(inputs) > 0, "must include a non-empty list of input vars"
        assert len(outputs) > 0, "must include a non-empty list of output vars"

        assert nn_type in allowed_NN_types, f"{nn_type} not in list {allowed_NN_types}"

        if len(self._vars) > 0:
            assert set(self._vars.keys()) == set(
                inputs
            ), "New NN definition does not match the input variables already defined"

        self._vars = {v: Symbol(v) for v in inputs}

        ocheck = self._nn_outs.intersection(set(outputs))
        assert len(ocheck) == 0, f"Redefining output variables {ocheck}"
        self._nn_outs = self._nn_outs.union(set(outputs))

        net = instantiate_arch(
            input_keys=[Key(v) for v in inputs],
            output_keys=[Key(v) for v in outputs],
            cfg=allowed_NN_types[nn_type],
        )
        self._nodes += [net.make_node(name=name, jit=_cfg.jit)]

        self._nns[name] = {
            "net": net,
            "inputs": inputs,
            "outputs": outputs,
            "nn_type": nn_type,
        }

        invars = [self._vars[v] for v in inputs]
        outvars = [Function(v)(*invars) for v in outputs]
        return invars, outvars

    def add_interior_subdomain(self, geom=None, params=None, criteria=None):
        # geom is required, the other two are optional
        assert geom is not None, "geom argument is required"

        if params is not None:
            ps = {str(k): v for k, v in params.items()}
            geom = PDParam(geom, param_ranges=ps)

        return InteriorSubdomain(geom, criteria)

    def add_boundary_subdomain(self, geom=None, params=None, criteria=None):
        # geom is required, the other two are optional
        assert geom is not None, "geom argument is required"

        if params is not None:
            ps = {str(k): v for k, v in params.items()}
            geom = PDParam(geom, param_ranges=ps)

        return BoundarySubdomain(geom, criteria)

    def add_data_constraint(self, name=None, model=None, data=None):
        assert (
            str(model) in self._submodels or str(model) in self._nn_outs
        ), f"undefined model {model}"
        assert data is not None, "data is required"

        assert str(model) in data, "model data not found"

        missingvars = "data does not include all variables"
        missingvars += f"\nshould have: {set(self._vars.keys())}"
        assert (
            len(set(self._vars.keys()).difference(set(data.keys()))) == 0
        ), missingvars

        assert name not in self._data_constraints, "data constraint already defined"

        # now add the constraint
        self._data_constraints[name] = {"model": model, "data": data}

        data_len = len(data[str(model)])
        data_constraint = PointwiseConstraint.from_numpy(
            nodes=self._nodes,
            invar={v: data[v] for v in self._vars.keys()},
            outvar={str(model): data[str(model)]},
            batch_size=min(data_len, 1024),
        )
        self.domain.add_constraint(data_constraint, name)

    def set_constraints(self, cdict):

        self._constraints = cdict

        variables_in_constraints = []

        # create the nodes first
        for c_name, c_v in cdict.items():
            eq = c_v["equation"]
            eq0 = str(eq.lhs - eq.rhs)
            for v in self._nn_outs:
                if v in eq0:
                    variables_in_constraints.append(v)
            self._nodes += [Node.from_sympy(eq.lhs - eq.rhs, str(c_name))]
        nodes = self._nodes

        for c_name, c_v in cdict.items():
            subdomain = c_v["on_domain"]
            print(f"instantiating constraint {c_name} on {subdomain}")
            if isinstance(subdomain, InteriorSubdomain):
                import time

                t0 = time.time()
                pic = PointwiseInteriorConstraint(
                    nodes=nodes,
                    geometry=subdomain._geom,
                    criteria=subdomain._criteria,
                    batch_size=_cfg.batch_size.interior,
                    outvar={c_name: 0},
                )
                self.domain.add_constraint(pic, c_name)
                t1 = time.time()
                print(f"PointwiseInteriorConstraint: elapsed {t1-t0:.3f} s")

            if isinstance(subdomain, BoundarySubdomain):
                pbc = PointwiseBoundaryConstraint(
                    nodes=nodes,
                    geometry=subdomain._geom,
                    criteria=subdomain._criteria,
                    batch_size=1024,
                    outvar={c_name: 0},
                )
                self.domain.add_constraint(pbc, c_name)

        # variables_in_constraints = set(variables_in_constraints)
        # unusedNNs = self._nn_outs.difference((variables_in_constraints))
        # assert len(unusedNNs) == 0, f"Unused NNs {unusedNNs}"

    def pprint_constraints(self):
        # print("=" * 80)
        print("Constraints")
        print("-" * 80)
        for c_name, c_v in self._constraints.items():
            eq = c_v["equation"]
            eq0 = str(eq.lhs - eq.rhs)
            print("| ", c_name, ":", eq.lhs - eq.rhs)

        print("|\nData --")
        for dc_name in self._data_constraints:
            print(f"|  {dc_name}")

        print("-" * 80, "\n")

    def pprint_models(self):
        # print("=" * 80)
        print("Models")
        print("-" * 80)
        for mname, model in self._submodels.items():
            print("| ", mname, ":", model)

        model = self._model
        print("|---")
        if len(model) > 0:
            vars_str = ",".join(self._vars.keys())
            print(f"| {model['name']}({vars_str}) = ")
            for cond in model["conditions"]:
                print(f"|", " " * 10, f"{cond['func']}  if  {cond['on']}")
        print("-" * 80, "\n")

    def pprint_nns(self):
        # print("=" * 80)
        print("Neural Networks")
        print("-" * 80)
        fmt = lambda l: [Symbol(v) for v in l]
        for nnn, nn in self._nns.items():
            spref = f"{nnn} = {nn['nn_type']}("
            s = spref + f"inputs={fmt(nn['inputs'])},"
            s += "\n" + " " * len(spref) + f"outputs={fmt(nn['outputs'])})"

            nns = s
            nns = "\n".join(["|  " + line for line in nns.split("\n")])
            print(nns)
        print("-" * 80, "\n")

    def pprint(self):
        print()
        self.pprint_nns()
        self.pprint_models()
        self.pprint_constraints()
