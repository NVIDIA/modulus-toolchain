from sympy import Symbol, Function, Or, And, Eq, Abs, Integral, expand
import sympy

import os, sys


def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def load_yaml(filename):
    """Loads a YAML file using a path relative to where this module resides"""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return yaml.safe_load(f)


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
        d = {
            "normal_x": 1,
            "normal_y": 1,
            "normal_z": 1,
        }  # self._geom.sample_boundary(1)
        return {v: Symbol(v) for v in d.keys()}


def enforce(equation=None, on_domain=None):
    return {"equation": equation, "on_domain": on_domain}


class PINNProblem:
    def __init__(self, name="PINN problem", cfg=None):

        self._problem_name = name
        self._vars = {}  # str -> sympy.Symbol
        self._nns = {}
        self._nn_outs = set()

        self.domain = None  # Domain()
        self._nodes = []

        self._model = []
        self._submodels = {}
        self._constraints = {}
        self._data_constraints = {}

        self._geom = {}
        self._interior_subdomains = {}
        self._boundary_subdomains = {}

        self._no_modulus_main = False

    def load_conf(self):
        import yaml

        with open(os.path.join("conf", "config.yaml")) as f:
            conf = yaml.safe_load(f)
        return conf

    def save_conf(self, conf):
        import yaml

        with open(os.path.join("conf", "config.yaml"), "w") as f:
            yaml.safe_dump(conf, f)

    def to_hdf(self, hdf_fname, data):
        import h5py

        with h5py.File(hdf_fname, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)

    def init_config(self, only1storder=False, max_steps=1000):
        if only1storder:
            self.compile_to_firstorder()

        conf = self.load_conf()

        def mkdeqc(dc):
            d = {k: str(v) for k, v in dc.items()}
            if dc["on_domain"] in self._interior_subdomains:
                d["domain_type"] = "interior"
            else:
                d["domain_type"] = "boundary"

            d["batch_size"] = 1000

            return d

        def make_training_stages():
            d = {
                "stage-dag": [],
                "stages": {
                    "stage1": {
                        "description": "Default stage",
                        "data": {},
                    },
                },
            }
            return d

        # if "modulus_project" not in conf:
        nn_type = "fully_connected"
        # nn_type = "fourier_net"
        nn = load_yaml(os.path.join("..", "mpc", "mpc", "config_types.yaml"))["arch"][
            nn_type
        ]
        conf["modulus_project"] = {
            "project_name": self._problem_name,
            "submodels": {k: str(v) for k, v in self._submodels.items()},
            "equation_constraints": {
                k: mkdeqc(v) for k, v in self._constraints.items()
            },
            "neural_networks": {
                nn_name: {
                    "nn_type": nn_type,
                    "_target_": nn["_target_"],
                    **{k: v["default"] for k, v in nn.items() if k != "_target_"},
                }
                for nn_name in self._nns
            },
        }

        conf["modulus_project"]["training"] = make_training_stages()

        self.save_conf(conf)

        from mtc.config_utils import customize_schema, config2dictV2

        conf["modulus_project"]["training"]["stages"]["stage1"]["data"] = config2dictV2(
            customize_schema()
        )

        s1data = conf["modulus_project"]["training"]["stages"]["stage1"]["data"]
        s1data["training"]["max_steps"] = max_steps

        self.save_conf(conf)

    def get_variables(self, name):
        assert (
            name in self._boundary_subdomains
        ), "variable must be a boundary subdomain"
        d = {
            "normal_x": 1,
            "normal_y": 1,
            "normal_z": 1,
        }  # self._geom.sample_boundary(1)
        return {v: Symbol(v) for v in d.keys()}

    def GeometryCustomWarp(self, name, code_str, func, param_list, params=None):
        self._custom_warp_code = code_str
        assert name not in self._geom
        self._geom[name] = {
            "type": "GeometryCustomWarp",
            "args": param_list,
            "func": func,
            "params": params,
        }
        return name

    def Rectangle(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Rectangle",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Box(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Box",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Line2D(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Line",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Channel2D(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Channel2D",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Polygon(self, name, line, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Polygon",
            "args": (line,),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Circle(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Circle",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Sphere(self, name, a, b, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Sphere",
            "args": (a, b),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Cylinder(self, name, center, radius, height, params=None, rotate=None):
        assert name not in self._geom
        self._geom[name] = {
            "type": "Cylinder",
            "args": (center, radius, height),
            "params": params,
            "rotate": rotate,
        }
        return name

    def Line1D(self, name, a, b, params=None):
        assert name not in self._geom
        self._geom[name] = {"type": "Line1D", "args": (a, b), "params": params}
        return name

    def Point1D(self, name, a, params=None):
        assert name not in self._geom
        self._geom[name] = {"type": "Point1D", "args": (a,), "params": params}
        return name

    def GeometryFromSTL(self, name, fname, airtight=False):
        assert name not in self._geom
        self._geom[name] = {"type": "STL", "args": (fname, airtight)}
        return name

    def GeometryDifference(self, name, g, og):
        assert g in self._geom
        assert og in self._geom

        self._geom[name] = {"type": "GeometryDifference", "g": g, "og": og}
        return name

    def GeometryUnion(self, name, g, og):
        assert g in self._geom
        assert og in self._geom

        self._geom[name] = {"type": "GeometryUnion", "g": g, "og": og}
        return name

    def GeometryIntersection(self, name, g, og):
        assert g in self._geom
        assert og in self._geom

        self._geom[name] = {"type": "GeometryIntersection", "g": g, "og": og}
        return name

    def CustomGeometry(
        self, name, module_path, module_name, class_name, *args, **kwargs
    ):

        self._geom[name] = {
            "type": "CustomGeometry",
            "module_path": module_path,
            "module_name": module_name,
            "class_name": class_name,
            "args": args,
            "kwargs": kwargs,
        }
        return name

    def compile_to_firstorder(self):
        print("compile_to_firstorder")

        # collect the unknown functions for which second and higher
        # order derivatives are used

        # collect all unknown functions across all NN definitions
        u_vars = [u for d in self._nns.values() for u in d["invars"]]
        u_fns = [u for d in self._nns.values() for u in d["outvars"]]
        # print(u_fns)

        ufn2order = {u: {v: 0 for v in u_vars} for u in u_fns}
        expanded_constraints = {}
        for cname, eq_constraint in self._constraints.items():
            eq = eq_constraint["equation"]
            eq = eq.lhs - eq.rhs

            # expand submodels in equation
            for sname, expr in self._submodels.items():
                sm_fn = Function(sname)(*self._vars.values())
                eq = sympy.simplify(eq.replace(sm_fn, expr))

            expanded_constraints[cname] = {
                "equation": Eq(eq, 0),
                "on_domain": eq_constraint["on_domain"],
            }

            for u in u_fns:
                if len(eq.find(u)) > 0:
                    for o in range(1, 5):
                        for v in ufn2order[u]:
                            if eq.find(u.diff(v, o)):
                                ufn2order[u][v] = max(o, ufn2order[u][v])

        # print("\n", expanded_constraints)
        # print()
        # print(self._submodels)
        # print()

        # for u, uo in ufn2order.items():
        #     print(u, ":", uo)

        # collect all required auxiliary unknown functions
        aux_ufns = []
        current_fn2aux_fn = {}
        aux_compat = {}  # compatibility constraints; e.g., d_dx1_u = u.diff(x)
        for u, v2o in ufn2order.items():
            for v, o in v2o.items():
                if o > 1:
                    fnname = str(type(u))
                    new_ufns = [f"d_d{v}{i+1}_{fnname}" for i in range(o - 1)]
                    aux_ufns += new_ufns

                    current_fn2aux_fn[u.diff(v)] = Function(new_ufns[0])(
                        *self._vars.values()
                    )
                    for io in range(o - 1):
                        e = u.diff(v, io + 1 + 1)
                        sname = new_ufns[io]
                        current_fn2aux_fn[e] = Function(sname)(
                            *self._vars.values()
                        ).diff(v)

                    ufn = u
                    new_u = Function(new_ufns[0])(*self._vars.values())
                    aux_compat[new_u] = Eq(ufn.diff(v), new_u)
                    for sname in new_ufns:
                        new_u = Function(sname)(*self._vars.values())
                        aux_compat[new_u.diff(v)] = Eq(ufn.diff(v), new_u)
                        ufn = new_u

        # print("\nNeed new aux u funs", aux_ufns)
        # print(current_fn2aux_fn)
        # print(aux_compat)

        # add new ufuncs
        for aufunc in aux_ufns:
            self.add_neural_network(
                name=f"{aufunc}NN", inputs=[v for v in self._vars], outputs=[aufunc]
            )

        # now rewrite the equations using the new auxiliary unknown functions
        rewritten_constraints = {}
        new_constraints = set()
        ci = 0
        handled = []
        for cname, eq_c in expanded_constraints.items():
            eq = eq_c["equation"]
            for cur_fn, new_fn in current_fn2aux_fn.items():
                if eq.find(cur_fn):
                    eq = eq.replace(cur_fn, new_fn)
                    new_constraints.add(aux_compat[new_fn])
                    rewritten_constraints[cname] = {
                        "equation": eq,
                        "on_domain": eq_c["on_domain"],
                    }

                    eqc_str = str(aux_compat[new_fn]) + str(eq_c["on_domain"])
                    if eqc_str not in handled:
                        handled += [eqc_str]
                        ci += 1
                        rewritten_constraints[cname + f"{ci}"] = {
                            "equation": aux_compat[new_fn],
                            "on_domain": eq_c["on_domain"],
                        }

        from pprint import pprint

        # print("rewritten_constraints")

        # pprint(rewritten_constraints)
        for k, v in rewritten_constraints.items():
            self._constraints[k] = v

    def compile_symbols(self):
        sl = ["", "# " + "-" * 40, "# Symbols for variables", ""]
        for v in self._vars:
            sl += [f"{v} = Symbol('{v}')"]

        return "\n".join(sl)

    def compile_neural_networks(self):
        def load_yaml(filename):
            """Loads a YAML file using a path relative to where this module resides"""
            import yaml

            with open(os.path.join(os.path.dirname(__file__), filename)) as f:
                return yaml.safe_load(f)

        # conf = load_yaml("../mpc/mpc/config_types_v2.yaml")
        conf = load_yaml("config_types_v2.yaml")

        sl = ["", "# " + "-" * 40, "# Neural Networks", ""]

        pvars = ",".join(self._vars.keys())
        for nnname, nn in self._nns.items():
            for ov in nn["outputs"]:
                sl += [f"{ov} = Function('{ov}')({pvars})"]

        sl += [""]

        for nnname, nn in self._nns.items():
            ins = ",".join(f"Key('{v}')" for v in nn["inputs"])
            outs = ",".join(f"Key('{v}')" for v in nn["outputs"])

            # nn_conf = self._conf["modulus_project"]["neural_networks"][nnname]
            nn_conf = self._conf["modulus_project"]["training"]["stages"][
                self._stage_id
            ]["data"][nnname]
            nn_target = nn_conf["_target_"].split(".")
            NNArch = nn_target[-1]
            sl += ["from " + ".".join(nn_target[:-1]) + " import " + NNArch]

            def get_hint(k):
                try:
                    return conf["arch"]["choices"][nn_conf["__selected__"]][k]["hint"]
                except:
                    return ""

            other_nn_args = "\n    ".join(
                [
                    f"{k}={v}, # {get_hint(k)}"
                    for k, v in nn_conf.items()
                    if k not in ["_target_", "nn_type"] and not k.startswith("__")
                ]
            )

            s = f"""net = {NNArch}(
    input_keys= [{ins}],
    output_keys=[{outs}],
    {other_nn_args}
)"""
            sl += [s]

            nntrainable = self._conf["modulus_project"]["training"]["stages"][
                self._stage_id
            ]["data"]["Neural Networks"][f"{nnname} Trainable"]
            sl += [
                f"nodes += [net.make_node(name='{nnname}', jit=cfg.jit, optimize={nntrainable})]"
            ]
            sl += [""]

        return "\n".join(sl)

    def compile_submodels(self):
        sl = ["", "# " + "-" * 40, "# SubModels", ""]

        for smname, expr in self._submodels.items():
            sl += [f"{smname} = sympy.sympify('{expr}')"]
            sl += [f"nodes += [Node.from_sympy({smname}, '{smname}')]", ""]

        return "\n".join(sl)

    def compile_geometries(self):
        sl = ["", "# " + "-" * 40, "# Geometries", ""]

        for gname, g in self._geom.items():
            if g["type"] == "CustomGeometry":

                # HACK -- Move module code to training/stageX
                import os

                path = os.path.join("training", self._stage_id)

                orig_module = os.path.join(g["module_path"], g["module_name"] + ".py")
                os.system(f"cp {orig_module} {path}")

                # End HACK

                sl += [f"from {g['module_name']} import {g['class_name']}"]
                sl += [f"{gname} = {g['class_name']}{g['args']}"]

                continue

            if g["type"] == "GeometryDifference":
                sl += [f"{gname}={g['g']}-{g['og']}"]
            elif g["type"] == "GeometryUnion":
                sl += [f"{gname}={g['g']}+{g['og']}"]
            elif g["type"] == "GeometryIntersection":
                sl += [f"{gname}={g['g']} & {g['og']}"]
            elif g["type"] == "STL":
                fname, airtight = g["args"]
                sl += [
                    f"{gname}=Tessellation.from_stl('{fname}', airtight={airtight}) "
                ]
            else:  # CSG like Line1D, Circle, Box, etc.
                rotate_str = ""
                if "rotate" in g and g["rotate"] is not None:
                    rotate_str = f".rotate{g['rotate']}"

                    if "params" in g and g["params"] is not None:
                        sl += [
                            "from modulus.geometry.parameterization import Parameterization, Parameter"
                        ]

                        # pstr = "Parameterization({"
                        # for k, v in g["params"].items():
                        #     pstr += f'Parameter("{k}"): {v}, '
                        # pstr += "})"

                        # # pstr = f"Parameterization({g['params']})"
                        # rotate_str = rotate_str[:-1] + f", parameterization={pstr})"

                if "params" in g and g["params"] is not None:
                    sl += [
                        "from modulus.geometry.parameterization import Parameterization"
                    ]
                    pstr = f"Parameterization({g['params']})"
                    args = (
                        "("
                        + ",".join([str(e) for e in g["args"]])
                        + f", parameterization={pstr})"
                    )
                    sl += [f"{gname}={g['type']}{args}{rotate_str}"]
                else:
                    sl += [f"{gname}={g['type']}{g['args']}{rotate_str}"]

            # if g["type"] != "GeometryDifference":
            #     sl += [f"{gname}={g['type']}{g['args']}"]
            # else:
            #     sl += [f"{gname}={g['g']}-{g['og']}"]

        return "\n".join(sl)

    def compile_interior_subdomains(self):
        sl = ["", "# " + "-" * 40, "# Interior SubDomains", ""]
        return "\n".join(sl)

    def compile_boundary_subdomains(self):
        sl = ["", "# " + "-" * 40, "# Boundary SubDomains", ""]
        return "\n".join(sl)

    def compile_equations(self):
        sl = ["", "# " + "-" * 40, "# Equations", ""]
        for cname, c in self._constraints.items():
            if isinstance(c["equation"].lhs, Integral):
                eqs = str(c["equation"].lhs.args[0])
                sl += [f"eq=sympy.sympify('{eqs}')"]
                sl += [f"{cname} = eq"]
                sl += [f"nodes += [Node.from_sympy({cname}, '{cname}')]"]

            else:
                eqs = str(c["equation"])
                sl += [f"eq=sympy.sympify('{eqs}')"]
                sl += [f"{cname} = eq.rhs - eq.lhs"]
                sl += [f"nodes += [Node.from_sympy({cname}, '{cname}')]"]
            # self._nodes += [Node.from_sympy(eq.lhs - eq.rhs, str(c_name))]
            sl += [""]
        sl += [""]
        return "\n".join(sl)

    def compile_equation_constraints(self):
        sl = ["", "# " + "-" * 40, "# Equation Constraints", ""]

        from pprint import pprint

        eq_cstr = self._conf["modulus_project"]["training"]["stages"][self._stage_id][
            "data"
        ]["Equation Constraints"]

        def validate_lambda_weighting_interior(cname, c, ls):
            import sys

            unavailable = ["normal_x", "normal_y", "normal_z"]
            for v in unavailable:
                if v in ls:
                    print(
                        f"[error][constraint: {cname}] variable `{v}` not available in Interior sub-domain, only in Boundary subdomains."
                    )
                    sys.exit(1)

        def validate_lambda_weighting_bdry(cname, c, ls):
            import sys

            if "sdf" in ls:
                print(
                    f"[error][constraint: {cname}] variable `sdf` not available in Boundary sub-domain"
                )
                sys.exit(1)

        for cname, c in self._constraints.items():
            if not eq_cstr[cname]["include"]:
                continue

            domain = c["on_domain"]
            if c["on_domain"] in self._interior_subdomains:
                sd = self._interior_subdomains[domain]
                geom = sd["geom"]
                ps = "None"
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    geom = f"PDParam(geom={geom}, param_ranges={ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                outvar = "{" + f"'{cname}': 0" + "}"
                validate_lambda_weighting_interior(
                    cname, c, eq_cstr[cname]["lambda_weighting"]
                )
                s = f"""
pic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
        compute_sdf_derivatives={sd["compute_sdf_derivatives"]},
        lambda_weighting={{'{cname}': sympy.sympify('{eq_cstr[cname]["lambda_weighting"]}')}}
)
domain.add_constraint(pic, '{cname}')
"""
                sl += [s]

            if c["on_domain"] in self._boundary_subdomains:
                sd = self._boundary_subdomains[domain]
                geom = sd["geom"]
                ps = "None"
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    geom = f"PDParam(geom={geom}, param_ranges={ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                outvar = "{" + f"'{cname}': 0" + "}"
                # check for errors in lambda_weightint expression
                validate_lambda_weighting_bdry(
                    cname, c, eq_cstr[cname]["lambda_weighting"]
                )

                if isinstance(c["equation"].lhs, Integral):
                    eq = c["equation"]
                    print("Integral constraint")
                    print(c)
                    lambda_exp = eq_cstr[cname]["lambda_weighting"]
                    lambda_w = "{" + f"'{cname}': sympy.sympify('{lambda_exp}'), " + "}"
                    outvar = "{" + f"'{cname}': {eq.rhs}" + "}"
                    s = f"""
ibc = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size=1,
        integral_batch_size={eq_cstr[cname]["batch_size"]}, 
        outvar={outvar},
        lambda_weighting={lambda_w}
)
domain.add_constraint(ibc, '{cname}')
"""
                    sl += [s]
                    continue

                s = f"""
pbc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
        lambda_weighting={{'{cname}': sympy.sympify('{eq_cstr[cname]["lambda_weighting"]}')}}
)
domain.add_constraint(pbc, '{cname}')
"""

                sl += [s]

        return "\n".join(sl)

    def compile_equation_constraints_opt(self):
        sl = ["", "# " + "-" * 40, "# Equation Constraints", ""]

        from pprint import pprint

        print("compile_equation_constraints_opt")

        eq_cstr = self._conf["modulus_project"]["training"]["stages"][self._stage_id][
            "data"
        ]["Equation Constraints"]

        def validate_lambda_weighting_interior(cname, c, ls):
            import sys

            unavailable = ["normal_x", "normal_y", "normal_z"]
            for v in unavailable:
                if v in ls:
                    print(
                        f"[error][constraint: {cname}] variable `{v}` not available in Interior sub-domain, only in Boundary subdomains."
                    )
                    sys.exit(1)

        def validate_lambda_weighting_bdry(cname, c, ls):
            import sys

            if "sdf" in ls:
                print(
                    f"[error][constraint: {cname}] variable `sdf` not available in Boundary sub-domain"
                )
                sys.exit(1)

        ## group constraints
        c_dict = {}
        for cname, c in self._constraints.items():
            if not eq_cstr[cname]["include"]:
                continue

            domain = c["on_domain"]
            batch_size = eq_cstr[cname]["batch_size"]

            k = (domain, batch_size)
            if k not in c_dict:
                c_dict[k] = []
            c_dict[k] += [(cname, c)]

        pprint({k: len(c_dict[k]) for k in c_dict.keys()})

        ##

        for (domain, batch_size), constraints in c_dict.items():

            # for cname, c in self._constraints.items():
            #     if not eq_cstr[cname]["include"]:
            #         continue
            parameterization = ""
            if domain in self._interior_subdomains:
                sd = self._interior_subdomains[domain]
                geom = sd["geom"]
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    # geom = f"PDParam(geom={geom}, param_ranges={ps})"
                    ps = (
                        "{"
                        + ",".join(
                            [f"Parameter('{k}'):{v}" for k, v in sd["params"].items()]
                        )
                        + "}"
                    )
                    parameterization = f", parameterization=Parameterization({ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                outvar = "{" + f"'{cname}': 0" + "}"
                outvar = "{"
                lambda_w = "{"

                for cname, c in constraints:
                    lambda_exp = eq_cstr[cname]["lambda_weighting"]
                    validate_lambda_weighting_interior(cname, c, lambda_exp)

                    outvar += f"'{cname}': 0, "
                    lambda_w += f"'{cname}': sympy.sympify('{lambda_exp}'), "

                outvar += "}"
                lambda_w += "}"

                s = f"""
pic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
        compute_sdf_derivatives={sd["compute_sdf_derivatives"]},
        lambda_weighting={lambda_w}{parameterization}
)
domain.add_constraint(pic, '{cname}')
"""
                sl += [s]

            if domain in self._boundary_subdomains:

                sd = self._boundary_subdomains[domain]
                geom = sd["geom"]
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    # geom = f"PDParam(geom={geom}, param_ranges={ps})"
                    ps = (
                        "{"
                        + ",".join(
                            [f"Parameter('{k}'):{v}" for k, v in sd["params"].items()]
                        )
                        + "}"
                    )
                    parameterization = f", parameterization=Parameterization({ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                cname, c = constraints[0]
                if isinstance(c["equation"].lhs, Integral):
                    eq = c["equation"]
                    print("Integral constraint")
                    print(c)
                    lambda_exp = eq_cstr[cname]["lambda_weighting"]
                    lambda_w = "{" + f"'{cname}': sympy.sympify('{lambda_exp}'), " + "}"
                    outvar = "{" + f"'{cname}': {eq.rhs}" + "}"
                    s = f"""
ibc = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size=1,
        integral_batch_size={eq_cstr[cname]["batch_size"]}, 
        outvar={outvar},
        lambda_weighting={lambda_w}{parameterization}
)
domain.add_constraint(ibc, '{cname}')
"""
                    sl += [s]
                    continue

                outvar = "{" + f"'{cname}': 0" + "}"
                outvar = "{"
                lambda_w = "{"

                for cname, c in constraints:
                    lambda_exp = eq_cstr[cname]["lambda_weighting"]
                    validate_lambda_weighting_interior(cname, c, lambda_exp)

                    outvar += f"'{cname}': 0, "
                    lambda_w += f"'{cname}': sympy.sympify('{lambda_exp}'), "

                outvar += "}"
                lambda_w += "}"

                s = f"""
pbc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry={geom},
        criteria={criteria},
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
        lambda_weighting={lambda_w}{parameterization}
)
domain.add_constraint(pbc, '{cname}')
"""

                sl += [s]

        return "\n".join(sl)

    def compile_equation_constraints_sampled(self):
        sl = ["", "# " + "-" * 40, "# Equation Constraints Sampled", ""]
        sl += ["import h5py # to load HDF5 samples"]

        from pprint import pprint

        eq_cstr = self._conf["modulus_project"]["training"]["stages"][self._stage_id][
            "data"
        ]["Equation Constraints"]

        for cname, c in self._constraints.items():
            if not eq_cstr[cname]["include"]:
                continue

            domain = c["on_domain"]
            if c["on_domain"] in self._interior_subdomains:
                sd = self._interior_subdomains[domain]
                geom = sd["geom"]
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    geom = f"PDParam(geom={geom}, param_ranges={ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                # vlist = list(self._vars) + ["sdf"]
                # invar = "{" + ", ".join([f"'{v}': dv['{v}'][:]" for v in vlist]) + "}"
                outvar = "{" + f"'{cname}': np.zeros_like(dv['x'])" + "}"
                fname = f"../samples/{domain}.hdf5"
                # fname = f"../samples/{cname}.hdf5"
                s = f"""
dv = dict()
with h5py.File('{fname}', 'r') as f:
    for k in f.keys():
        dv[k] = f[k][:]
        if k == 'area':
            dv[k] *= 1000 # HACK
pic = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=dv,
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
)
domain.add_constraint(pic, '{cname}')
"""
                sl += [s]

            if c["on_domain"] in self._boundary_subdomains:
                sd = self._boundary_subdomains[domain]
                geom = sd["geom"]
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    geom = f"PDParam(geom={geom}, param_ranges={ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                # invar = "{" + ", ".join([f"'{v}': dv['{v}'][:]" for v in vlist]) + "}"
                outvar = "{" + f"'{cname}': np.zeros_like(dv['x'])" + "}"
                fname = f"../samples/{domain}.hdf5"
                # fname = f"../samples/{cname}.hdf5"
                s = f"""
dv = dict()
with h5py.File('{fname}', 'r') as f:
    for k in f.keys():
        dv[k] = f[k][:]
        if k == 'area':
            dv[k] *= 1000 # HACK
pic = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=dv,
        batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
        outvar={outvar},
)
domain.add_constraint(pic, '{cname}')
"""
                #                 s = f"""
                # pbc = PointwiseBoundaryConstraint(
                #         nodes=nodes,
                #         geometry={geom},
                #         criteria={criteria},
                #         batch_size={eq_cstr[cname]["batch_size"]}, #cfg.modulus_project.equation_constraints.{cname}.batch_size,
                #         outvar={outvar},
                # )
                # domain.add_constraint(pbc, '{cname}')
                # """

                sl += [s]

        return "\n".join(sl)

    def compile_equation_constraints_sampler(self):
        sl = ["", "# " + "-" * 40, "# Equation Constraints Sampler", ""]
        sl += ["import time, os", "ecs_t0 = time.time()"]
        sl += ["if not os.path.exists('samples'):", "  os.system('mkdir samples')"]

        from pprint import pprint

        eq_cstr = self._conf["modulus_project"]["training"]["stages"][self._stage_id][
            "data"
        ]["Equation Constraints"]

        for cname, c in self._constraints.items():
            if not eq_cstr[cname]["include"]:
                continue

            domain = c["on_domain"]
            if c["on_domain"] in self._interior_subdomains:
                sd = self._interior_subdomains[domain]
                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                geom = sd["geom"]
                parameterization = ""
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    # geom = f"PDParam(geom={geom}, param_ranges={ps})"
                    ps = (
                        "{"
                        + ",".join(
                            [f"Parameter('{k}'):{v}" for k, v in sd["params"].items()]
                        )
                        + "}"
                    )
                    parameterization = f", parameterization=Parameterization({ps})"

                outvar = "{" + f"'{cname}': 0" + "}"
                nr = eq_cstr[cname]["batch_size"] * 1000

                # fname = f"samples/{cname}.hdf5"
                fname = f"samples/{domain}.hdf5"
                s = f"""
import h5py
from modulus.geometry.parameterization import Parameterization, Parameter
subdomain = {geom}
samples = subdomain.sample_interior({nr}, criteria={criteria}, compute_sdf_derivatives={sd["compute_sdf_derivatives"]}{parameterization})
with h5py.File("{fname}", "w") as f:
    for k,v in samples.items():
        f.create_dataset(k, data = v)  
print(f'wrote {fname}')
"""

                sl += [s]

            if c["on_domain"] in self._boundary_subdomains:
                sd = self._boundary_subdomains[domain]
                geom = sd["geom"]
                parameterization = ""
                if sd["params"] is not None:
                    ps = (
                        "{"
                        + ",".join([f"'{k}':{v}" for k, v in sd["params"].items()])
                        + "}"
                    )
                    # geom = f"PDParam(geom={geom}, param_ranges={ps})"
                    ps = (
                        "{"
                        + ",".join(
                            [f"Parameter('{k}'):{v}" for k, v in sd["params"].items()]
                        )
                        + "}"
                    )
                    parameterization = f", parameterization=Parameterization({ps})"

                criteria = sd["criteria"]
                if criteria is not None:
                    criteria = f"sympy.sympify('{str(criteria)}')"

                outvar = "{" + f"'{cname}': 0" + "}"
                nr = eq_cstr[cname]["batch_size"] * 1000
                fname = f"samples/{domain}.hdf5"
                # fname = f"samples/{cname}.hdf5"
                s = f"""
import h5py
from modulus.geometry.parameterization import Parameterization, Parameter
subdomain = {geom}
t0=time.time()
samples = subdomain.sample_boundary({nr}, criteria={criteria}{parameterization})
t1=time.time()
print(f"sampled {cname} in", t1-t0, "s")
with h5py.File("{fname}", "w") as f:
    for k,v in samples.items():
        f.create_dataset(k, data = v)
print(f'wrote {fname}')
"""

                sl += [s]
        sl += [
            "ecs_t1 = time.time()",
            "print('sampled in ', f'{ecs_t1-ecs_t0:.3f}s')",
        ]

        return "\n".join(sl)

    def compile_data_constraints(self):
        sl = ["", "# " + "-" * 40, "# Data Constraints", ""]
        for cname, v in self._data_constraints.items():
            invars = ",".join([f"'{v}': f['{v}']" for v in self._vars])
            invars = "{" + invars + "}"
            s = f"""
import h5py
with h5py.File("{v['data_fname']}", 'r') as f:
    pbc = PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar={invars},
            batch_size=int(np.min([f['x'].shape[0], 512])),
            outvar={{'{v['outvar']}':f["{v['outvar']}"]}},
    )
    domain.add_constraint(pbc, '{v["data_fname"]}')
"""
            sl += [s]
        return "\n".join(sl)

    def compile_target_training(self, constraint_opt):
        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        sl = ["", "# " + "-" * 40, "# General Variables", ""]
        sl += ["nodes = []", "domain=Domain()"]
        sl += [self.compile_symbols()]
        sl += [self.compile_neural_networks()]
        sl += [self.compile_submodels()]
        sl += [self.compile_geometries()]
        sl += [self.compile_interior_subdomains()]
        sl += [self.compile_boundary_subdomains()]

        sl += [self.compile_equations()]

        if constraint_opt:
            sl += [self.compile_equation_constraints_opt()]
        else:
            sl += [self.compile_equation_constraints()]

        sl += [self.compile_data_constraints()]

        sl += ["", "# " + "-" * 40, "# Start Training Loop", ""]
        sl += ["slv = Solver(cfg, domain)"]
        sl += ["slv.solve()"]

        preamble = "# Generated by `mtc compile inference`\n\n"

        t = env.from_string(
            load_template(os.path.join("templates", "train-imports.py"))
        )
        preamble += t.render(no_modulus_main=self._no_modulus_main, conf_path=".")
        body = "\n".join(sl)
        body = "\n".join([" " * 4 + line for line in body.split("\n")])

        comp_str = preamble + "\n" + body + "\nrun()"
        return comp_str

    def compile_target_training_sampled(self):
        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        sl = ["", "# " + "-" * 40, "# General Variables", ""]
        sl += ["nodes = []", "domain=Domain()"]
        sl += [self.compile_symbols()]
        sl += [self.compile_neural_networks()]
        sl += [self.compile_submodels()]
        # sl += [self.compile_geometries()]
        # sl += [self.compile_interior_subdomains()]
        # sl += [self.compile_boundary_subdomains()]

        sl += [self.compile_equations()]
        sl += [self.compile_equation_constraints_sampled()]
        sl += [self.compile_data_constraints()]

        sl += ["", "# " + "-" * 40, "# Start Training Loop", ""]
        sl += ["slv = Solver(cfg, domain)"]
        sl += ["slv.solve()"]

        preamble = "# Generated by `mtc compile inference`\n\n"

        t = env.from_string(
            load_template(os.path.join("templates", "train-imports.py"))
        )
        preamble += t.render(no_modulus_main=self._no_modulus_main, conf_path=".")
        body = "\n".join(sl)
        body = "\n".join([" " * 4 + line for line in body.split("\n")])

        comp_str = preamble + "\n" + body + "\nrun()"
        return comp_str

    def compile_inference_section(self, stageid):
        from jinja2 import Environment, PackageLoader, select_autoescape

        template = load_template(os.path.join("templates", "inference_section.py"))

        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()
        t = env.from_string(template)

        outvars = [e for e in self._nn_outs]
        # if len(self._submodels) > 0:
        #     outvars += list(self._submodels.keys())

        possible_outvars = outvars + list(self._constraints.keys())

        coll_models = []
        try:
            if len(self._model["name"]):
                coll_models = [self._model["name"]]
                outvars += coll_models
        except:
            pass

        return t.render(
            stageid=stageid,
            _vars=self._vars,
            _submodels=outvars,
            self_model=self._model,
            possible_outvars=possible_outvars,
            coll_models=coll_models,
        )

    def make_infer_info(self, stage):
        template = """
info = {
    "stage": "{{stage}}",
    "__file__":__file__,
    "inputs": [{% for item in _vars %}'{{ item  }}',{% endfor %}],
    "default-outputs": [{% for item in outputs %}'{{ item  }}',{% endfor %}],
    "possible-outputs": [{% for item in possible_outvars %}'{{ item}}', {% endfor %} ]

}
        """
        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()
        t = env.from_string(template)

        outvars = [e for e in self._nn_outs]
        if len(self._submodels) > 0:
            outvars += list(self._submodels.keys())
        possible_outvars = outvars + list(self._constraints.keys())

        try:
            if len(self._model["name"]):
                outvars.append(self._model["name"])
        except:
            pass

        print(outvars)

        return t.render(
            stage=stage,
            _vars=self._vars,
            outputs=outvars,
            possible_outvars=possible_outvars,
        )

    def compile_target_inference(self, stageid):
        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        sl = ["", "# " + "-" * 40, "# General Variables", ""]
        sl += ["nodes = []", "domain=Domain()"]
        sl += [self.compile_symbols()]
        sl += [self.compile_neural_networks()]
        sl += [self.compile_submodels()]
        sl += [self.compile_geometries()]
        sl += [self.compile_interior_subdomains()]
        sl += [self.compile_boundary_subdomains()]

        sl += [self.compile_equations()]

        sl += ["", "# " + "-" * 40, "# Inference", "#" + "-" * 40, ""]
        sl += [self.compile_inference_section(stageid)]

        # no need to include constraints in inference
        sl += [self.compile_equation_constraints_opt()]
        sl += [self.compile_data_constraints()]

        sl += ["", "# " + "-" * 40, "# Start Training Loop", ""]
        sl += ["slv = Solver(cfg, domain)"]
        sl += ["slv._eval()"]

        preamble = "# Generated by `mtc compile inference`\n\n"

        t = env.from_string(
            load_template(os.path.join("templates", "train-imports.py"))
        )
        conf_path = os.path.join("training", stageid)
        preamble += t.render(no_modulus_main=self._no_modulus_main, conf_path=conf_path)
        body = "\n".join(sl)
        body = "\n".join([" " * 4 + line for line in body.split("\n")])

        comp_str = preamble + "\n" + body + "\nrun()\n"
        comp_str += self.make_infer_info(stageid)
        return comp_str

    def compile_target_sampler(self, stageid):
        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        sl = ["", "# " + "-" * 40, "# General Variables", ""]
        sl += ["nodes = []", "domain=Domain()"]
        sl += [self.compile_symbols()]
        # sl += [self.compile_neural_networks()]
        sl += [self.compile_submodels()]
        sl += [self.compile_geometries()]
        sl += [self.compile_interior_subdomains()]
        sl += [self.compile_boundary_subdomains()]

        # sl += [self.compile_equations()]

        # sl += ["", "# " + "-" * 40, "# Inference", "#" + "-" * 40, ""]
        # sl += [self.compile_inference_section(stageid)]

        # no need to include constraints in inference
        # sl += [self.compile_equation_constraints()]
        # sl += [self.compile_data_constraints()]
        sl += [self.compile_equation_constraints_sampler()]

        # sl += ["", "# " + "-" * 40, "# Start Training Loop", ""]
        # sl += ["slv = Solver(cfg, domain)"]
        # sl += ["slv._eval()"]

        preamble = "# Generated by `mtc compile --target sampler`\n\n"

        t = env.from_string(
            load_template(os.path.join("templates", "train-imports.py"))
        )
        conf_path = os.path.join("training", stageid)
        preamble += t.render(no_modulus_main=self._no_modulus_main, conf_path=conf_path)
        body = "\n".join(sl)
        body = "\n".join([" " * 4 + line for line in body.split("\n")])

        comp_str = preamble + "\n" + body + "\nrun()\n"
        # comp_str += self.make_infer_info(stageid)
        return comp_str

    def compile(
        self,
        compile_type="training",
        stageid="stage1",
        only1storder=False,
        constraint_opt=False,
    ):

        if only1storder:
            self.compile_to_firstorder()

        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        compile_types = ["training", "inference", "sampler", "geometry"]
        assert (
            compile_type in compile_types
        ), f"Got {compile_type}... Allowed compiling tragets: {compile_types}"

        self._conf = self.load_conf()
        self._stage_id = stageid

        if not os.path.exists("training"):
            os.makedirs("training")

        stagedir = os.path.join("training", stageid)
        if not os.path.exists(stagedir):
            os.makedirs(stagedir)

        if compile_type == "training":
            target_file = os.path.join(stagedir, "train.py")
            print(f"[mtc] compiling {target_file}")
            trainpy = self.compile_target_training(constraint_opt)
            with open(target_file, "w") as f:
                f.write(trainpy)
            print(f"[mtc] wrote {target_file}")

        if compile_type == "inference":
            self._no_modulus_main = True
            target_file = os.path.join(stagedir, "infer.py")
            print(f"[mtc] compiling {target_file}")
            inferpy = self.compile_target_inference(stageid)
            with open(target_file, "w") as f:
                f.write(inferpy)
            print(f"[mtc] wrote {target_file}")

        if compile_type == "geometry":
            from .compile.target_geometry import compile

            self._no_modulus_main = True
            target_file = os.path.join(stagedir, "geometry.py")
            print(f"[mtc] compiling {target_file}")
            inferpy = compile(self)  # self.compile_target_sampler(stageid)
            with open(target_file, "w") as f:
                f.write(inferpy)
            print(f"[mtc] wrote {target_file}")

        if compile_type == "sampler":
            self._no_modulus_main = True
            target_file = os.path.join(stagedir, "sample.py")
            print(f"[mtc] compiling {target_file}")
            inferpy = self.compile_target_sampler(stageid)
            with open(target_file, "w") as f:
                f.write(inferpy)
            print(f"[mtc] wrote {target_file}")

            self._no_modulus_main = False
            target_file = os.path.join(stagedir, "train_sampled.py")
            print(f"[mtc] compiling {target_file}")
            trainpy = self.compile_target_training_sampled()
            with open(target_file, "w") as f:
                f.write(trainpy)
            print(f"[mtc] wrote {target_file}")

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

        ## CREATE compile_submodels()
        # self._nodes += [Node.from_sympy(expr, str(name))]

        return Function(name)(*self._vars.values())

    def add_neural_network(self, name="", nn_type="", inputs=[], outputs=[]):
        assert len(name) > 0, "must provide a network name"

        assert name not in self._nns, f"{name} NN already exists"

        assert len(inputs) > 0, "must include a non-empty list of input vars"
        assert len(outputs) > 0, "must include a non-empty list of output vars"

        # assert nn_type in allowed_NN_types, f"{nn_type} not in list {allowed_NN_types}"

        # if len(self._vars) > 0:
        #     assert set(self._vars.keys()) == set(
        #         inputs
        #     ), "New NN definition does not match the input variables already defined"

        for v in inputs:
            self._vars[v] = Symbol(v)
        # self._vars = {v: Symbol(v) for v in inputs}

        ocheck = self._nn_outs.intersection(set(outputs))
        assert len(ocheck) == 0, f"Redefining output variables {ocheck}"
        self._nn_outs = self._nn_outs.union(set(outputs))

        ## CREATE a compile_neural_networks()

        # net = instantiate_arch(
        #     input_keys=[Key(v) for v in inputs],
        #     output_keys=[Key(v) for v in outputs],
        #     cfg=allowed_NN_types[nn_type],
        # )
        # self._nodes += [net.make_node(name=name, jit=_cfg.jit)]
        net = None

        invars = [self._vars[v] for v in inputs]
        outvars = [Function(v)(*invars) for v in outputs]

        self._nns[name] = {
            "net": net,
            "inputs": inputs,
            "outputs": outputs,
            "nn_type": nn_type,
            "invars": invars,
            "outvars": outvars,
        }

        return invars, outvars

    def add_interior_subdomain(
        self, name, geom=None, params=None, criteria=None, compute_sdf_derivatives=False
    ):
        # geom is required, the other two are optional
        assert geom is not None, "geom argument is required"

        assert (
            name not in self._interior_subdomains
            and name not in self._boundary_subdomains
        ), "subdomains must have unique names"

        self._interior_subdomains[name] = {
            "geom": geom,
            "params": params,
            "criteria": criteria,
            "compute_sdf_derivatives": compute_sdf_derivatives,
        }

        return name

    def add_boundary_subdomain(self, name, geom=None, params=None, criteria=None):
        # geom is required, the other two are optional
        assert geom is not None, "geom argument is required"
        assert (
            name not in self._interior_subdomains
            and name not in self._boundary_subdomains
        ), "subdomains must have unique names"

        self._boundary_subdomains[name] = {
            "geom": geom,
            "params": params,
            "criteria": criteria,
        }
        return name

    def add_data_constraint(self, name=None, model=None, data_fname=None):
        import h5py

        with h5py.File(data_fname) as f:
            vset = set(self._vars.keys())
            r = vset.difference(set(f.keys()))
            assert len(r) == 0, f"some variables are not represented in dataset: {r}"

            m = str(model)
            m = m[: m.find("(")]
            r = set([m]).difference(set(f.keys()))
            assert len(r) == 0, f"model not represented in dataset {r}"

        assert (
            str(type(model)) in self._submodels or str(m) in self._nn_outs
        ), f"undefined model {model}"

        # now add the constraint
        self._data_constraints[name] = {
            "model": model,
            "data_fname": data_fname,
            "outvar": m,
        }

    def add_constraint(self, cname, eq_constraint):
        "cname -- name of constraint"
        assert (
            cname not in self._constraints
        ), f"\n\n[{self.__class__.__name__}.add_constraint] '{cname}' constraint already defined\n\n"
        self._constraints[cname] = eq_constraint

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
            # self._nodes += [Node.from_sympy(eq.lhs - eq.rhs, str(c_name))]
        nodes = self._nodes

    def pprint_constraints(self):
        # print("=" * 80)
        print("Constraints")
        print("-" * 80)
        varstr = ", ".join(self._vars)
        varstr = f"({varstr})"
        if len(varstr) > 3:
            print("|  ", f"(.) = {varstr}\n|")
        for c_name, c_v in self._constraints.items():
            eq = c_v["equation"]
            eq0 = str(eq.lhs - eq.rhs)

            pre = f"|   {c_name}: "
            eq = f"{eq.lhs} = {eq.rhs}"

            if len(varstr) > 3:
                eq = eq.replace(varstr, "(.)")

            domain_type = (
                "int." if c_v["on_domain"] in self._interior_subdomains else "bdry"
            )

            domain_str = f"[on {domain_type} subdomain]"
            print(pre, eq)
            print("|   " + domain_str, c_v["on_domain"])
            print("|")

        print("| Data", "-" * 40)
        for dc_name in self._data_constraints:
            pre = f"  {dc_name}: "
            cd = self._data_constraints[dc_name]
            print("|", pre, f"[model]", cd["model"])
            print("|", " " * len(pre), f" [file]", cd["data_fname"])

            import h5py

            with h5py.File(cd["data_fname"]) as f:
                k = list(f.keys())[0]
                n = f[k].shape[0]
                print(
                    "|", " " * len(pre), f" [info] {n:,} pts | keys: {list(f.keys())}"
                )

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
            # spref = f"{nnn} = {nn['nn_type']}("
            spref = f"{nnn} = NeuralNetwork("
            s = spref + f"inputs={fmt(nn['inputs'])},"
            s += "\n" + " " * len(spref) + f"outputs={fmt(nn['outputs'])})"

            nns = s
            nns = "\n".join(["|  " + line for line in nns.split("\n")])
            print(nns)
        print("-" * 80, "\n")

    def pprint(self):
        print("------------")
        print("PINN Problem")
        print("------------")
        print()
        self.pprint_nns()
        self.pprint_models()
        self.pprint_constraints()
