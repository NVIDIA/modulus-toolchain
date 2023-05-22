from sympy import Symbol, Function, Or, And, Eq, Abs, Integral, expand
import sympy
import sympy as sp

from jinja2 import Template
import os, sys

import h5py


def load_template(filename):
    import jinja2 as j

    path = os.path.join(os.path.dirname(__file__), "templates", "fno-problem")
    env = j.Environment(loader=j.FileSystemLoader(path))

    return env.get_template(filename)


def load_yaml(filename):
    """Loads a YAML file using a path relative to where this module resides"""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return yaml.safe_load(f)


class FNOProblem:
    def __init__(self, name="FNO problem", cfg=None):

        self._problem_name = name
        self._nns = {}
        self._nn_outs = set()
        self._vars = {}
        self._interior_subdomains = {}
        self._constraints = {}
        self._distributions = {}

        self._grid = {}

    def to_hdf(self, hdf_fname, data):
        import h5py

        with h5py.File(hdf_fname, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)

    def load_conf(self):
        import yaml

        with open(os.path.join("conf", "config.yaml")) as f:
            conf = yaml.safe_load(f)
        return conf

    def save_conf(self, conf):
        import yaml

        with open(os.path.join("conf", "config.yaml"), "w") as f:
            yaml.safe_dump(conf, f)

    def add_neural_network(self, name="", nn_type="", inputs=[], outputs=[]):
        assert "vars" in self._grid, "Must set grid before defining NNs"
        assert len(name) > 0, "must provide a network name"

        assert name not in self._nns, f"{name} NN already exists"

        assert len(inputs) > 0, "must include a non-empty list of input vars"
        assert len(outputs) > 0, "must include a non-empty list of output vars"

        for v in inputs:
            self._vars[v] = Symbol(v)
        # self._vars = {v: Symbol(v) for v in inputs}

        ocheck = self._nn_outs.intersection(set(outputs))
        assert len(ocheck) == 0, f"Redefining output variables {ocheck}"
        self._nn_outs = self._nn_outs.union(set(outputs))

        gridvars = self._grid["vars"].keys()
        invars = [Function(v)(*gridvars) for v in inputs]
        outvars = [Function(v)(*gridvars) for v in outputs]
        net = None

        self._nns[name] = {
            "net": net,
            "inputs": inputs,
            "outputs": outputs,
            "nn_type": nn_type,
            "invars": invars,
            "outvars": outvars,
        }
        gridvarfuncs = [Function(str(v).upper())(*gridvars) for v in gridvars]
        return invars, gridvarfuncs, outvars

    def set_grid_domain(self, N: int, extent):
        assert self._grid == {}, "Grid may be defined only once!"
        assert len(extent) == 2, "Exactly 2 dimensions are required for the grid"

        N = int(N)
        self._grid["N"] = N
        self._grid["vars"] = {
            sp.Symbol(vn): {
                "var_id": vid,
                "extent": (float(ve[0]), float(ve[1])),
                "delta": (float(ve[1]) - float(ve[0])) / float(N - 1),
            }
            for vid, (vn, ve) in enumerate(extent.items())
        }

        vars = list(self._grid["vars"].keys())
        return vars + [self._grid["vars"][v]["delta"] for v in vars]

    def add_distribution(self, name, hdf5_file=""):
        assert "vars" in self._grid, "Must set grid before defining NNs"
        assert (
            name not in self._distributions
        ), f"[{name}] already defined (change name?)"
        assert hdf5_file != "", "Must provide full-path HDF5 file with data"

        # check file for valid data
        with h5py.File(hdf5_file, "r") as f:
            keys = list(f.keys())
            N = self._grid["N"]
            for k in f.keys():
                sh = f[k].shape
                assert (
                    sh[-2] == N and sh[-1] == N
                ), f"{hdf5_file}\nWrong dimensions of data -- last two must be ({N}, {N}), found {sh[-2:]} instead.\n"

        self._distributions[name] = {"file": hdf5_file, "nsamples": sh[0], "keys": keys}
        return name

    def add_data_constraint(self, name, over=None):
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"
        assert (
            over is not None
        ), "Must provide distribution that includes input/output data"

        self._constraints[name] = {"type": "data", "distribution": over}

    def add_interior_constraint(
        self, name, equation, on=None, over=None, criteria=None
    ):
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"
        assert (
            over is not None
        ), "Must provide distribution that includes input/output data"

        if criteria is not None:
            srepr_criteria = sp.srepr(criteria)
        else:
            srepr_criteria = ""
        self._constraints[name] = {
            "type": "interior",
            "equation": equation,
            "eq_srepr": sp.srepr(equation.lhs - equation.rhs),
            "distribution": over,
            "srepr_criteria": srepr_criteria,
            "criteria": criteria,
            "onFunc": str(on.func),
        }

    def add_dirichlet_gen_constraint(self, name, on=None, equation=None, at=None):
        assert on is not None and equation is not None and at is not None
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"

        # varinfo = self._grid["vars"][at.lhs]
        # if float(at.rhs) == float(varinfo["extent"][0]):
        #     var_offset = 0
        # elif float(at.rhs) == float(varinfo["extent"][1]):
        #     var_offset = -1
        # else:
        #     assert (
        #         False
        # ), f"The 'at' value must be either end of the extent {varinfo['extent']}"
        self._constraints[name] = {
            "type": "boundary-dirichlet-gen",
            "equation": equation,
            "expr_srepr": sp.srepr(equation.rhs),
            "at": sp.srepr(at),
            "condition": at,
            "onFunc": str(on.func),
        }

    def add_dirichlet_constraint(self, name, on=None, equation=None, at=None):
        assert on is not None and equation is not None and at is not None
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"

        varinfo = self._grid["vars"][at.lhs]
        if float(at.rhs) == float(varinfo["extent"][0]):
            var_offset = 0
        elif float(at.rhs) == float(varinfo["extent"][1]):
            var_offset = -1
        else:
            assert (
                False
            ), f"The 'at' value must be either end of the extent {varinfo['extent']}"
        self._constraints[name] = {
            "type": "boundary-dirichlet",
            "equation": equation,
            "expr_srepr": sp.srepr(equation.rhs),
            "at": at,
            "onFunc": str(on.func),
            "var_id": varinfo["var_id"],
            "var_offset": var_offset,
        }

    def add_neumann_constraint(self, name, on=None, equation=None, at=None):
        assert on is not None and equation is not None and at is not None
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"

        varinfo = self._grid["vars"][at.lhs]
        if float(at.rhs) == float(varinfo["extent"][0]):
            var_offset = 1
        elif float(at.rhs) == float(varinfo["extent"][1]):
            var_offset = self._grid["N"] - 2
        else:
            assert (
                False
            ), f"The 'at' value must be either end of the extent {varinfo['extent']}"

        self._constraints[name] = {
            "type": "boundary-neumann",
            "equation": equation,
            "expr_srepr": sp.srepr(equation.rhs),
            "at": at,
            "onFunc": on.func,
            "var_id": varinfo["var_id"],
            "var_offset": var_offset,
        }

    def add_boundary_constraint(self, name, equation, over=None, criteria=None):
        assert (
            name not in self._constraints
        ), f"Constraint named [{name}] already exists, choose a different name"
        assert (
            over is not None
        ), "Must provide distribution that includes input/output data"

        if criteria is not None:
            srepr_criteria = sp.srepr(criteria)
        else:
            srepr_criteria = ""
        self._constraints[name] = {
            "type": "boundary",
            "equation": equation,
            "eq_srepr": sp.srepr(equation.lhs - equation.rhs),
            "distribution": over,
            "srepr_criteria": srepr_criteria,
            "criteria": criteria,
        }

    def init_config(self, only1storder=False, max_steps=1000):
        if only1storder:
            self.compile_to_firstorder()

        conf = self.load_conf()

        def mkdeqc(dc):
            d = {k: str(v) for k, v in dc.items()}
            # if dc["on_domain"] in self._interior_subdomains:
            #     d["domain_type"] = "interior"
            # else:
            #     d["domain_type"] = "boundary"

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
            # "submodels": {k: str(v) for k, v in self._submodels.items()},
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

    def compile_target_inference(self, stageid):
        T = load_template("infer.py")

        data = {
            "problem_name": self._problem_name,
            "nns": self._nns,
            "distributions": self._distributions,
            "constraints": self._constraints,
            "grid": self._grid,
            "nn_ins": self._vars,
            "nn_outs": self._nn_outs,
            "stageid": stageid,
        }
        return T.render(data)

    def compile_target_training(self):
        T = load_template("train.py")

        data = {
            "problem_name": self._problem_name,
            "nns": self._nns,
            "distributions": self._distributions,
            "constraints": self._constraints,
            "grid": self._grid,
            "nn_ins": self._vars,
            "nn_outs": self._nn_outs,
        }
        return T.render(data)

    def compile(
        self,
        compile_type="training",
        stageid="stage1",
        only1storder=False,
        constraint_opt=False,
    ):

        from jinja2.nativetypes import NativeEnvironment

        env = NativeEnvironment()

        compile_types = ["training", "inference"]
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
            trainpy = self.compile_target_training()
            with open(target_file, "w") as f:
                f.write(trainpy)
            print(f"[mtc] wrote {target_file}")

        elif compile_type == "inference":
            target_file = os.path.join(stagedir, "infer.py")
            print(f"[mtc] compiling {target_file}")
            inferpy = self.compile_target_inference(stageid)
            with open(target_file, "w") as f:
                f.write(inferpy)
            print(f"[mtc] wrote {target_file}")

    def pprint_nns(self):
        # print("=" * 80)
        print("Neural Networks")
        print("-" * 80)
        fmt = lambda l: [Symbol(v) for v in l]
        for nnn, nn in self._nns.items():
            # spref = f"{nnn} = {nn['nn_type']}("
            spref = f"{nnn} = FNO("
            s = spref + f"input_funcs={fmt(nn['inputs'])},"
            s += "\n" + " " * len(spref) + f"output_funcs={fmt(nn['outputs'])})"

            nns = s
            nns = "\n".join(["|  " + line for line in nns.split("\n")])
            print(nns)
        print("-" * 80, "\n")

    def pprint_discrete_domain(self):
        print("Discrete Domain (Grid)")
        print("-" * 80)
        gvars = self._grid["vars"]
        vars = ",".join([str(v) for v in gvars.keys()])
        dtype = "x".join([f"{gvars[v]['extent']}" for v in gvars.keys()])
        gdelta = ", ".join([f"d{v}={gvars[v]['delta']}" for v in gvars.keys()])
        gsize = ", ".join([f"N_{v}={self._grid['N']}" for v in gvars.keys()])
        print("|  " + f"({vars})" + " in " + dtype)
        print("|  " + gdelta + " | " + gsize)
        print("-" * 80, "\n")

    def pprint_distributions(self):
        print("Distributions")
        print("-" * 80)
        for dname, d in self._distributions.items():
            print(
                f"|  {dname}: #samples = {d['nsamples']} | entries = {', '.join(d['keys'])}"
            )
        print("-" * 80, "\n")

    def pprint_constraints(self):
        print("Constraints")
        print("-" * 80)
        for cn, c in self._constraints.items():
            more = ""
            if c["type"] != "data":
                eq = c["equation"]
                more = f"\n|    {eq.lhs} = {eq.rhs}"
            print(f"|  {cn} [{c['type']}] over '{c['distribution']}'" + more + "\n|")
        print("-" * 80, "\n")

    def pprint(self):
        print("----------------")
        print("FNO/PINO Problem")
        print("----------------")
        print()
        self.pprint_discrete_domain()
        self.pprint_nns()
        self.pprint_distributions()
        self.pprint_constraints()

    def latex_constraints(self):
        s = ""

        for cn, c in self._constraints.items():
            s += f"\\text{{{cn} }} : \quad & "
            if c["type"] == "boundary-dirichlet-gen":
                eq = c["equation"]
                s += sp.latex(eq.lhs) + " = " + sp.latex(eq.rhs)
                s += ",  \\quad " + sp.latex(c["condition"]) + "  "
            elif c["type"] == "boundary-dirichlet":
                eq = c["equation"]
                s += sp.latex(eq.lhs) + " = " + sp.latex(eq.rhs)
                s += ",  \\quad " + sp.latex(c["at"]) + "  "
            elif c["type"] == "boundary-neumann":
                eq = c["equation"]
                s += sp.latex(eq.lhs) + " = " + sp.latex(eq.rhs)
                s += ",  \\quad " + sp.latex(c["at"]) + "  "

            elif c["type"] != "data":
                eq = c["equation"]
                s += sp.latex(eq.lhs) + " = " + sp.latex(eq.rhs)

                if "criteria" in c and c["criteria"] is not None:
                    s += " ; \quad " + sp.latex(c["criteria"])

            typestr = c["type"].replace("boundary-", "")
            s += f"& \\text{{ {typestr} }} \\\\ \\\\\n"
        r = "\\begin{split}\\begin{aligned}\n" + s + "\n\\end{aligned}\\end{split}"
        return r

    def _repr_latex_(self):
        s = ""

        s += f"\\text{{ Project: }} & \\text{{ {self._problem_name} }} \\\\\n"

        svars = ", ".join([str(v) for v in self._grid["vars"]])
        ss = "\\times".join(
            f"{v['extent']}^{{ {self._grid['N']} }}"
            for vs, v in self._grid["vars"].items()
        )
        deltas = ", ".join(
            f"d{vs}={v['delta']}" for vs, v in self._grid["vars"].items()
        )
        s += f"\\text{{Grid:}}& \\quad ({svars}) \in {ss} \\quad {deltas} \\\\\n"
        for nnn, nn in self._nns.items():
            domain = ", ".join([sp.latex(e) for e in nn["invars"]])
            image = ", ".join([sp.latex(e) for e in nn["outvars"]])
            ss = f"\\text{{ {nnn} [FNO]:}}& \\quad ({domain}) \mapsto {image} \\\\ \n"
            s += ss
        s += "\\\\ \n"
        s = "\\begin{split}\\begin{aligned}\n" + s + "\n\\end{aligned}\\end{split}"

        s += self.latex_constraints()
        return s
