from typing import Dict

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.node import Node

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset

from modulus.utils.io.plotter import GridValidatorPlotter

import torch
import h5py
import numpy as np
#from utilities import download_FNO_dataset, load_FNO_dataset

{% include "constraints.py" %}

@modulus.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    "{{problem_name}}"
{% filter indent(width=4) %}

domain = Domain()
nodes = []
{% for nn_name, nn in nns.items() %}

from modulus.models.fno import FNOArch
model{{nn_name}} = FNOArch(
        input_keys=[{% for v in nn.inputs %}Key("{{v}}"),{% endfor %}],
        output_keys=[{% for v in nn.outputs %}Key("{{v}}"),{% endfor %}],
        dimension=2
)    
nodes += model{{nn_name}}.make_nodes(name="{{nn_name}}_FNO", jit=False)
{% endfor %}
####################################################
# Define constraints
####################################################
ins_outs = [{% for v in nn_ins %}"{{v}}",{% endfor %}]+[{% for v in nn_outs %}"{{v}}"{% endfor %}]
_constraints = {}
{% for cn, c in constraints.items() %}
# Constraint: {{cn}} [{{c.type}}]
{% if c.type != "data" %}#  Equation: {{c.equation}}
srepr = "{{c.eq_srepr}}"
ctype = "interior"
gridvarfuncs={ {% for gv in grid.vars %}"{{gv|upper}}":{{grid.vars[gv]}},{% endfor %} }
{% if c.type == "boundary" %}ctype="boundary"{% endif %}
_constraints["{{cn}}"] = FNOEquationConstraint(ins_outs, gridvarfuncs, srepr, "{{cn}}", ctype=ctype, criteria="{{c.srepr_criteria}}")
# distribution = {{c.distribution}}
inputs = [{% for v in nn_ins %}"{{v}}",{% endfor %}]
_node = Node(
        inputs=ins_outs,
        outputs=["{{cn}}"],
        evaluate=_constraints["{{cn}}"],
        name="[{{cn}}] Node",
    )
# add constraints to domain
with h5py.File("{{distributions[c.distribution].file}}", "r") as f:
    invars = {}
    for k in f.keys():
        invars[k] = f[k][:]
        sh = invars[k].shape
        invars[k]= invars[k].reshape(sh[0], 1, sh[1], sh[2])
        print(k, invars[k].shape, invars.keys())
train_dataset = DictGridDataset(invars, {"{{cn}}": np.zeros_like(invars[k])})
supervised = SupervisedGridConstraint(
    nodes=nodes + [_node],
    dataset=train_dataset,
    batch_size=16,
)
domain.add_constraint(supervised, "supervised{{cn}}")
#nodes += [_node]
{% endif %}
{% endfor %}
# make solver
slv = Solver(cfg, domain)

# start solver
slv.solve()
{% endfilter %}

if __name__ == "__main__":
    run()
