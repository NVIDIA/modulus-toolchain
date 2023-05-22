from typing import Dict

import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.node import Node

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import DictGridDataset

from modulus.sym.utils.io.plotter import GridValidatorPlotter

import torch
import h5py
import numpy as np
#from utilities import download_FNO_dataset, load_FNO_dataset

{% include "constraints.py" %}

@modulus.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    "{{problem_name}}"
{% filter indent(width=4) %}

domain = Domain()
nodes = []
{% for nn_name, nn in nns.items() %}

from modulus.sym.models.fully_connected import FullyConnectedArch
decoder_net = FullyConnectedArch(
    input_keys=[Key("z")], # input to decoder
    output_keys=[{% for v in nn.outputs %}Key("{{v}}"),{% endfor %}], # output keys of decoder
    nr_layers=1, # number of layers
    layer_size=32 # layer size
)

from modulus.sym.models.fno import FNOArch
model{{nn_name}} = FNOArch(
        input_keys=[{% for v in nn.inputs %}Key("{{v}}"),{% endfor %}],
        decoder_net=decoder_net,
        dimension=2
)    
nodes += [model{{nn_name}}.make_node(name="{{nn_name}}_FNO")]
{% endfor %}
####################################################
# Define constraints
####################################################
ins_outs = [{% for v in nn_ins %}"{{v}}",{% endfor %}]+[{% for v in nn_outs %}"{{v}}"{% endfor %}]
_constraints = {}

dirichlet_conds = []
dirichlet_gen_conds = []
neumann_conds = []

{% for cn, c in constraints.items() %}
{% if c.type == "boundary-dirichlet" %}
var_id={{c.var_id}}
offset = {{c.var_offset}}
expr_srepr= "{{c.expr_srepr}}"
dirichlet_conds += [(var_id, offset, expr_srepr)]
{% elif c.type == "boundary-dirichlet-gen" %}
at_srepr = "{{c.at}}"
expr_srepr= "{{c.expr_srepr}}"
dirichlet_gen_conds += [(expr_srepr, at_srepr)]
{% elif c.type == "boundary-neumann" %}
var_id={{c.var_id}}
offset = {{c.var_offset}}
expr_srepr= "{{c.expr_srepr}}"
neumann_conds += [(var_id, offset, expr_srepr)]
{% endif %}
{% endfor %}

{% for cn, c in constraints.items() %}
# Constraint: {{cn}} [{{c.type}}]
{% if c.type == "interior" %}
srepr = "{{c.eq_srepr}}"
ctype = "interior"
gridvarfuncs={ {% for gv in grid.vars %}"{{gv|upper}}":{{grid.vars[gv]}},{% endfor %} }

_constraints["{{cn}}"] = FNOEquationConstraint(ins_outs, gridvarfuncs, srepr, "{{cn}}", "{{c.onFunc}}", dirichlet_gen_conds, dirichlet_conds, neumann_conds,
                                                ctype=ctype, criteria="{{c.srepr_criteria}}")
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
