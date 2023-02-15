from typing import Dict

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.node import Node
from modulus.graph import Graph

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset

from modulus.utils.io.plotter import GridValidatorPlotter

import torch
import h5py
import numpy as np

global struct

from modulus.hydra.utils import compose
cfg = compose(config_path="conf", config_name="config_PINO")
cfg.network_dir = "outputs/darcy_PINO"
def run(cfg=cfg) -> None:{% filter indent(width=4) %}
{% for nn_name, nn in nns.items() %}
nodes=[]
from modulus.models.fno import FNOArch
model{{nn_name}} = FNOArch(
        input_keys=[{% for v in nn.inputs %}Key("{{v}}"),{% endfor %}],
        output_keys=[{% for v in nn.outputs %}Key("{{v}}"),{% endfor %}],
        dimension=2
)    
nodes += model{{nn_name}}.make_nodes(name="{{nn_name}}_FNO", jit=False)
for node in nodes:
    node.evaluate.load("training/{{stageid}}/outputs")
gmodel = Graph(nodes, 
               [{% for v in nn.inputs %}Key("{{v}}"),{% endfor %}],
               [{% for v in nn.outputs %}Key("{{v}}"),{% endfor %}])
global struct
struct = {"graph": gmodel}
{% endfor %}{% endfilter %}

run()

info = {"__file__":__file__,
        "grid": {"N": {{grid.N}}, "vars": { {% for vn,v in grid.vars.items() %}"{{vn}}":{{v}}, {% endfor %} } }, 
        "input_grids": [{% for v in nn_ins %}Key("{{v}}"),{% endfor %}],
        "output_grids": [{% for v in nn_outs %}Key("{{v}}"),{% endfor %}]}

def infer({% for v in nn_ins %}{{v}},{% endfor %}):
        {% for v in nn_ins %}sh={{v}}.shape;{% endfor %}
        g =struct['graph']
        def arr2_tensor(a):
                sh = a.shape
                return torch.Tensor(a.reshape(sh[0], 1, sh[1], sh[2]))
        result = g.forward({ {% for v in nn_ins %}"{{v}}":arr2_tensor({{v}}),{% endfor %} })
        return {k: r.cpu().detach().numpy().reshape(sh) for k,r in result.items()}