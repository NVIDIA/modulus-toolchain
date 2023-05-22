from typing import Dict

# import modulus
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.graph import Graph

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import DictGridDataset

from modulus.sym.utils.io.plotter import GridValidatorPlotter

import torch
import h5py
import numpy as np

global struct

from modulus.sym.hydra.utils import compose
cfg = compose(config_path="conf", config_name="config_PINO")
cfg.network_dir = "outputs/darcy_PINO"
def run(cfg=cfg) -> None:{% filter indent(width=4) %}
{% for nn_name, nn in nns.items() %}
nodes=[]
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