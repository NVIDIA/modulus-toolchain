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

import os


def load_yaml(filename):
    """Loads a YAML file using a path relative to where this module resides"""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return yaml.safe_load(f)


def load_config(path="./"):
    import yaml

    conf_path = os.path.join(path, "conf", "config.yaml")

    with open(conf_path) as f:
        return yaml.safe_load(f)


def customize_schema(path="./"):
    fname = os.path.join(os.path.dirname(__file__), "config_types_v2.yaml")
    schema = load_yaml(fname)

    config = load_config(path=path)

    # customize schema to reflect project
    arch = schema["arch"].copy()
    del schema["arch"]

    for nn_var in config["modulus_project"]["neural_networks"]:
        schema[nn_var] = arch.copy()
        schema[nn_var]["label"] = f"[{nn_var}] NN Architecture"

    # equation constraints
    constraints = schema["constraints"].copy()
    del schema["constraints"]

    cstr = {}
    eqc = {"type": "group", "default": {}}
    import json

    for eqn in config["modulus_project"]["equation_constraints"].keys():
        eqc["default"][eqn] = json.loads(json.dumps(constraints))
        eqc["default"][eqn]["label"] = f"{eqn}"
    schema["Equation Constraints"] = eqc

    eqc = {"type": "group", "default": {}}
    for nn_var in config["modulus_project"]["neural_networks"]:
        eqc["default"][f"{nn_var} Trainable"] = {"type": "bool", "default": True}
    schema["Neural Networks"] = eqc

    return schema


def config2dictV2(ctype):
    d = {}

    for k, v in ctype.items():
        if v["type"] == "option":
            assert v["default"] in v["choices"], f"wrong default in {k}"
            d[k] = config2dictV2(v["choices"][v["default"]])
            d[k]["__selected__"] = v["default"]
        elif v["type"] == "group":
            d[k] = config2dictV2(v["default"])
        else:
            d[k] = v["default"]

    return d
