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
