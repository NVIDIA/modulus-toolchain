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

import os, time
import ipywidgets as ipw
import ipyvuetify as v
import time
import traitlets as tr

import os


def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


var2name = {"lr": "Learning Rate"}

var2hint = {
    "eps": "Numerical threshold below which numbers are considered 0",
    "betas": "Need two fo those; inside [0,1]",
}


def load_config():
    import yaml

    conf_path = os.path.join(os.environ["MPC_PROJECT_PATH"], "conf", "config.yaml")

    with open(conf_path) as f:
        return yaml.safe_load(f)


def save_config(conf):
    import yaml

    conf_path = os.path.join(os.environ["MPC_PROJECT_PATH"], "conf", "config.yaml")

    with open(conf_path, "w") as f:
        return yaml.safe_dump(conf, f)


def load_yaml(filename):
    """Loads a YAML file using a path relative to where this module resides"""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return yaml.safe_load(f)


def config2dict(ctype):
    choice = list(config_types[ctype].keys())[0]
    conf = config_types[ctype][choice]
    d = {k: v["default"] for k, v in conf.items() if "default" in v}

    return d


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


def config2UI(schema, cdict, app, indent=""):
    fname_opt = "vue-templates/config-from-yaml-option.vue"
    fname_grp = "vue-templates/config-from-yaml-group.vue"
    fname_bas = "vue-templates/config-from-yaml-base.vue"

    l = []
    for k, v in schema.items():

        print("[config2UI]", indent, k, f'[{v["type"]}]')

        if v["type"] == "option":
            w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
            l.append(w)
            w.vbox.children = config2UI(
                schema[k]["choices"][cdict[k]["__selected__"]],
                cdict[k],
                app,
                indent=indent + "  ",
            )

        elif v["type"] == "group":
            w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
            l.append(w)
            w.vbox.children = config2UI(
                schema[k]["default"], cdict[k], app, indent=indent + "  "
            )

        else:  # base
            w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
            l.append(w)

    return l


def config2UIfast(schema, cdict, app, indent=""):
    fname_opt = "vue-templates/config-from-yaml-fast.vue"

    l = []
    for k, v in schema.items():

        print("[config2UIfast]", indent, k, f'[{v["type"]}]')

        if v["type"] == "option":
            w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
            l.append(w)
            w.vbox.children = config2UIfast(
                schema[k]["choices"][cdict[k]["__selected__"]],
                cdict[k],
                app,
                indent=indent + "  ",
            )

        elif v["type"] == "group":
            w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
            l.append(w)
            w.vbox.children = config2UIfast(
                schema[k]["default"], cdict[k], app, indent=indent + "  "
            )

        # else:  # base
        #     w = ConfigFromYAMLClass(k, cdict, schema, app=app, template_path=fname_opt)
        #     l.append(w)

    return l


# config_types = load_yaml("config_types.yaml")


class ConfigFromYAMLClass(v.VuetifyTemplate):
    template = tr.Unicode(
        load_template("vue-templates/config-from-yaml-option.vue")
    ).tag(sync=True)
    cdict = tr.Dict({}).tag(sync=True)
    schema = tr.Dict({}).tag(sync=True)
    key = tr.Unicode("option").tag(sync=True)

    visible = tr.Bool(False).tag(sync=True)

    def __init__(self, key, cdict, schema, *ag, app=None, template_path="", **kargs):
        super().__init__(*ag, **kargs)
        self.app = app

        self.template = load_template(template_path)
        self.cdict = cdict
        self.schema = schema
        self.key = key

        self.vbox = ipw.VBox([])
        self.components = {"yaml-items": self.vbox}

    def vue_update_cdict(self, data):
        with self.app.output:
            print("update", self.key)
            self.app.update_cdict()

    def vue_update_choice(self, selection):
        with self.app.output:
            print("selection", selection)

            print("before")
            print(self.cdict)
            print("-" * 20)

            self.schema[self.key]["default"] = selection
            nschema = self.schema[self.key]["choices"][selection]
            self.cdict = config2dictV2(self.schema)
            print(self.cdict)
            self.vbox.children = config2UIfast(nschema, self.cdict[self.key], self.app)


class ConfigParent(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/config-parent.vue")).tag(
        sync=True
    )
    cdict = tr.Dict({}).tag(sync=True)
    schema = tr.Dict({}).tag(sync=True)
    stageid = tr.Unicode("option").tag(sync=True)

    visible = tr.Bool(False).tag(sync=True)
    show_cdict = tr.Bool(False).tag(sync=True)

    def __init__(self, *ag, app=None, template_path="", **kargs):
        super().__init__(*ag, **kargs)
        self.app = app

        schema = load_yaml("config_types_v2.yaml")

        self.schema = schema
        ######
        self.cdict = config2dictV2(schema)

        self.components = {"stage-config": ipw.VBox(), "stage-config-fast": ipw.VBox()}

        t0 = time.time()
        # self.components["stage-config"].children = config2UI(
        #     schema, self.cdict, self.app
        # )
        t1 = time.time()
        self.components["stage-config"].children = config2UIfast(
            schema, self.cdict, self.app
        )
        t2 = time.time()

        print(f"stage-config       {t1-t0:.3f} s")
        print(f"stage-config-fast  {t2-t1:.3f} s")

    def select_stage(self, stageid):
        import json

        self.stageid = stageid

        stage = self.app.config["modulus_project"]["training"]["stages"][stageid]
        if "data" not in stage:
            stage_data = config2dictV2(self.schema.copy())
        else:
            stage_data = stage["data"]
        self.cdict = json.loads(json.dumps(stage_data))

        self.components["stage-config-fast"].children = config2UIfast(
            schema, self.cdict, self.app
        )

        # self.components["stage-config"].children = config2UIfast(
        #     self.schema, self.cdict, self.app
        # )

    def vue_update_cdict(self, data):
        with self.app.output:

            def update_cdict(w):
                "Recursively update the chain of dicts"
                if w.schema[w.key]["type"] in ["option", "group"]:
                    if w.schema[w.key]["type"] == "group":
                        for skey, sschema in w.schema[w.key]["default"].items():
                            if sschema["type"] in ["int", "float"]:
                                fn = eval(sschema["type"])
                                w.cdict[w.key][skey] = fn(w.cdict[w.key][skey])
                    if w.schema[w.key]["type"] == "option":
                        for skey, sschema in w.schema[w.key]["choices"][
                            w.cdict[w.key]["__selected__"]
                        ].items():
                            if sschema["type"] in ["int", "float"]:
                                fn = eval(sschema["type"])
                                w.cdict[w.key][skey] = fn(w.cdict[w.key][skey])

                sschema = w.schema[w.key]
                # print("base case", w.key, sschema["type"])
                if sschema["type"] in ["int", "float"]:
                    fn = eval(sschema["type"])
                    w.cdict[w.key][skey] = fn(w.cdict[w.key][skey])

                # print(w.key, len(w.vbox.children))
                for ww in w.vbox.children:
                    update_cdict(ww)
                    w.cdict[w.key][ww.key] = ww.cdict[ww.key]

            d = self.cdict.copy()
            for w in self.components["stage-config"].children:
                update_cdict(w)
                d[w.key] = w.cdict[w.key]

            self.cdict = {}
            self.cdict = d


class App(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/app.vue")).tag(sync=True)
    # projname = tr.Unicode(os.environ["MPC_PROJECT_PATH"]).tag(sync=True)
    # config = tr.Dict(load_config()).tag(sync=True)
    # v2n = tr.Dict(var2name).tag(sync=True)
    # v2hint = tr.Dict(var2hint).tag(sync=True)
    # loading_problem = tr.Bool(False).tag(sync=True)
    problemstr = tr.Unicode("").tag(sync=True)

    problem_dialog = tr.Bool(False).tag(sync=True)

    def __init__(self, *ag, output=None, **kargs):
        super().__init__(*ag, **kargs)
        self.output = output  # for error and other messages

        self.components = {
            "config": ConfigParent(app=self),
        }

    def refresh_config(self):
        self.config = load_config()

    def vue_load_problem(self, data):

        self.loading_problem = True
        t0 = time.time()
        res = os.popen("cd $MPC_PROJECT_PATH; mtc show problem").read()
        t1 = time.time()
        self.problemstr = (
            f"[{os.environ['MPC_PROJECT_PATH']}] Fetched in {t1-t0:.3f} s\n" + res
        )
        self.loading_problem = False


def new(output=None):
    """Creates a new app"""

    return App(output=output)
