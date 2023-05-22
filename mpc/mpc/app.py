import os, time
import ipywidgets as ipw
import ipyvuetify as v
import time
import traitlets as tr

import os

import sys

sys.path.append(os.environ["MPC_PATH"] + "../mtc")
print(sys.path)
from mtc.config_utils import customize_schema, config2dictV2


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


# def config2dictV2(ctype):
#     d = {}

#     for k, v in ctype.items():
#         if v["type"] == "option":
#             assert v["default"] in v["choices"], f"wrong default in {k}"
#             d[k] = config2dictV2(v["choices"][v["default"]])
#             d[k]["__selected__"] = v["default"]
#         elif v["type"] == "group":
#             d[k] = config2dictV2(v["default"])
#         else:
#             d[k] = v["default"]

#     return d


def config2UI(schema, cdict, app, indent=""):
    fname_opt = "vue-templates/config-from-yaml-fast.vue"

    l = []
    for k, v in schema.items():

        print("[config2UIfast]", indent, k, f'[{v["type"]}]')

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

    return l


def config2UIold(schema, cdict, app, indent=""):
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


config_types = load_yaml("config_types.yaml")


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
            self.vbox.children = config2UI(nschema, self.cdict[self.key], self.app)


class ConfigStage(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/config-stage.vue")).tag(
        sync=True
    )
    cdict = tr.Dict({}).tag(sync=True)
    schema = tr.Dict({}).tag(sync=True)
    stageid = tr.Unicode("option").tag(sync=True)

    visible = tr.Bool(False).tag(sync=True)
    show_cdict = tr.Bool(False).tag(sync=True)

    def __init__(self, stageid, *ag, app=None, template_path="", **kargs):
        super().__init__(*ag, **kargs)
        self.app = app

        self.stageid = stageid

        # schema = load_yaml("config_types_v2.yaml")

        # # customize schema to reflect project
        # arch = schema["arch"].copy()
        # del schema["arch"]

        # for nn_var in self.app.config["modulus_project"]["neural_networks"]:
        #     schema[nn_var] = arch.copy()
        #     schema[nn_var]["label"] = f"[{nn_var}] NN Architecture"

        # # equation constraints
        # constraints = schema["constraints"].copy()
        # del schema["constraints"]

        # cstr = {}
        # eqc = {"type": "group", "default": {}}
        # import json

        # for eqn in self.app.config["modulus_project"]["equation_constraints"].keys():
        #     # eqc["default"][eqn] = {"type": "bool", "default": True}

        #     # eqc["default"][eqn] = {
        #     #     "type": "group",
        #     #     "label": eqn,
        #     #     "default": {"include": {"type": "bool", "default": True}},
        #     # }

        #     eqc["default"][eqn] = json.loads(json.dumps(constraints))
        #     eqc["default"][eqn]["label"] = f"{eqn}"
        # schema["Equation Constraints"] = eqc

        # eqc = {"type": "group", "default": {}}
        # for nn_var in self.app.config["modulus_project"]["neural_networks"]:
        #     eqc["default"][f"{nn_var} Trainable"] = {"type": "bool", "default": True}
        # schema["Neural Networks"] = eqc

        # self.schema = schema
        schema = customize_schema(path=os.environ["MPC_PROJECT_PATH"])
        self.schema = schema
        ######
        self.cdict = config2dictV2(schema)
        from pprint import pprint

        pprint(self.cdict)

        self.components = {
            "stage-config": ipw.VBox()
            # "stage-config": ipw.VBox(config2UI(schema, self.cdict, self.app))
        }

        print("done", self.__class__)

    def select_stage(self, stageid):
        import json

        self.stageid = stageid

        stage = self.app.config["modulus_project"]["training"]["stages"][stageid]
        if "data" not in stage:
            stage_data = config2dictV2(self.schema.copy())
        else:
            stage_data = stage["data"]
        self.cdict = json.loads(json.dumps(stage_data))

        t0 = time.time()
        print("inside select_stage", self.__class__)
        self.components["stage-config"].children = config2UI(
            self.schema, self.cdict, self.app
        )
        print(
            "[finished] inside select_stage", self.__class__, f"{time.time()-t0:.3f} s"
        )

    def vue_update_cdict(self, data):
        with self.app.output:

            # def update_cdict(w):
            #     "Recursively update the chain of dicts"
            #     if w.schema[w.key]["type"] in ["option", "group"]:
            #         for ww in w.vbox.children:
            #             if ww.schema[ww.key]["type"] in ["int", "float"]:
            #                 fn = eval(ww.schema[ww.key]["type"])
            #                 ww.cdict[ww.key] = fn(ww.cdict[ww.key])
            #             w.cdict[w.key][ww.key] = ww.cdict[ww.key]
            #             update_cdict(ww)

            # def update_cdict(w):
            #     "Recursively update the chain of dicts"
            #     if w.schema[w.key]["type"] in ["option", "group"]:
            #         if w.schema[w.key]["type"] == "group":
            #             for skey, sschema in w.schema[w.key]["default"].items():
            #                 if sschema["type"] in ["int", "float"]:
            #                     fn = eval(sschema["type"])
            #                     w.cdict[w.key][skey] = fn(w.cdict[w.key][skey])
            #         for ww in w.vbox.children:
            #             # print(ww)
            #             # if ww.schema[ww.key]["type"] in ["int", "float"]:
            #             #     fn = eval(ww.schema[ww.key]["type"])
            #             #     ww.cdict[ww.key] = fn(ww.cdict[ww.key])
            #             # w.cdict[w.key][ww.key] = ww.cdict[ww.key]
            #             update_cdict(ww)
            def update_cdict(w):
                "Recursively update the chain of dicts"
                if w.schema[w.key]["type"] in ["option", "group"]:
                    if w.schema[w.key]["type"] == "group":
                        for skey, sschema in w.schema[w.key]["default"].items():
                            if sschema["type"] in ["int", "float", "bool"]:
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


class TrainingSelectedStage(v.VuetifyTemplate):
    template = tr.Unicode(
        load_template("vue-templates/training-selected-stage.vue")
    ).tag(sync=True)
    stage = tr.Dict({"lr": 0.1}).tag(sync=True)
    opt_schema = tr.Dict(config_types["optimizer"]["adam"]).tag(sync=True)
    sched_schema = tr.Dict(config_types["scheduler"]["default"]).tag(sync=True)
    stage_id = tr.Unicode("stage1").tag(sync=True)

    def __init__(self, *ag, parent=None, app=None, **kargs):
        super().__init__(*ag, **kargs)
        self.app = app
        self.parent = parent

        self.components = {"config-stage": ConfigStage(self.stage_id, app=app)}
        self._first_selection = True
        self.select_stage(self.stage_id)

    def sync_conf(self):
        cs = self.components["config-stage"]
        cs.vue_update_cdict(0)
        self.stage["data"] = cs.cdict

        self.app.config["modulus_project"]["training"]["stages"][
            self.stage_id
        ] = self.stage.copy()
        self.parent.not_saved = True

    def select_stage(self, stageid):
        with self.app.output:
            # first, update the values from the UI
            if not self._first_selection:
                self.sync_conf()
            else:
                self._first_selection = False

            self.stage_id = stageid
            trconf = self.app.config["modulus_project"]["training"]
            self.stage = trconf["stages"][self.stage_id]

            for field in ["optimizer", "scheduler"]:
                if field not in self.stage:
                    self.stage[field] = config2dict(field)

            cs = self.components["config-stage"]
            cs.select_stage(stageid)

            self.stage = {}
            self.stage = trconf["stages"][self.stage_id]

    def vue_update(self, data):
        with self.app.output:
            self.sync_conf()
            self.parent.update()

    def vue_add_child_stage(self, data):
        self.parent.vue_extend_stage(self.stage_id)


class TrainingStages(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/training-stages.vue")).tag(
        sync=True
    )
    svgstr = tr.Unicode("").tag(sync=True)
    stage_dag = tr.List([]).tag(sync=True)
    stages = tr.List([]).tag(sync=True)

    not_saved = tr.Bool(False).tag(sync=True)
    editing_metadata = tr.Bool(False).tag(sync=True)

    show_stage_dag = tr.Bool(True).tag(sync=True)
    show_stage_ui = tr.Bool(True).tag(sync=True)

    metadata = tr.Dict({}).tag(sync=True)

    # nn_data = tr.Dict({}).tag(sync=True)
    # nn_schema = tr.Dict({}).tag(sync=True)
    # nn_types = tr.List([]).tag(sync=True)

    def __init__(self, *ag, app=None, **kargs):
        super().__init__(*ag, **kargs)
        self.app = app

        meta_schema = load_yaml("config_metadata.yaml")

        if "modulus_project" not in self.app.config:
            self.app.config["modulus_project"] = {}

        cfg = self.app.config["modulus_project"]

        if "metadata" in cfg:
            meta_cdict = {"metadata": cfg["metadata"]}
        else:
            meta_cdict = config2dictV2(meta_schema)
        self.metadata = meta_cdict["metadata"]

        self.components = {
            "project-metadata": ipw.VBox(config2UI(meta_schema, meta_cdict, self.app)),
            "selected-stage": TrainingSelectedStage(parent=self, app=self.app),
        }
        self.meta_cdict = meta_cdict

        self.update()

    def vue_save_metadata(self, data):
        with self.app.output:

            def update_cdict(w):
                "Recursively update the chain of dicts"
                if w.schema[w.key]["type"] in ["option", "group"]:
                    for ww in w.vbox.children:
                        w.cdict[w.key][ww.key] = ww.cdict[ww.key]
                        update_cdict(ww)

            d = self.meta_cdict.copy()
            for w in self.components["project-metadata"].children:
                update_cdict(w)
                d[w.key] = w.cdict[w.key]

            self.metadata = d["metadata"]
            self.app.config["modulus_project"]["metadata"] = d["metadata"]

            self.editing_metadata = False
            self.not_saved = True

    def vue_save_config_to_file(self, data):
        self.components["selected-stage"].sync_conf()
        self.update()

        save_config(self.app.config)
        self.not_saved = False

    def vue_add_new_stage(self, data):
        with self.app.output:
            sn = f"stage{len(self.stages)+1}"
            trconf = self.app.config["modulus_project"]["training"]
            trconf["stages"][sn] = {"description": ""}
            self.vue_select_stage(sn)

    def vue_select_stage(self, stageid):
        self.components["selected-stage"].select_stage(stageid)
        self.update()

    def vue_extend_stage(self, data):
        with self.app.output:
            sn = f"stage{len(self.stages)+1}"
            trconf = self.app.config["modulus_project"]["training"]

            # add new stage
            trconf["stages"][sn] = {"description": ""}

            # update DAG
            trconf["stage-dag"] += [[data, sn]]

            self.vue_select_stage(sn)

    def update(self):
        import graphviz as gv

        with self.app.output:

            trconf = self.app.config["modulus_project"]["training"]
            self.stages = list(trconf["stages"].keys())
            self.stage_dag = trconf["stage-dag"]

            dot = gv.Digraph(comment="the round table")
            dot.attr(rankdir="LR")
            dot.attr("node", shape="box", style="rounded,filled")

            dot.edges(trconf["stage-dag"])

            for sname, stage in trconf["stages"].items():
                fc = ""
                if self.components["selected-stage"].stage_id == sname:
                    fc = "yellow"
                dot.node(sname, sname + "\n" + stage["description"], fillcolor=fc)

            self.svgstr = dot._repr_image_svg_xml()


class TrainingPlots(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/training-plots.vue")).tag(
        sync=True
    )

    stages = tr.Dict({}).tag(sync=True)

    not_saved = tr.Bool(False).tag(sync=True)
    editing_metadata = tr.Bool(False).tag(sync=True)

    metadata = tr.Dict({}).tag(sync=True)

    def __init__(self, *ag, app=None, **kargs):
        super().__init__(*ag, **kargs)
        self.app = app

        stages = {}
        for stage in self.app.config["modulus_project"]["training"]["stages"].keys():
            stages[stage] = {"plot": False, "trained": False}
            stages[stage + "_sampled"] = {"plot": False, "trained": False}
        self.stages = stages

        self.outputwidget = ipw.Output()
        with self.outputwidget:
            import matplotlib.pyplot as plt

            plt.close("all")
            f = plt.figure(figsize=(5, 3))
            self.figure = f
            self.ax = plt.gca()

        # self.components = {"plt-figure": self.outputwidget}
        self.components = {"plt-figure": self.figure.canvas}
        self.update_plot()

    def vue_switch_stage(self, stage):
        self.stages[stage]["plot"] = not self.stages[stage]["plot"]
        self.update_plot()

    def update_plot(self):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        with self.app.output:
            self.ax.cla()
            for stage in self.stages:
                print(stage)

                stage_path = os.path.join(
                    os.environ["MPC_PROJECT_PATH"], "training", stage.split("_")[0]
                )
                logfile = "train_sampled.log" if "_sampled" in stage else "train.log"
                log_file = os.path.join(stage_path, "outputs", logfile)
                file_exists = os.path.exists(log_file)
                print(file_exists)
                self.stages[stage]["trained"] = file_exists

                if not self.stages[stage]["plot"]:
                    continue

                try:  # os.path.exists(log_file):
                    data = []
                    with open(log_file) as f:
                        file = f.read()
                        loss_lines = [l for l in file.split("\n") if "loss:" in l]

                        def line2float(s):
                            ns = s[s.find("[step:") :]
                            step = float(ns[ns.find(":") + 1 : ns.find("]")])

                            ns = s[s.find("loss") :]
                            loss = float(ns[ns.find(":") + 1 : ns.find(",")])
                            return (step, loss)

                        data = np.array([line2float(l) for l in loss_lines])

                    self.ax.semilogy(data[:, 0], data[:, 1], label=stage)
                    self.ax.set_xlabel("Step")
                    self.ax.set_ylabel("Avg Loss")

                except Exception as e:
                    print(e)

                self.ax.legend()
                plt.tight_layout()
            d = self.stages.copy()
            self.stages = {}
            self.stages = d


class App(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/app.vue")).tag(sync=True)
    projname = tr.Unicode(os.environ["MPC_PROJECT_PATH"]).tag(sync=True)
    config = tr.Dict(load_config()).tag(sync=True)
    v2n = tr.Dict(var2name).tag(sync=True)
    v2hint = tr.Dict(var2hint).tag(sync=True)
    loading_problem = tr.Bool(False).tag(sync=True)
    problemstr = tr.Unicode("").tag(sync=True)
    problem_show_cmd = tr.Unicode("mtc show problem").tag(sync=True)
    summary_type = tr.Unicode("Problem").tag(sync=True)

    problem_dialog = tr.Bool(False).tag(sync=True)

    def __init__(self, *ag, output=None, **kargs):
        super().__init__(*ag, **kargs)
        self.output = output  # for error and other messages

        self.components = {
            "training": TrainingStages(app=self),
            "training-plots": TrainingPlots(app=self),
        }

    # def update_cdict(self):
    #     # def update_cdict_working(cdict, w):
    #     #     if w.schema[w.key]["type"] in ["option", "group"]:
    #     #         for ww in w.vbox.children:
    #     #             cdict[ww.key] = update_cdict(cdict[ww.key], ww)
    #     #     else:
    #     #         return w.cdict[w.key]
    #     #     return cdict  # cdict

    #     def update_cdict(w):
    #         "Recursively update the chain of dicts"
    #         if w.schema[w.key]["type"] in ["option", "group"]:
    #             for ww in w.vbox.children:
    #                 w.cdict[w.key][ww.key] = ww.cdict[ww.key]
    #                 update_cdict(ww)

    #     d = self.cdict.copy()
    #     for w in self.components["test"].children:
    #         update_cdict(w)
    #         d[w.key] = w.cdict[w.key]

    #     self.cdict = {}
    #     self.cdict = d

    def refresh_config(self):
        self.config = load_config()

    def vue_load_show_training(self, data):
        self.summary_type = "Training"
        self.problem_show_cmd = "mtc show training"
        self.loading_problem = True
        t0 = time.time()
        res = os.popen("cd $MPC_PROJECT_PATH; mtc show training").read()
        t1 = time.time()
        self.problemstr = (
            f"[{os.environ['MPC_PROJECT_PATH']}] Fetched in {t1-t0:.3f} s\n" + res
        )
        self.loading_problem = False

    def vue_load_problem(self, data):
        self.summary_type = "Problem"
        self.problem_show_cmd = "mtc show problem"

        self.loading_problem = True
        t0 = time.time()
        res = os.popen("cd $MPC_PROJECT_PATH; mtc show problem").read()
        t1 = time.time()
        self.problemstr = (
            f"[{os.environ['MPC_PROJECT_PATH']}] Fetched in {t1-t0:.3f} s\n" + res
        )
        self.loading_problem = False

    def vue_load_problem_1storder(self, data):
        self.summary_type = "Problem"
        self.problem_show_cmd = "mtc show problem --only-first-order-ufunc"

        self.loading_problem = True
        t0 = time.time()
        res = os.popen(
            "cd $MPC_PROJECT_PATH; mtc show problem --only-first-order-ufunc"
        ).read()
        t1 = time.time()
        self.problemstr = (
            f"[{os.environ['MPC_PROJECT_PATH']}] Fetched in {t1-t0:.3f} s\n" + res
        )
        self.loading_problem = False


def new(output=None):
    """Creates a new app"""

    return App(output=output)
