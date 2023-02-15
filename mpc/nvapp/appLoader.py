import ipyvuetify as v
import traitlets as tr
from .common import load_template, reload_module

import ipywidgets as ipw
import time


class AppLoader(v.VuetifyTemplate):
    template = tr.Unicode(load_template("vue-templates/app-loader.vue")).tag(sync=True)
    apps = tr.List(["widget1", "widget2"]).tag(sync=True)
    selected_app = tr.Unicode("").tag(sync=True)
    info = tr.Unicode("").tag(sync=True)
    app_dialog = tr.Bool(False).tag(sync=True)
    wsz = tr.Dict({}).tag(sync=True)
    output_dialog = tr.Bool(False).tag(sync=True)

    loading_app = tr.Bool(False).tag(sync=True)
    loading_app_result = tr.Unicode("").tag(sync=True)

    def __init__(self, appcfg, *ag, **kargs):
        super().__init__(*ag, **kargs)

        self.appcfg = appcfg
        self.app_box = ipw.VBox([])
        self.app_output = ipw.Output()
        self.components = {"mycontent": self.app_box, "app-output": self.app_output}

        self.apps = self.appcfg["apps"]
        self.selected_app = self.apps[0]

    def vue_clear_output(self, data):
        self.app_output.clear_output()

    def vue_reload(self, data):

        try:
            self.app_output.clear_output()
            self.loading_app_result = ""
            self.loading_app = True
            with self.app_output:
                try:
                    t0 = time.time()
                    print(f"Loading {self.selected_app}")
                    self.info = ""
                    mod = reload_module(self.selected_app)
                    self.app = mod.new(output=self.app_output)
                    self.app_box.children = (self.app,)
                    self.loading_app_result = "success"

                    t1 = time.time()
                    print(f"Loaded {self.selected_app} in {t1-t0:.1f} s")
                except Exception as e:
                    self.loading_app_result = "error"
                    self.info = f"Error: check out the output"
                    raise e
                finally:
                    self.loading_app = False
        except Exception as e:
            self.info = f"{e} "
        finally:
            self.loading_app = False

    def vue_pressed_r(self, data):
        self.selected_app = "Pressed R" + str(data)

        if data == "h":
            self.m.center = [-70, 10]
