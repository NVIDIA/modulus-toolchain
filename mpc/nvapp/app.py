import json
import ipyvuetify as v

from .appLoader import AppLoader
from .common import load_template

# update the CSS a bit


# get_ipython().run_cell_magic(
#     "HTML",
#     "",
#     "<style>\n.jp-Cell {\n    margin:unset;\n    padding: unset;\n}\n.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper{\n    margin:unset;\n}\n.jp-Notebook {\n    margin:unset;\n    padding: unset;\n}\n.p-Widget {\n    width: 100%;\n}\n</style>",
# )


# load the app configuration
with open("app.json") as f:
    j = json.load(f)
theapp = AppLoader(j)
# with theapp.app_output:
display(theapp)
