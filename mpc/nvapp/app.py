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
