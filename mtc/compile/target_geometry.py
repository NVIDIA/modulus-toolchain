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
from jinja2 import Template


def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def compile(problem):
    T = Template(load_template(os.path.join("templates", "warp_geometry.py")))
    data = {}
    data["geometries"] = [{"name": name, "g": g} for name, g in problem._geom.items()]
    params = {}
    for name, g in problem._geom.items():
        p = g.get("params", {})
        if p is not None:
            for k, v in p.items():
                try:
                    if len(v) > 0:
                        v = v[0]
                except:
                    v = float(v)
                params[k] = v
    print(params)
    data["params"] = params

    paramstr = ", ".join([f"{k}={v}" for k, v in params.items()])
    if len(paramstr) > 0:
        paramstr = ", " + paramstr
    data["paramstr"] = paramstr

    paramcallstr = ", ".join([f"{k}" for k, v in params.items()])
    if len(paramcallstr) > 0:
        paramcallstr = ", " + paramcallstr
    data["paramcallstr"] = paramcallstr

    return T.render(data)
