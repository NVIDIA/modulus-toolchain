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

import os, sys
import importlib

# RELOAD_PATHS = ["/home/pavel/work/repos/L2RPN/individual/pavel/topo-sim/"]


def should_reload(m, RELOAD_PATHS):
    if m is None:
        return False

    try:
        for rp in RELOAD_PATHS:
            if "__file__" in dir(m) and rp in m.__file__:
                return True
    except:
        pass
    return False


def reload(mod, RELOAD_PATHS):
    # prepare list of modules that should be reloaded (by removing them)
    dellst = []
    for mname, m in sys.modules.items():
        if should_reload(m, RELOAD_PATHS):  # and mod != m:
            dellst += [mname]

    for mname in dellst:
        if mname in sys.modules:
            del sys.modules[mname]

    # now reload
    # importlib.reload(mod)
    mod = importlib.import_module(modname)
    return mod


def load_template(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        return f.read()


def reload_module(modname: str):
    if modname not in sys.modules:
        if "." in modname:
            a, b = modname.split(".")
            importlib.import_module(a)
        mod = importlib.import_module(modname)
    else:
        mod = sys.modules[modname]
        RELOAD_PATHS = [os.path.dirname(mod.__file__) + "/"]

        try:
            mod = reload(mod, RELOAD_PATHS)
        except:
            mod = importlib.import_module(modname)
    return mod
