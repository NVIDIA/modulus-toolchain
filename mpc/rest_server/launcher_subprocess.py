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

"""
This module receives commands from sys.stdin, one line at a time
and loads infer() functions. 

This module is inside the sub-process specific to a infer.py (a stage-specific
sub-process)

Valid commands:
load-infer-function [path]
info
infer
exit
"""

import sys
import importlib
import time

import numpy as np

n = 0
while True:
    n += 1
    line = sys.stdin.readline()
    if line == "":
        sys.exit(0)

    cmd = line.split()[0]
    rest = " ".join(line.split(" ")[1:])[:-1]
    if cmd == "exit":
        sys.exit()

    if cmd == "load-infer-function":
        path = " ".join(line.split(" ")[1:])[:-1]

        t0 = time.time()
        m = importlib.import_module(path)
        t1 = time.time()
        print(f"[mpc] loaded {path} in {t1-t0:.3f} s", flush=True)

    if cmd == "info":
        import json

        # d = {"m.__file__": m.__file__, "m.infer.__doc__": m.infer.__doc__}
        # print(json.dumps(d), flush=True)
        print(json.dumps(m.info), flush=True)

    if cmd == "infer":
        r = m.infer(**eval(rest))
        for k, v in r.items():
            fname = f"/dev/shm/mpc/{k}.npy"
            np.save(file=fname, arr=v)
            print("wrote", fname, flush=True)

# load-infer-function infer
# infer {"Q": np.array([1]).reshape(-1, 1),"x": np.array([1]).reshape(-1, 1),"y": np.array([1]).reshape(-1, 1), "a": np.array([1]).reshape(-1, 1), "b": np.array([1]).reshape(-1, 1), "q": np.array([1]).reshape(-1, 1), "airT": np.array([1]).reshape(-1, 1)}
