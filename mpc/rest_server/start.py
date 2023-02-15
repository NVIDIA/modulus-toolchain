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

import sys, os
from subprocess import Popen, PIPE, check_call, run

CMD = "python -m mpc.rest_server.launcher_subprocess".split()

stageid = sys.argv[1]
path = f"infer"
print(os.getcwd(), path)
print(CMD)

p = Popen(CMD, stderr=PIPE, stdin=PIPE, stdout=PIPE, universal_newlines=True)


p.stdin.write(f"load-infer-function {path}\n")
p.stdin.flush()

ll, l = "", ""
while not l.startswith(f"[mpc] loaded {path}"):
    l = p.stdout.readline()
    ll += l
    print(l, end="")

p.stdin.write(f"info\n")
p.stdin.flush()
import json
from pprint import pprint

info = json.loads(p.stdout.readline())
info["load-infer"] = ll
pprint(info)
