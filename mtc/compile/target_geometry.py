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
