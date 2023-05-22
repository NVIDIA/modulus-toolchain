
def make_infer_fn(outputs=[{% for item in _submodels %}'{{ item  }}',{% endfor %}]):

    coll_models=[{% for item in coll_models %}'{{ item  }}',{% endfor %}]

    invals = {str(v): np.array([0]).reshape(-1, 1) for v in [{% for item in _vars %}'{{ item  }}',{% endfor %}]}

    # requires_grad = False
    requires_grad = True
    for v in invals:
        for o in outputs:
            if f"__{v}" in o:
                requires_grad=True

    output_names = set(outputs).difference(set(coll_models))
    inferencer = PointwiseInferencer(
        invar=invals,
        output_names=output_names, #[submodel for submodel in self._submodels],
        nodes=nodes,
        batch_size=256 * 4 * 4 * 4,
        requires_grad=requires_grad
    )
    domain.add_inferencer(inferencer)
    # create inference function
    def infer_fn(*args, **kargs):
        "[{{ stageid }}] infer: ({% for item in _vars %}{{ item  }},{% endfor %}) -> ({% for item in _submodels %}{{ item  }}, {% endfor %}{{self_model['name']}})"
        from modulus.sym.domain.constraint import Constraint


        invals = {str(v): kargs[v].reshape(-1, 1) for v in [{% for item in _vars %}'{{ item  }}',{% endfor %}]}

        invar0 = invals
        invar = Constraint._set_device(
            invar0, requires_grad=requires_grad, device=inferencer.device
        )
        pred_outvar = inferencer.forward(invar)

        result = {}
        for submodel in output_names: #[{% for item in _submodels %}'{{ item  }}',{% endfor %}]:
            ret = pred_outvar[submodel].cpu().detach().numpy()
            ret_val = np.array([v for v in ret[:, 0]])
            result[submodel] = ret_val

        # now build the main model
        model = {"name": '{{ self_model["name"] }}', "conditions": [ {% for item in self_model['conditions'] %} { "func":sympy.sympify('{{item['func']}}'),"on":sympy.sympify('{{item['on']}}')}, {% endfor %}  ]}
        main_result = ret_val.copy().reshape(-1, 1)
        invars, invals = [], []
        for varn, varval in invar0.items():
            invars.append(Symbol(varn))
            invals.append(varval)

        if model['name'] in outputs:
            for smodel in model["conditions"]:
                func = smodel["func"]
                cond = smodel["on"]

                submodel_result = result[str(func)].reshape(-1, 1)
                from sympy import lambdify

                sel = lambdify(invars, cond)(*invals)
                main_result[sel] = submodel_result[sel]

            result[model["name"]] = main_result

        return result
    return infer_fn

infer_fn=make_infer_fn()
def infer_dispatch(*args, **kargs):
    if "outputs" not in kargs:
        return infer_fn(*args, **kargs)


    std = ({% for item in _submodels %}'{{ item  }}', {% endfor %}"{{self_model['name']}}")
    o2fn = {std: infer_fn}

    not_found=True
    for k, fn in o2fn.items():
        if len(set(kargs['outputs']).difference(set(k))) == 0:
            return fn(*args, **kargs)

    fn = make_infer_fn(kargs['outputs'])
    o2fn[tuple(kargs['outputs'])] = fn
    return fn(*args, **kargs)


global infer
infer = infer_dispatch