class FNOEquationConstraint(torch.nn.Module):
    "Custom Equation Constraint"

    def __init__(self, vars, gridvarfuncs, eq_srepr, outvar, ctype="interior", criteria=None):
        "ctype in ['interior', 'boundary']"
        ctypes=['interior', 'boundary']
        assert ctype in ctypes, f"Invalid ctype={ctype} -- Constraint type must be one of {ctypes}"
        super().__init__()
        self.vars = vars
        self.gridvarfuncs = gridvarfuncs
        self.eq_srepr = eq_srepr
        self.outvar = outvar
        self.ctype=ctype

        self.criteria=criteria

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _vars = {
            "_dh": { {% for vn,v in grid.vars.items() %}"{{vn}}":{{v.delta}}, {% endfor %} }, #{"x": 1.0, "y": 1.0},
            "_symbol2order": { {% for vn,v in grid.vars.items() %}"{{vn}}":{{loop.index0}}, {% endfor %} },
        }
        for v in self.vars:
            _vars[v] = input_var[v]
            inT = input_var[v]

        mgargs = [ np.arange(gv['extent'][0], gv['extent'][1], gv['delta']) for gvn, gv in self.gridvarfuncs.items()]

        # Set up the grid variable functions to allow for, e.g., sin(X)
        r = np.meshgrid(*mgargs)
        for gvn, _V in zip(self.gridvarfuncs, r):
            _V = torch.Tensor( _V.reshape([1,1]+list(_V.shape)) ).to(inT.device)
            _V = torch.repeat_interleave(_V, inT.shape[0], dim=0)
            _vars[gvn] = _V



        import torch.nn.functional as F
        # requirements
        # 1. need to pas a _vars object
        # 2. _vars["_symbol2order"] is a dict mapping grid variables (like x,y,t) to dimension
        # 3. _vars should include the tensors for each function; e.g. _vars["u"] = tensor
        # 4. _vars["_dh"] is a dict mapping grid vars to grid size for numerical differentiation
        class TorchEvalExp:
            def ln(x):
                return torch.ln(x)

            def log(x):
                return torch.log(x)

            def Pow(x, p):
                return torch.pow(x, p)

            def Abs(x):
                return torch.abs(x)

            def exp(x):
                return torch.exp(x)

            def sin(x):
                return torch.sin(x)

            def cos(x):
                return torch.cos(x)

            def LessThan(a,b):
                return a<=b

            def StrictLessThan(a,b):
                return a<b
            
            def StrictGreaterThan(a,b):
                return a>b

            def GreaterThan(a,b):
                return a>=b

            def Not(a):
                return ~a

            def Or(*args):
                r = args[0]
                for a in args[1:]:
                    r = r|a
                return r

            def And(*args):
                r = args[0]
                for a in args[1:]:
                    r = r&a
                return r

            def Equality(a,b):
                return a==b
                
            def Unequality(a,b):
                return a!=b
                
            def Integer(v):
                return v

            def Float(v, precision=53):
                return float(v)

            def Tuple(*args):
                return args

            def Add(*args):
                r = args[0]
                for a in args[1:]:
                    r = r + a
                return r

            def Mul(*args):
                r = args[0]
                for a in args[1:]:
                    r = r * a
                return r

            def Symbol(s):
                return str(s)

            class Function:
                def __init__(self, name):
                    self.name = name
                    self.args = []

                def __call__(self, *args):
                    for a in args:
                        assert isinstance(a, str)
                    self.args = [a for a in args]
                    return _vars[self.name]

            def Derivative(t, p):

                vnum = _vars["_symbol2order"][p[0]]
                deriv_order = p[1]
                dh = _vars["_dh"][p[0]]

                if deriv_order == 1:
                    stencil = torch.Tensor([-0.5, 0.0, 0.5]).to(t.device)
                elif deriv_order == 2:
                    stencil = torch.Tensor([1.0, -2.0, 1.0]).to(t.device)
                    dh = dh * dh
                else:
                    print("ERROR: only derivatives up to order 2 are supported")
                dim = vnum  # len(_vars["_symbol2order"])
                stencilD = torch.reshape(stencil, [1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
                var = F.pad(t, 4 * [(stencil.shape[0] - 1) // 2], "replicate")  # "constant", 0)

                output = F.conv2d(var, stencilD, padding="valid") / dh

                if dim == 0:
                    output = output[
                        :, :, :, (stencil.shape[0] - 1) // 2 : -(stencil.shape[0] - 1) // 2
                    ]
                elif dim == 1:
                    output = output[
                        :, :, (stencil.shape[0] - 1) // 2 : -(stencil.shape[0] - 1) // 2, :
                    ]
                return output

        fns = [e for e in dir(TorchEvalExp) if not e.startswith("__")]
        ctxt = {fn: getattr(TorchEvalExp, fn) for fn in fns}

        result =  eval(self.eq_srepr,{}, ctxt)

        if self.ctype == "interior":
            # Zero outer boundary        
            #result = F.pad(result[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
            if self.criteria != "":
                crit =  eval(self.criteria,{}, ctxt)
                result[~crit] = 0

        elif self.ctype == "boundary":
            # zero interior
            if self.criteria != "":
                crit =  eval(self.criteria,{}, ctxt)
                result[~crit] = 0
            else:
                T=result
                t0 = 0*T
                t0[:,:,:,-2:] = T[:,:,:,-2:]
                t0[:,:,:,:2] = T[:,:,:,:2]
                t0[:,:,-2:,:] = T[:,:,-2:,:]
                t0[:,:,:2,:] = T[:,:,:2,:]
                result = t0


        return {self.outvar: result}
## ----------------
