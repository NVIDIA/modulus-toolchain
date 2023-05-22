class FNOEquationConstraint(torch.nn.Module):
    "Custom Equation Constraint"

    def __init__(self, vars, gridvarfuncs, eq_srepr, outvar, onFunc, dirichlet_gen_conds, dirichlet_conds, neumann_conds, ctype="interior", criteria=None):
        "ctype in ['interior', 'boundary']"
        ctypes=['interior', 'boundary']
        assert ctype in ctypes, f"Invalid ctype={ctype} -- Constraint type must be one of {ctypes}"
        super().__init__()
        self.vars = vars
        self.gridvarfuncs = gridvarfuncs
        self.eq_srepr = eq_srepr
        self.outvar = outvar
        self.ctype=ctype

        self.onFunc = onFunc
        # the format of the Dirichlet and Neumann conditions is a list of:
        # (var_id, offset, expr_srepr)
        # var_id: the variable number (e.g. 0 for x, ...)
        # offset: in grid point id units (usually 0, 1 or -1, -2)
        # expr_srepr: the string to evaluate to get the value

        self.dirichlet_conds=dirichlet_conds
        self.dirichlet_gen_conds=dirichlet_gen_conds
        self.neumann_conds=neumann_conds

        self.criteria=criteria

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _vars = {
            "_dh4id": { {% for vn,v in grid.vars.items() %} {{v.var_id}} :{{v.delta}}, {% endfor %} }, #{"x": 1.0, "y": 1.0},
            "_dh": { {% for vn,v in grid.vars.items() %}"{{vn}}":{{v.delta}}, {% endfor %} }, #{"x": 1.0, "y": 1.0},
            "_symbol2order": { {% for vn,v in grid.vars.items() %}"{{vn}}":{{v.var_id}}, {% endfor %} },
        }
        for v in self.vars:
            _vars[v] = input_var[v].clone()
            inT = input_var[v]

        # mgargs = [ np.arange(gv['extent'][0], gv['extent'][1], gv['delta']) for gvn, gv in self.gridvarfuncs.items()]
        mgargs = [ np.linspace(gv['extent'][0], gv['extent'][1], inT.shape[-1]) for gvn, gv in self.gridvarfuncs.items()]

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
                return float(v)

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
                dim = (vnum+1)%2  # len(_vars["_symbol2order"])
                stencilD = torch.reshape(stencil, [1, 1] + dim * [1] + [-1] + (1 - dim) * [1])
                # var = F.pad(t, 4 * [(stencil.shape[0] - 1) // 2], "replicate")  # "constant", 0)

                var=t
                output = F.conv2d(var, stencilD, padding="same") / dh
                return output
            def Integral(t, p):
                dim = int(_vars["_symbol2order"][p[0]])
                dh = _vars["_dh"][p[0]]
                # nT=torch.zeros_like(T)
                if dim == 0:
                    sT = dh*torch.sum(t,2+dim, keepdim=True)#.shape            
                    # nT[0,0,:,:]=sT[0,0,0,:]
                    return sT.repeat(1,1,t.shape[2],1)
                elif dim == 1:
                    sT = dh*torch.sum(t,2+dim, keepdim=True)#.shape
                    return sT.repeat(1,1,1,t.shape[3])
                return nT


        fns = [e for e in dir(TorchEvalExp) if not e.startswith("__")]
        ctxt = {fn: getattr(TorchEvalExp, fn) for fn in fns}

        # Select the target tensor to prepare with boundary info
        T = _vars[self.onFunc]

        # First, evaluate all Dirichlet General conditions
        for expr_srepr, at_srepr in self.dirichlet_gen_conds:
            Tc = eval(expr_srepr,{}, ctxt)
            if isinstance(Tc, float):
                Tc = 0*T + Tc
            atCond = eval(at_srepr,{}, ctxt)
            T[atCond] = Tc[atCond]

        # First, evaluate all Dirichlet conditions
        for var_id, offset, expr_srepr in self.dirichlet_conds:
            Tc = eval(expr_srepr,{}, ctxt)
            if isinstance(Tc, float):
                Tc = 0*T + Tc
            if var_id == 0:
                T[:,:,:, offset] = Tc[:,:,:, offset]
            else:
                T[:,:,offset,:] = Tc[:,:,offset,:]

        # Then, evaluate all Neumann conditions
        
        for var_id, offset, expr_srepr in self.neumann_conds:
            dh = _vars["_dh4id"][var_id]
            Tc = -eval(expr_srepr,{}, ctxt)
            if isinstance(Tc, float):
                Tc = 0*T + Tc
            off = -1
            if offset > 1:
                off = 1
                dh = -dh
            if var_id == 0:
                T[:,:,1:-1, offset] = T[:,:,1:-1, offset+off] + off*Tc[:,:,1:-1, offset]*dh
            else:
                T[:,:, offset,1:-1] = T[:,:,offset+off,1:-1] + off*Tc[:,:,offset,1:-1]*dh

        result =  eval(self.eq_srepr,{}, ctxt)

        ## must ignore boundary in loss function when using 3-point stencils
        result[:,:,0,:]=0
        result[:,:,-1,:]=0
        result[:,:,:,0]=0
        result[:,:,:,-1]=0

        return {self.outvar: result}
## ----------------
