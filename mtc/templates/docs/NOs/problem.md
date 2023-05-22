

# Elements of a `problem.py` -- for FNOs

The problem definition of this type of model seeks to create a map that computes one or more target unknown functions `U1, U2, ...` from one or more known grid functions `G1, G2, ...`. In symbols
$$
(G_1, G_2, ...) \mapsto (U_1, U_2, ...)
$$
where each of these functions is defined on a grid in $n$ dimensions; e.g., in 2 dimensions
 $$
 G_i(x,y): [x_0, x_1]\times[y_0, y_1] \to \mathbb{R} \\
 U_i(x,y): [x_0, x_1]\times[y_0, y_1] \to \mathbb{R} \\
 $$

A problem is defined using the `p` object imported from the `cfg.py` local module. The object provides facilities to:

1. Define the grid on which the model will be trained
1. Define models with "unknown functions" -- the neural networks
1. Define data distributions
1. Define equation-based constraints (using the sub-models)
1. Define data-based constraints

## Grid Declaration

The following declares grid variables and specifies the grid extent:
```python
x, t, dx, dt = p.set_grid_domain(N=N, extent={"x": (0.0, 10.0), "t": (0,1)})
```
The `x` and `t` are `sympy` symbols that may be used later to, for example, define derivatives (see below).

The `dx` and `dt` are the step sizes (floats) for each of the variables--these are the grid point spacing values.

## Neural Network Declaration / Variables

The functions to be learned or to be used as input are defined with the following statement:

```python
[V, ], [X, T], [P] = p.add_neural_network(name="NN", inputs=["V", ], outputs=["P"])
```

Note that this uses the grid declaration already explained. So, the return values are:
- the grid function `V` (a `sympy.Function` object -- `V(x,t)` in this case)
- the grid variable functions `X` and `T` of type `sympy.Function` -- these are useful to define constraints as we'll see below
- the unknown grid function `P` of type `sympy.Function`

## Grid Function (inputs) Distributions

In order to train the model to compute an unknown function `P(x,t)` using a specified grid function `V(x,t)` we need to generate a distribution of possible such functions `V`. We use the grid declaration to specify the requirements for how these grid function distributions are to be provided. For example, if there are $N$ grid points for each of the two dimensions, then the distribution is defined by a 3-dimensional array of size `(nsamples, N,N)`. 

```python
Vd = np.ones(shape=(16,N,N))*vmag
fname = os.path.join(os.getcwd(), "inputs.hdf5")
p.to_hdf(fname, {"V": Vd})
d1 = p.add_distribution("InputOnly: V in (1,4)", hdf5_file=fname)
```
This creates the appropriate array, saves it to disk, and then declares the distribution variable `d1` using the API call `p.add_distribution(name, hdf5_file=str)`. Note that if the the data is already on disk only the last line is needed.

## Equation Constraints

Now the problem is almost fully specified, except for the physical constraints that the unknown functions must respect.

### Dirichlet Boundary Conditions
```python
p.add_dirichlet_constraint("sides1", on=P, equation=Eq(P, 0), at=Eq(x,0))
```
### Neumann Boundary Conditions
```python
p.add_neumann_constraint("top-du_dy", on=P, equation=Eq(P.diff(t), vmag*sp.sin(X)), at=Eq(t,0))
```
Note that `sp.sin` is the `simpy.Function` representing the trig function. Also note that here we are using the grid function variable `X` (upper case) instead of the grid variable `x` (lower case)

### Interior Constraints (PDEs)
```python
eq = Eq(P.diff(t,t), V**2 * P.diff(x,x))
p.add_interior_constraint("wave eq", equation=eq, over=d1, on=P)
```
Note that here we are using the grid variable `x` (lower case) to specify the second derivative with respect to $x$ of the unknown function $P(x,t)$. 

We can also specify diffeo-integral equations with the same `p.add_interior_constraint` API call. For example,
```python
eq = Eq( T*P.diff(t)+P, sp.Integral(P,x) )
p.add_interior_constraint("wave eq", equation=eq, over=d1, on=P)
```
This is a valid expression that will use a simple quadrature to approximate the integral over the full extent of the variable `t` (in this example $t\in[0,1]$ as defined in the `p.set_grid_domain` call above)
$$
T{\left(x,t \right)} \frac{\partial}{\partial x} P{\left(x,t \right)} + P{\left(x,t \right)} = \int P{\left(x,t \right)} \, dt 
$$

### Code for full example
Start by creating a new project with
```
mtc create fno-api-example
cd fno-api-example
```

The following should be saved in the `problem.py` file
```python
from cfg import *

# Select Problem Type
p = FNO
# -------------------
N = 64*2
L = 3.14
tL = L*2
x, t, dx, dt = p.set_grid_domain(N=N, extent={"x": (0.0, L), "t": (0.0, tL)})

[V, ], [X, T], [P] = p.add_neural_network(name="NN", inputs=["V", ], outputs=["P"])

## make training distribution
import numpy as np
vmag = 1.0 #1/2.0
Vd = np.ones(shape=(16,N,N))*vmag

fname = os.path.join(os.getcwd(), "inputs.hdf5")
p.to_hdf(fname, {"V": Vd})
d1 = p.add_distribution("InputOnly: V in (1,4)", hdf5_file=fname)
##
import sympy as sp
p.add_dirichlet_constraint("sides1", on=P, equation=Eq(P, 0), at=Eq(x,0))
p.add_dirichlet_constraint("sides2", on=P, equation=Eq(P, 0), at=Eq(x,L))

p.add_dirichlet_constraint("top", on=P, equation=Eq(P, sp.sin(X)), at=Eq(t,0))
p.add_neumann_constraint("top-dudy", on=P, equation=Eq(P.diff(t), vmag*sp.sin(X)), at=Eq(t,0))

eq = Eq(P.diff(t,t), V**2 * P.diff(x,x))
p.add_interior_constraint("wave eq", equation=eq, over=d1, on=P)
```

Then train with
```
mtc init-conf
mtc clean
mtc train
```

And then, inside the same project directory, create a notebook, insert the following code in the first cell, and then run the cell.
```python
import training.stage1.infer as infer1
import numpy as np

ns = 0

xextent=infer1.info['grid']['vars']['x']['extent']
textent=infer1.info['grid']['vars']['t']['extent']
ter=list(reversed(list(textent)))
extent = list(xextent)+ter
import h5py
import matplotlib.pyplot as plt
N = infer1.info['grid']['N']
with h5py.File("inputs.hdf5") as f:
    V = f['V'][ns:ns+1,:,:]

U = infer1.infer(V)['P']
U=U.reshape(N,N)
plt.imshow(U, extent=extent)
plt.xlabel("x")
plt.ylabel("t")
```