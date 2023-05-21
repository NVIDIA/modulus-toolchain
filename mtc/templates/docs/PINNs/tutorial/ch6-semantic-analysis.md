# Chapter 6: Automatic "Semantic Analysis"
[index](index.md)

Given this `problem.py`

---
```python
from cfg import *

[x, a], [u] = p.add_neural_network(name="NN", inputs=["x", "a"], outputs=["u"])

params = {a: (0,1)}
geom = p.Line1D("geom", 0,1)
interior = p.add_interior_subdomain("interior", geom=geom, params=params)
bdry = p.add_boundary_subdomain("bdry", geom=geom, params=params)

diff_eq = Eq(u.diff(x,2) + (x-a), 0)
p.add_constraint("diffusion", enforce(equation=diff_eq, on_domain=interior))
p.add_constraint("bdry", enforce(equation=Eq(u,0), on_domain=bdry))
```
---

Original problem needs 2nd derivatives of neural network (output of `mtc show problem`)

```
Neural Networks
--------------------------------------------------------------------------------
|  NN = NeuralNetwork(inputs=[x, a],
|                     outputs=[u])
-------------------------------------------------------------------------------- 

Models
--------------------------------------------------------------------------------
|---
-------------------------------------------------------------------------------- 

Constraints
--------------------------------------------------------------------------------
|   (.) = (x, a)
|
|   diffusion:  -a + x + Derivative(u(.), (x, 2)) = 0
|   [on int. subdomain] interior
|
|   bdry:  u(.) = 0
|   [on bdry subdomain] bdry
|
| Data ----------------------------------------
--------------------------------------------------------------------------------
```

Automatically transformed problem only requires 1st derivatives of neural networks (output of `mtc show problem --only-first-order-ufunc`)

```
Neural Networks
--------------------------------------------------------------------------------
|  NN = NeuralNetwork(inputs=[x, a],
|                     outputs=[u])
|  d_dx1_uNN = NeuralNetwork(inputs=[x, a],
|                            outputs=[d_dx1_u])
-------------------------------------------------------------------------------- 

Models
--------------------------------------------------------------------------------
|---
-------------------------------------------------------------------------------- 

Constraints
--------------------------------------------------------------------------------
|   (.) = (x, a)
|
|   diffusion:  -a + x + Derivative(d_dx1_u(.), x) = 0
|   [on int. subdomain] interior
|
|   bdry:  u(.) = 0
|   [on bdry subdomain] bdry
|
|   diffusion1:  Derivative(u(.), x) = d_dx1_u(.)
|   [on int. subdomain] interior
|
| Data ----------------------------------------
--------------------------------------------------------------------------------
```

This also works if the derivatives are on a submodel which uses a neural network as in the case of the exact boundary condition formulation of the problem above

---
```python
from cfg import *

[x, a], [u] = p.add_neural_network(name="NN", inputs=["x", 'a'], outputs=["u"])

geom = p.Line1D("geom", 0,1)
interior = p.add_interior_subdomain("interior", geom=geom, params={a:(0,1)})

g = p.add_submodel("g", x*(x-1)*u)
diff_eq = Eq(g.diff(x,2) + (x-a), 0)
p.add_constraint("diffusion", enforce(equation=diff_eq, on_domain=interior))
```
---

Output of `mtc show problem`

```
Neural Networks
--------------------------------------------------------------------------------
|  NN = NeuralNetwork(inputs=[x, a],
|                     outputs=[u])
-------------------------------------------------------------------------------- 

Models
--------------------------------------------------------------------------------
|  g : x*(x - 1)*u(x, a)
|---
-------------------------------------------------------------------------------- 

Constraints
--------------------------------------------------------------------------------
|   (.) = (x, a)
|
|   diffusion:  -a + x + Derivative(g(.), (x, 2)) = 0
|   [on int. subdomain] interior
|
| Data ----------------------------------------
--------------------------------------------------------------------------------
```

Output of `mtc show problem --only-first-order-ufunc`
```
Neural Networks
--------------------------------------------------------------------------------
|  NN = NeuralNetwork(inputs=[x, a],
|                     outputs=[u])
|  d_dx1_uNN = NeuralNetwork(inputs=[x, a],
|                            outputs=[d_dx1_u])
-------------------------------------------------------------------------------- 

Models
--------------------------------------------------------------------------------
|  g : x*(x - 1)*u(x, a)
|---
-------------------------------------------------------------------------------- 

Constraints
--------------------------------------------------------------------------------
|   (.) = (x, a)
|
|   diffusion:  -a + x*(x - 1)*Derivative(d_dx1_u(.), x) + 2*x*d_dx1_u(.) + x + 2*(x - 1)*d_dx1_u(.) + 2*u(.) = 0
|   [on int. subdomain] interior
|
|   diffusion1:  Derivative(u(.), x) = d_dx1_u(.)
|   [on int. subdomain] interior
|
| Data ----------------------------------------
-------------------------------------------------------------------------------- 
```

## Training using the transformed problem

First, the configuration has to be initialized with the transformed problem (because the transformation introduces new NNs and constraints); run

```
mtc init-conf --only-first-order-ufunc
```

Then, to train, use the same flag and run

```
mtc train --only-first-order-ufunc
```