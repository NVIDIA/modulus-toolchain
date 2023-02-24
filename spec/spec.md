

# Proposal Overview

Inspired by tooling in modern programming languages like Rust and Go, the idea is to provide high-level APIs and tool chain with the following goals:
- [tool chain] Project management facilities 
    - Creation, cloning / replicating / extending projects
    - Introspection and analysis
    - Training process control (start/stop, multi-stage training, history clearing, multi-gpu/node)
    - Inference (server)
    - Configuration initialization and validation (e.g. a Configurator GUI)
    - Pulling examples from a central repository
- [APIs] Better separation of the following concerns
    - Defining a mathematical problem
    - Training the model
    - Using the trained model (inference)

See [spec-tool-chain.md](spec-tool-chain.md) document for details of the proposed tool chain functionality.

# API Proposal Goals

The highest level goal of this proposal is to flesh out the idea of separating the mathematical problem specification from implementation details for Pointwise PINN problems using Modulus 22.07.

The proposed high-level API separates concerns in the following way
- The `Problem` API is solely responsible for collecting the math problem definition.

- The `Training` API includes configuration files and code to launch the training of the models defined in the `Problem` API in a number of flexible ways.

- The `Inference` API facilitates using the trained models either during training (for debugging purposes) or in production.

This structure is expected to present the following benefits

- Improve readability of the code by removing "unnecessary code" (by relegating code to the lower level API -- the current one)

- Collect enough information about the problem to enable human-readable error messages and warnings

- More generally, improved debugging capabilities (e.g., pretty-printing of problem formulation with varying degrees of verbosity)

- To enable a level of "semantic analysis" that could result in the automatic adjustment of the problem formulation so as to improve convergence and accuracy.

- Single source to define both the training script and then the inference functions (including a REST inference server)


# `Problem` API 

## Summary

The idea is to wrap the existing (Modulus 22.07) API in the proposed higher level API. The basic concept is to define a `Problem` object that collects the following:

1. The input and output variables of the model and any submodels.
1. The functions that define the overall model and any submodels.
1. Any training data
1. The equations that are used to define constraints on sub-domains
1. The sub-domains defined as a geometry and any additional constraints on them and, optionally, wrapping additional parameters.


Initial Implementation
- `add_neural_network`
- `add_submodel`
- `add_interior_subdomain`
- `add_boundary_subdomain`
- `set_constraints`
- `pprint`
    - `pprint_constraints`
    - `pprint_models`
    - `pprint_nns`
- `add_data_constraint`

TODO
- `add_collection_model`

## Models and Sub-Models

A `Problem` must define at least one sub-model. A model can be defined in any of the following ways:
1. Directly as the output of a neural network
1. As a sub-model using the `problem.add_submodel(...)` API
1. As a high-level model

The automatic inference generator can be queried for any of these models. See below.

## Define a Neural Network and Obtaining Input Variables

The `problem.add_neural_network` API provides a mechanism to both define a neural network and define the input variables for the whole problem. Multiple neural networks may be defined, but all of them have to admit the same input variables.

Example
```python
(x, a, b), (u,) = p.add_neural_network(
    name="NN", nn_type="fully_connected", inputs=["x", "a", "b"], outputs=["u"]
)
```
The `nn_type` depends on the config file (the `arch` section). If the `nn_type` is unkown, the inputs are empty or incompatible with a previous definition, the outputs are empty or redefine the same function, then the tool will issue an Error or a Warning.

## Define a SubModel

The `p.add_model` API defines a named sub-model that is essentially a way to evaluate an expression. It returns a `sympy.Function` that can be used for training (with external data or inside an equation-driven constraint, a PINN).

API
```python
problem.add_submodel(name:str, expression:sympy.Expr) -> sympy.Function
```
Example
```python
g = p.add_submodel("g", u * (x-1)*(x+1)))
```

Note that this function `g(x)` is such that `g(-1)=g(1)=0`. The `u`, if defined as in the previous section, is the trainable part of the `g` model.

## Define a CollectionModel

Sometimes it is useful to define the final solution over the full `Domain` as a collection of sub-models each providing a value on their own sub-domain. The `problem.add_collection_model` API provides the mechanism to define such models.

The API
```python
problem.add_collection_model(
    name: str,
    collection: Union[{"func": existing model, "on": sympy boolean expression}]
) -> None
```

Example
```python
problem.add_collection_model(
    "T",
    [
        {"func": g_metal, "on": And(y > mny, y < mxy)},
        {"func": g_air, "on": ~And(y > mny, y < mxy)},
    ],
)
```

An even better (conceptually) definition here would be to use the `sub-domain` definitions as the constraints (selectors) for when to use which sub-model. For now, the `sympy` boolean expression is the fastest to implement...

# Using the Trained Models

The above definitions collect enough information to automatically build an `inference` function that allows us to use the trained models. The `Problem` API to create such as function is outlined here.

```python
infer_fn = problem.get_infer_fn()
```

Then, this function can be used to obtain results from the trained models. Suppose the variables are `x` and `y` and that the trained models are called `u`, `f` and `g`, then the following call
```python
{"u": np.array, "f": np.array, "g": np.array} = infer_fn(x=np.array, y=np.array)
```

We could restrict the output or even add to it by creating a different `infer_fn`

```python
# only compute one of the models
infer_f = problem.get_infer_fn(outputs=["f"])
{"f": np.array} = infer_f(x=np.array, y=np.array)

# compute derivatives of some of the models
infer_f_fx_gy = problem.get_infer_fn(outputs=["f", "f__x", "g__y"])
{"f": np.array,"f__x": np.array,"g__y": np.array} = infer_f_fx_gy(x=np.array, y=np.array)
```

By default, `outputs=None` and that results in producing an inference function that returns results for all sub-models and all collection models, but no derivatives.

This functionality is also useful for debugging and monitoring progress during training. For example, if `interior_heat_g` is a constraint defined by an equation to be satisfied on a subdomain, then the `infer` function defined as
```python
infer=problem.get_infer_fn(outputs=["interior_heat_g"])
```
will return a value that shows how close the equation is to being sastisfied (evaluate aand return the expression `eq.lhs-eq.rhs` where `eq` is the equation and `lhs`/`rhs` stand for right/left hand side of the equation).


# Complete Example: Overall structure

Define a problem as shown below (also in [example1.py](example1.py) ).
```python
from pointwise_problem import *

# SECTION 1 -- define the problem instance
p = Problem()

# SECTION 2 -- define neural networks which defines variables
(x, a, b), (u,) = p.add_neural_network(
    name="NN", nn_type="fully_connected", inputs=["x", "a", "b"], outputs=["u"]
)

# SECTION 3 -- define sub-domains (geometry+sub-domain)
x0, x1 = -1, 1
geom = Line1D(x0, x1)
interior_domain = p.add_interior_subdomain(geom=geom, 
                                           params={a: [-1, 1], b: [0, 1]})

# SECTION 4 -- define sub-models
zf = (x - x0) * (x - x1)
vf = zf + a + (b - a) * (x - x0) / 2.0
g = p.add_submodel("g", u * zf + vf)

# SECTION 5 -- define equations
Dcoeff = 1.0  # diffusion coefficient
Q = 1.0  # heat source
heat_g_eq = Eq(Dcoeff * g.diff(x, x) + Q, 0)

# SECTION 6 -- finally, set up constraints (equation-on-domain list)
p.set_constraints(
    {"interior_heat_g": enforce(equation=heat_g_eq, on_domain=interior_domain)}
)

```

Now to train the model
```python 
python -c "from example1 import p; p.train()"
```

Then use the inference functions as in the previous section.

# Use Case: Using training data

Given all the information collected by the `Problem` object, using training data may be simplified.

```python
from pointwise_problem import *
p = Problem()

## STEP 1: Define sub-models
# Start with neural networks
(x, a, b), (u,) = p.add_neural_network(
    name="NN", nn_type="fully_connected", inputs=["x", "a", "b"], outputs=["u"]
)

# Then any sub-models using the neural networks
zf = (x - x0) * (x - x1)
vf = zf + a + (b - a) * (x - x0) / 2.0
g = p.add_submodel("g", u * zf + vf)

## STEP 2: Define training data constraint
# Option 1: directly train a NN function
p.add_data_constraint(model=u, data=data)

# Option 2: train a sub-model
p.add_data_constraint(model=g, data=data)
```

# Use Case 1

Suppose we want to train a model that

```python
problem = Problem()

# define Neural Networks and obtain input/output variables
(x,a,b), (u,) = problem.add_neural_network(name="diffNN",
                               nn_type=FullyConnected, 
                               inputs=["x","a","b"], outputs=["u"])


# define equations used in constraints
zf = (x+1)*(x-1)
vf = zf + a + (b-a)*(x+1)/2.0
g = u*zf + vf

# define model
problem.set_model(g)

# define subdomains
geom = Line1D(-1,1)
interior_domain = problem.make_interior_subdomain(geom=geom, 
                                                  params={a:[-1,1], b:[0,1]})

# define constraints
D = 1.0
Q = 1.0
problem.set_constraints({
    "interior": enforce(equation=Eq(D*g.diff(x,x) - Q, 0), 
                        on_domain=interior_domain)
})

```

# Use Case 2

```python
p = Problem()

# define Neural Networks
(x,a,b,Q,airT), (u_metal,) = p.add_neural_network(name="metalNN", 
                                            nn_type="fully_connected", 
                                            inputs=["x", "y", "a", "b", "Q", "airT"], 
                                            outputs=["u_metal"])

(x,a,b,Q,airT), (u_air,) = p.add_neural_network(name="airNN", 
                                            nn_type="fully_connected", 
                                            inputs=["x", "y", "a", "b", "Q", "airT"], 
                                            outputs=["u_air"])
# define parameter ranges
params = {a:[-1,1], b:[-1,1], Q: [-1,1], airT:[-1,1]}

# define geometries and sub-domains
x0, x1 = -1,1               # domain bounds for x
y0, y1 = 0, 1               # domain bounds for y (air)
mny,mxy = 0.5-0.1, 0.5+0.1  # metal y bounds

Dmetal, Dair = 1.0, 1/1e2   # diffusion coefficients for the to media

metal_geom = Rectangle((x0, mny), (x1,mxy))
air_geom = Rectangle((x0,y0),  (x1,y1)) - metal_geom

metal_domain = p.add_interior_subdomain(geom=metal_geom, params=params)
air_domain = p.add_interior_subdomain(geom=air_geom, params=params)

metal_air_interface = p.add_boundary_subdomain(geom=metal_geom, 
                                               params=params, 
                                               criteria=Or(Eq(y,mxy), Eq(y,mny))


# define model functions
zf = (x-x0)*(x-x1)
vf = zf + a + (b-a)*(x-x0)/2.0
g_metal = u_metal*zf + vf

zf = (x-y0)*(x-y1)
g_air = u_metal*zf + airT

heat_eq_metal = Eq(Dmetal*(g_metal.diff(x,x)+g_metal.diff(y,y)+Q), 0)
heat_eq_air = Eq(Dair*(g_metal.diff(x,x)+g_metal.diff(y,y)), 0)

ma_vars = metal_air_interface.get_variables()
nx, ny = ma_vars['normal_x'], ma_vars['normal_y']
interface_dirichlet = Eq(g_air, g_metal)
interface_neumann = Eq(g_air.diff(x)*nx+g_air.diff(y)*ny, 
                       g_metal.diff(x)*nx+g_metal.diff(y)*ny)


# finally, define constraints
p.set_constraints({
    "interior_metal": enforce(equation=heat_eq_metal, 
                              on_domain=metal_domain),
    "interior_air": enforce(equation=heat_eq_air, 
                              on_domain=metal_air),
    "interface_air_metal_continuity": enforce(equation=interface_dirichlet, 
                              on_domain=metal_air_interface),
    "interior_air_metal_flux": enforce(equation=interface_neumann, 
                              on_domain=metal_air_interface),                    
})

# define the model (to automate inference)
problem.set_model([{"func":g_metal, "on":metal_domain}, 
                   {"func":g_air, "on":air_domain}])

```

# Train and Infer from a Single Source

```bash
# To train the models
python pinn train problem_script.py

# To start the inference server
python pinn inference-server problem_script.py
```


# `Training` API

# `Inference` API

The inference server implements the following GET endpoints

- `/info` -- returns the problem definition a JSON dict with keys:
    - `problem`: string from `problem.pprint()` 
    - `training`: string with training details
    - `variables`: list of input variable names
    - `models`: list of models (functions) available for inference

Requesting data (model values given input variable assignments) is done with the following POST request
- endpoint `/infer`
- the body must include a JSON dict with the following entries:
    - `models`: list of requested models (must be a subset of the list returned by `/info` - see above)
    - an array of n values for each of the input variables returned by `/info` -- each array must contain the same number of values