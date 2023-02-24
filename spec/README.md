# Modulus Simplified API Proposal

## Description

See the [spec](spec.md) document for more.

The highest level goal of this proposal is to flesh out the idea of separating the mathematical problem specification from implementation details for Pointwise PINN problems using Modulus 22.07. We propose to achieve this by adding a higher-level API that wraps the existing one.

For example:
- Use a `Problem` object to define the problem mathematically and without any implementation or training details
- Use a `TrainingPlan` object to define complex training plans (e.g. start with use Adam for X steps then use LBFGS for Y steps; or in multi-network setups train one, then the other, etc.)
- Use the hydra configuration to provide implementation, training, and inference details (like batch size, learning rates, etc.)

This separation of concerns is expected to present the following benefits

- **Readability** Improve readability of the code by removing "unnecessary code" (e.g., no need to think about a computational graph when defining the mathematical problem)

- **Error/Warning Messages** Collect enough information about the problem to enable human-readable error messages and warnings

- **Debugging** More generally, improved debugging capabilities (e.g., pretty-printing of problem formulation with varying degrees of verbosity)

- **Code Generation** To enable a level of "semantic analysis" that could result in the automatic adjustment of the problem formulation so as to improve convergence and accuracy.

- **Consistent Training&Inference** Single source to define both the training script and then the inference functions (including a REST inference server)

## Design Principles/Goals

1. Think of the primary user as a scientist first and developer second
1. Keep math problem definition in one API and separate from implementation, training and inference APIs.
1. Collect enough information to help the developer (scientist)
1. Aim for representational simplicity... but without sacrificing expressiveness


## Outline

The proposed high-level API splits concerns in the following way
- The `Problem` API is solely responsible for collecting the math problem definition.

- The `Training` API includes configuration files and code to launch the training of the models defined in the `Problem` API in a number of flexible ways.

- The `Inference` API facilitates using the trained models either during training (for debugging purposes) or in production.

## TODO

- [Spec] Describe what `TrainingPlan` should do and how it interacts with config file

- [Functionality] Add a sample implementation of `TrainingPlan`

- [Error/Warning] Add checks to make sure domains do not overlap and so multiple constraints are attempted on the same non-trivial subdomain

- [Spec] Improve `train` vs `infer` vs `info` functionality

- [Tests] Add more unit tests

- [Functionality] Add more architectures -- automatically from the config file

- [Spec] Figure out how to specify `lambda_weightings` (part of `TrainingPlan` ?)

- [Error/Warning] Make sure that the main model uses submodels that have been defined

## Comments from Praveen N.

Selected comments from here [Specs_for_code_improvements.docx](https://nam11.safelinks.protection.outlook.com/ap/w-59584e83/?url=https%3A%2F%2Fnvidia-my.sharepoint.com%2F%3Aw%3A%2Fr%2Fpersonal%2Fsotta_nvidia_com%2F_layouts%2F15%2Fdoc2.aspx%3Fsourcedoc%3D%257BC13FC7AF-9BC7-4BF3-9258-7111FB20A425%257D%26file%3DSpecs_for_code_improvements.docx%26action%3Ddefault%26mobileredirect%3Dtrue%26cid%3D400f4edf-6d98-4b57-ad5a-affc3fc5fe5b&data=05%7C01%7Cpdimitrov%40nvidia.com%7C2dc3e6aa55434152473d08da8b6d8b6a%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C637975599847875760%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=WVdUmnQSV%2FQyOaP32GV1C1hdnuoPQjw4excqSjsSzUk%3D&reserved=0) 

[6] Do we have a way to use our own logging in conjunction with the built-in one? For example, if I want to print the contributions from different loss terms that form the full loss. It enables us to see which loss contributions are converging etc.

    Having the full problem specified in one place allows you to do this and also any overall checks (requiring knowledge of all coupled constraints). Not implemented now, could be added. How would you suggest to extend the Spec? Perhaps a `stats` command line instruction?

[7] More control over the train loop. Currently, there is no way to access the optimizer. For example, one does first order optimization schemes like Adam far from the solution and as we get closer to the solution, one switches to second order schemes like LBFGS. 

    Having all the information in one place should enable a `training plan` to be implemented. Need to define a way to specify that plan -- should be separate from the problem definition. Perhaps a `TrainingPlan` object could take a `Problem` object...

    This use case can extend to other situations where multi-stage training is needed. For example, a "curriculum learning" type where 

## Installation

Assumes Modulus 22.07 is installed.

For example, you may load a container and mount the root dir of this repo. Here is one way (run in the top level of this repo)

```
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`/:/tests -p 7777:7777 -it --rm nvcr.io/nvidia/modulus/modulus:22.07
```
The port forwarding is to enable the REST server (see below) to connect to the world outside the container.

## Usage

See the [Examples](Examples.md) for a more detailed version of this section.

Create a test directory by running
```
mkdir test_dir
cp -r conf pointwise_problem.py example2.py rest_server.py test_dir/
cd test_dir
```
To train the model, run the following inside the `test_dir`
```
python -c "from example2 import p; p.train()"
```

The same problem definiton (the `Problem p` object) can be used to define an inference function
```python
infer_fn = p.get_infer_fn()
```
which makes it easy to implement a [REST server](rest_server.py)
```
python rest_server.py
```
See below for an example on how to query sample the REST server

Run the following (at any point: before or after training) to get information about the problem
```
python -c "from example3 import p; p.pprint()"
```
Should yield
```

Neural Networks
--------------------------------------------------------------------------------
|  metal_airNN = fully_connected(inputs=[x, y, a, b, Q, airT],
|                                outputs=[u_metal, u_air])
-------------------------------------------------------------------------------- 

Models
--------------------------------------------------------------------------------
|  g_metal : a + 0.5*(-a + b)*(x + 1) + (x - 1)*(x + 1)*u_metal + (x - 1)*(x + 1)
|  g_air : airT + y*(y - 1)*u_air
|---
| T(x,y,a,b,Q,airT) = 
|            g_metal  if  (y > 0.4) & (y < 0.6)
|            g_air  if  ~((y > 0.4) & (y < 0.6))
-------------------------------------------------------------------------------- 

Constraints
--------------------------------------------------------------------------------
|  interior_metal : -Q - 1.0*g_metal__x__x - 1.0*g_metal__y__y
|  interior_air : 0.01*g_air__x__x + 0.01*g_air__y__y
|  interface_air_metal_continuity : g_air - g_metal
|  interior_air_metal_flux : 0.01*normal_x*g_air__x - 1.0*normal_x*g_metal__x + 0.01*normal_y*g_air__y - 1.0*normal_y*g_metal__y
|
Data --
-------------------------------------------------------------------------------- 
```

And the server may be queried from a remote client (inside a Jupyter notebook in this case)
![modulus](modulus-REST-client-jupyter-notebook.png)




## Authors and acknowledgment
Pavel Dimitrov pdimitrov@nvidia.com

## License
For open source projects, say how it is licensed.

