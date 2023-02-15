# TODO

- [  ] **doc** Add an `Inference API` doc `inference.md`

- [  ] Fix problems with CFD problem set -- ask Natliia from SoftServe to do it and to contribute their OpenFoam solution as well

- [  ] **feature** include a proper (LaTex) rendering of the problem with `mtc render problem`

- [  ] **feature** Add a Warp-based geometry compilation target; e.g., `mtc compile --target geometry`

- [  ] **feature** Add support for static models that use a data array and interpolation to produce a value. E.g., `m(x,y)` would (through a transformation of `x` and `y` look into a data array and apply interpolation to produce a value). Motivation: velocity in wave equation.

- [  ] **feature** Add support for importance sampling. E.g., in defining equation constraints introduce a new way to "enforce" the constraint:
    1. Use `enfoce_uniformly` for the current version of `enforce` (and keep it?)
    1. Introduce `enforce_selectively(equation=..., on_domain=..., distribution=...)` where the `distribution` argument provides the "importance" sampling function.

- [  ] **feature** Add support for `IntegralBoundaryConstraint` -- the bulk of the work is to support sampling and compilation into `train_sampled.py`

- [  ] Update `mtc` to always check if it is inside an mtc project

- [  ] Update problem.py API documentation

- [  ] Add error message if the subdomain cannot generate all the required input variables for the equation (e.g. NN needs x and y but subdomain is only 1D)


- [  ] Add comments in compilation output based on the hints in the UI schema. E.g., when creating a neural network, explain the purpose of each parameter.

- [  ] Add Inference section to configurator GUI (to select the standard output models and to explain how to create new infer fns with derivatives, etc.)


- [  ] Add geometry
    ```
    p.get_normals(geom) -> 2 (Nx(x,y), Ny(x,y)) or 3 (Nx(x,y,z), ...)
    ```

    the geometry subsystem should also keep track of the dimensionality and issue an error if a boolean operation is performed on mismatching geometries


- `mtc show training`: 
    - display the loss function as a sum (or other aggregate) of the constraints where each constraint has the form `constraint_name: ConstraintClass(batch=X, weight=Y, npts=Z)`--note the `npts` would be equal to the number of data points in the case of data constraints. 
    - Display, for each constraint if it is going to be used. For example, if constraint C only needs NN u, but that NN is not trainable, then that constraint is invalid -- issue warning (the UI should take care of that automatically, but this is an independent check)
    - Also, add an option to this command to only display the info for a given stage.

- Add documentation for each of these: in the spec and in user doc

# DONE (?)

- [ X ] Improve `mtc show training` by printing, for each stage:
    1. The selected optimizer and relevant parameters (like max steps, learning rate for Adam...)
    1. The NNs that will be trained (optimized) vs not
    1. The constraints that will be used

- [ X ] **feature** add support for geometry from meshes (e.g. STL files)

- [ X ] Add union and intersection operations for geometries

- [ X ] Add tutorials for MTC
    1. "Hello World!" PINN (no configurator), introduces:
        1. Set up a problem (structure of `problem.py` and `mtc show problem`)
        1. Train models (`mtc init-conf`, `mtc train`, and `mtc show training`)
        1. Use trained models (inference)
    1. Inspecting the point cloud: introduces `mtc sample`
    1. Collection models (no configurator)
    1. PINN + data: introduces `add_data_constraint()`
    1. Multi-stage training: introduces the configuration UI tool


- [ X ] Make `mtc fix-conf` work without having to run `mtc configurator` ... switch command name to `mtc init-conf` 

- [ X ] Add an `mtc clean` command to remove the training and such with options to do it for a given stage

- [ X ] Improve configurator UI to toggle visibility for: the stage DAG, config ui, training plot


- [ X ] First order rewrite facility -- turn higher-order PDEs into first order derivatives only by including more unknown (auxiliary) functions


- [ X ] Add `--only-first-order-ufunc` optional to these `mtc` commands: `fix-conf`, `compile`, `train`

- [ X ] Switch to `add_constraint` and only one at a time instead of `set_constraints` -- this will allow to check for previously declared constraints (names must be different)

- [ X ] Add a call to `compile --target inference` inside the `mtc train` command 

- [ X ] Update default `problem.py` (template) to include comments outlining the structure of the file.

- [ X ] For the demo to Modulus team:
    1. `mtc create demo`
    1. copy the `problem.py` from the "bushing" example (`example2`)
    1. run `mtc fix-conf` and then configure with `mtc configurator`
        - create `stage1` with defaults (fully connected)
        - create `stage2` with Fourier Nets
    1. run `mtc train`
    1. run `mtc train --stage stage2`
    1. run `mtc compile --target inference --stage stage1` and then for `stage2`
    1. Open in notebook and load the two inferencers side-by-side; plot results.


- [ x ] Create `demoYAMLui` project to experiment with the speed of the class, etc.

- [ x ] UI: add parameters to equation constraints def:
    - `batch_size`, `alpha_weighting`, and `criteria` to the parameters for each constraint (`criteria` only for the equation constraints)
    - interior constraints get `PointwiseInteriorConstraint`
    - boundary constraints get `PointwiseBoundaryConstraint`
    - This allows users to extend the constraint by writing a custom one inside the project dir (need to add YAML schema for it)
