# MTC Tutorial

[Chapter 0: A Conceptual Framework for PINNs and Modulus](ch0-pinn-conceptual-framework.md)

[Chapter 1: A "Hello world!" PINN](ch1-hello-world.md)
- Introduces the `mtc create` command to start a new project
- Introduces the structure of problem definition in a `problem.py` file
- Introduces `mtc show problem` to obtain a quick view of the problem
- Introduces `mtc init-conf`: the automatic configuration initialization
- Introduces `mtc show training` to show a summary of the training configuration
- Introduces `mtc train` to start training the model(s)
- Demonstrates how to use the trained model in a Jupyter notebook (or script)


[Chapter 2: Inspecting the Point Cloud](ch2-sample-subdomains.md)
- Introduces `mtc sample`: the facility to sample subdomains and save the results in HDF5 files
- Introduces `mtc train --sampled` the facility to use those HDF5 files instead of sampling at the beginning of training (the default)

[Chapter 3: Sub-Models and Piecewise Models](ch3-piecewise-models.md)
- Introduces sub-models; example using them for exact boundary conditions
- Introduces the concept of piecewise models; example: solves a diffusion heat transport problem in 1D with two subdomains with very different diffusion coefficient


[Chapter 4: Adding Data Constraints](ch4-data-constraints.md)
- Introduces `add_data_constraint` and shows how to build a PINN constrained by data
- Inversion for $Q(x)$ using the constraint $T_{xx}(x) + Q(x)=0$ and data for $T(x)$ over $x\in(0,1)$ interval.

[Chapter 5: Multi-stage Training](ch5-multi-stage-training.md)
- Introduces the configuration UI tool
- Builds a two-stage pipeline where one network is trained first, and then the second one is trained in the next stage.
- Adds a third stage to train both pre-trained networks but with LBFGS optimizer instead of Adam
- Creates a separate training branch to explore training in a single stage but with Fourier Net architecture for all NNs instead of the default Fully Connected architecture.

[Chapter 6: Automatic "Semantic Analysis"](ch6-semantic-analysis.md)
- Introduces experimental advanced features like the automatic facility to rewrite the problem in an equivalent form but ensuring that only first derivatives of NN functions are used.
- Thoughts on future facilities

