# Modulus Tool Chain Spec

In this document we outline the proposed functionality for the tool chain that provides the following functionality
- Creation, cloning / replicating / extending projects
- Introspection and analysis
- Training process control (start/stop, multi-stage training, history clearing, multi-gpu/node)
- Inference (server)
- Configuration initialization and validation
- Pulling examples from a central repository

The Modulus Tool Chain is collected in the command line interface `mtc`

- `mtc create <project name>` sets up a new directory with the project name
- [inside project dir] `mtc compile [stage]` will generate the execution script `train.py` for all stages or, if provided, for the chosen one. The command will also set up the `config.yaml` file inside the each `stage` directory.

- [inside project dir] `mtc show` has a number of sub-commands
    - `mtc show problem` prints the problem definition
    - `mtc show training` shows stages and the parameters used in training (batch sizes, constraint classes, loss functions, lambda weightings)

- [inside project dir] `mtc train [stage]` trains, optionally trains a stage defined by `stage` (`stage="stage1"` by default)

- [inside project dir] `mtc train-samples [stage]` trains as above, but using a pre-sampled domains (geometry) (see `mtc sample` below)

- [inside project dir] `mtc init-config` updates the configuration; given a `problem.py` file defining the problem using the `Problem` API, this command prepares the `/conf` and `training.py` with default values

- [inside project dir] `mtc start-inference-server` starts the inference REST server and prints its functionality; optional arguments
    - `--port P` the port may be specified
    - `--model-list` by default the sub-models and collection models will all be served, but this option allows to more precisely define which models to infer (e.g. derivatives of sub-models may be requested in this way, constraint equations, etc.)

- `mtc list-projects` lists available examples from remote server

- `mtc pull <example-project>` pulls a project from remote server

- `mtc configurator` run inside the project directory starts the Modulus Project Configurator, e.g. a web-based GUI

- `mtc sample` sample the domains (geometry) and save result in HDF5 format at `training/stageX/samples` (can specify which stage with `--stage stageX` option)

## Modulus Project

To facilitate managing complex projects and to capture sufficient information to help the user, a Modulus Project consists of a directory with the following structure:
- `conf/` -- a directory with configuration files (may be edited manually or using the configurator, preferred)
- `problem.py` the `Problem` object definition
- additional dirs like `output` collecting training data (logs, neural networks, opt, etc.)

## Modulus Project Configurator

This GUI-based tool is necessary due to the evolving complexity of the configuration files. The tool provides:
- Guidance on options for configurations
- Description of the config values (how they are used, full words instead of abbreviations)
- Ensures correctness (to a great but not full extent) by adhering to a well-defined schema

The multi-stage training functionality is a great motivation for this kind of tool. The functionality calls for a (potentially) complex configuration at each stage and for a non-linear inter-stage dependence: a DAG, not a line graph.

The tool helps configure:

1. Mapping the constraints to 
    - the classes that implement them
    - the arguments to these classes (e.g., `batch_size`, `lambda_weighting`, Loss function)

1. Creating the training stages 
    - add stages and configure the above for each stage (default: start with one stage)
    - define stage dependence (the Stage DAG)

1. Define the available neural networks types and configs and map those to the neural networks from the `Problem`. **Note:** this mapping is the same for all stages and cannot be changed from stage to stage.

In all cases, the `Problem` object is used to validate and draw the required choices. Defaults are provided to speed up the process.