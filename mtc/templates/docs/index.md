# Modulus Toolchain Documentation

## Physics-Informed Neural Networks (PINNs)

[PINN API documentation](PINNs/problem.md)

[PINN Tutorial](PINNs/tutorial/index.md)

## Neural Operators (only FNOs for now -- experimental)

[FNO API Documentation](NOs/problem.md)

## Graph Neural Networks (planned)

Nothing yet.

# Toolchain

![c](PINNs/compiler-toolchain.svg)

```
$ mtc
Usage: python -m mtc.mtc [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  clean      DESTRUCTIVE! Remove training data for all stages (default)...
  compile    Compile problem into a sampler.py, train.py, or infer.py
  create     Create new Modulus project
  docs       Modulus Simplified API Docs Server
  hdf5-info  Prints HDF5 top-level keys and array size
  init-conf  DESTRUCTIVE! Initialize configuration file for problem.
  sample     Sample and save the point cloud for each domain in an HDF5...
  show       Show information info_type= problem | training
  train      Train models
  version    Prints the version string
```

Also, for each command running with the `--help` switch will provide additional information. For example, for PINNs:
```
$ mtc train --help
Usage: python -m mtc.mtc train [OPTIONS]

  Train models

Options:
  --stage TEXT
  --compile / --no-compile        problem.py is compiled into train.py by
                                  default, use this to avoid compilation
  --sampled / --no-sampled        run the pre-sampled (load from disk) domains
                                  (train_sampled.py) or sample on the fly
                                  (train.py)
  --only-first-order-ufunc / --not-only-first-order-ufunc
                                  ensure that only first order derivatives are
                                  taken for all unknown functions (will
                                  introduce auxiliary ufuncs if necessary)
  --constraint-opt / --no-constraint-opt
                                  Optimize constraints by grouping like
                                  constraints
  --ngpus INTEGER                 Multi-gpu training (default=1)
  --help                          Show this message and exit.
```
