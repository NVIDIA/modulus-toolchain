import click
import os, sys

# def load_template(filename):
#     with open(os.path.join(os.path.dirname(__file__), filename)) as f:
#         return f.read()
MTC_ROOT = os.path.dirname(__file__)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("project-name")
def create(project_name):
    """Create new Modulus project"""
    import os, shutil

    basedir = os.path.split(__file__)[0]
    newdir = os.path.join(os.path.curdir, project_name)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
        os.makedirs(os.path.join(newdir, "conf"))
        src = os.path.join(basedir, "templates", "conf", "config.yaml")
        dst = os.path.join(newdir, "conf", "config.yaml")
        shutil.copyfile(src, dst)
        src = os.path.join(basedir, "templates", "conf", "config_PINO.yaml")
        dst = os.path.join(newdir, "conf", "config_PINO.yaml")
        shutil.copyfile(src, dst)
        src = os.path.join(basedir, "templates", "conf", "__init__.py")
        dst = os.path.join(newdir, "conf", "__init__.py")
        shutil.copyfile(src, dst)

        src = os.path.join(basedir, "templates", "configurator.ipynb")
        dst = os.path.join(newdir, "configurator.ipynb")
        shutil.copyfile(src, dst)

        os.system(f"cp -r {basedir}/templates/docs {newdir}")

        probtpl = os.path.join(basedir, "templates", "problem.py")
        with open(probtpl) as f:
            probstr = f.read().replace("%%projname%%", project_name)
        with open(os.path.join(newdir, "problem.py"), "w") as f:
            f.write(probstr)

        with open(os.path.join(basedir, "templates", "cfg.py")) as f:
            probstr = f.read().replace("%%projname%%", project_name)
        with open(os.path.join(newdir, "cfg.py"), "w") as f:
            f.write(probstr)
    else:
        print(f"error: {project_name} already exists")


def run_system_cmd(s):
    import sys, os
    from click import ClickException

    r = os.system(s)
    if r != 0:
        raise ClickException(f"system call [{s}] failed with exit code {r}")


@cli.command()
@click.argument("fname")
def hdf5_info(fname):
    "Prints HDF5 top-level keys and array size"
    import h5py

    with h5py.File(fname) as f:
        keys = list(f.keys())
        print("HDF5 file", fname)
        print("keys", keys)
        print("shape", f[keys[0]].shape)


@cli.command()
@click.argument("info_type")
@click.option(
    "--only-first-order-ufunc/--not-only-first-order-ufunc",
    default=False,
    help="ensure that only first order derivatives are taken for all unknown functions (will introduce auxiliary ufuncs if necessary)",
)
def show(info_type, only_first_order_ufunc):
    """Show information info_type= problem | training"""
    import os

    subcmds = ["problem", "training"]
    assert info_type in subcmds, f"allowed commands {subcmds}"
    if info_type in ["problem"]:

        if only_first_order_ufunc:
            run_system_cmd(
                'python -c "from problem import p; p.compile_to_firstorder(); p.pprint()"'
            )
        else:
            run_system_cmd('python -c "from problem import p; p.pprint()"')

    elif info_type == "training":
        import yaml

        def printinfo(pre, s):
            n = len("Optimize NNs")
            desc = s.split("\n")
            desc = desc[0] + "\n" + "\n".join([" " * (3 + n) + l for l in desc[1:]])
            if desc[-1] == "\n":
                desc = desc[:-1]
            print(f"[{pre.rjust(n)}]", desc)

        with open(os.path.join("conf", "config.yaml")) as f:
            conf = yaml.safe_load(f)
            tr = conf["modulus_project"]["training"]
            print("Stage DAG:", ", ".join([f"{a}->{b}" for a, b in tr["stage-dag"]]))
            print()
            for stageid, stage in tr["stages"].items():
                sdata = stage["data"]
                print(stageid, "-" * 40)

                # desc = stage["description"].split("\n")

                # print(pre, desc)
                printinfo("description", stage["description"])

                optinfo = f', lr = {sdata["optimizer"]["lr"]}'
                optinfo += ", steps < " + str(sdata["training"]["max_steps"])
                sel_opt = sdata["optimizer"]["__selected__"]
                if sel_opt == "lbfgs":
                    optinfo = f', lr = {sdata["optimizer"]["lr"]}, max_iter = {sdata["optimizer"]["max_iter"]}'

                printinfo("optimizer", sel_opt + optinfo)

                nnopt = []
                for nn_name, optimizing in sdata["Neural Networks"].items():
                    nnopt += [f"{nn_name}={optimizing}"]

                nntype = []
                for nn in conf["modulus_project"]["neural_networks"].keys():
                    nntype += [
                        f"[{'training' if sdata['Neural Networks'][nn+' Trainable'] else 'not training'}] {nn}: {sdata[nn]['__selected__']}"
                    ]

                printinfo("NNs", "\n".join(nntype))

                import sympy

                cs = [
                    f"{c} | bs={v['batch_size']:,} | weight = {sympy.sympify(v['lambda_weighting'])}"
                    for c, v in sdata["Equation Constraints"].items()
                    if v["include"]
                ]
                printinfo("constraints", "\n".join(cs))

                print("=" * 40)
    else:
        print("error: allowed subcommands are", subcmds)


@cli.command()
@click.option(
    "--target",
    default="training",
    help="one of: training (default), inference, sampler",
)
@click.option(
    "--stage", default="stage1", help="default=stage1 (ignored if target=sampler)"
)
@click.option(
    "--only-first-order-ufunc/--not-only-first-order-ufunc",
    default=False,
    help="ensure that only first order derivatives are taken for all unknown functions (will introduce auxiliary ufuncs if necessary)",
)
@click.option(
    "--constraint-opt/--no-constraint-opt",
    default=False,
    help="Optimize constraints by grouping like constraints",
)
def compile(target, stage, only_first_order_ufunc, constraint_opt):
    """Compile problem into a sampler.py, train.py, or infer.py"""

    print(f"[compile] {target} {stage}")
    s = f"'{target}'"
    sid = f"'{stage}'"
    run_system_cmd(
        f'python -c "from problem import p; p.compile(compile_type={s},stageid={sid}, only1storder={only_first_order_ufunc}, constraint_opt={constraint_opt})"'
    )


@cli.command()
@click.option("--stage", default="stage1")
@click.option(
    "--compile/--no-compile",
    default=True,
    help="problem.py is compiled into train.py by default, use this to avoid compilation",
)
def sample(stage, compile):
    """Sample and save the point cloud for each domain in an HDF5 file in the training/stageX/samples/ dir. Splits problem into sample.py and train_sampled.py. Use `mtc train --sampled` to train."""
    print(f"[train] {stage}")

    if compile:
        os.system(f"mtc compile --target sampler --stage {stage}")

    # create stage conf subdir if needed
    stagedir = os.path.join("training", stage)
    # if not os.path.exists(os.path.join({stagedir}, "conf")):
    if True:
        os.system(f"cp -r conf {stagedir}")
        os.system(f"touch {stagedir}/__init__.py")
        os.system(f"touch training/__init__.py")

    print(f"[mtc] running {stagedir}/sample.py ")
    os.system(f"cd {stagedir}; python sample.py")


@cli.command()
@click.option("--stage", default="stage1")
@click.option(
    "--compile/--no-compile",
    default=True,
    help="problem.py is compiled into train.py by default, use this to avoid compilation",
)
@click.option(
    "--sampled/--no-sampled",
    default=False,
    help="run the pre-sampled (load from disk) domains (train_sampled.py) or sample on the fly (train.py)",
)
@click.option(
    "--only-first-order-ufunc/--not-only-first-order-ufunc",
    default=False,
    help="ensure that only first order derivatives are taken for all unknown functions (will introduce auxiliary ufuncs if necessary)",
)
@click.option(
    "--constraint-opt/--no-constraint-opt",
    default=True,
    help="Optimize constraints by grouping like constraints",
)
@click.option(
    "--ngpus",
    default=1,
    help="Multi-gpu training (default=1)",
)
def train(stage, compile, sampled, only_first_order_ufunc, constraint_opt, ngpus):
    """Train models"""
    print(f"[train] {stage}")

    copt = "--constraint-opt" if constraint_opt else "--no-constraint-opt"

    if compile:
        if sampled:
            run_system_cmd(f"mtc compile --target sampler --stage {stage}")
        else:
            if only_first_order_ufunc:
                run_system_cmd(
                    f"mtc compile --target training {copt} --stage {stage} --only-first-order-ufunc"
                )
                run_system_cmd(
                    f"mtc compile --target inference --stage {stage} --only-first-order-ufunc"
                )

            else:
                run_system_cmd(f"mtc compile --target training {copt} --stage {stage}")
                run_system_cmd(f"mtc compile --target inference --stage {stage}")

    # create stage conf subdir if needed
    stagedir = os.path.join("training", stage)
    # if not os.path.exists(os.path.join({stagedir}, "conf")):
    if True:
        run_system_cmd(f"cp -r conf {stagedir}")
        run_system_cmd(f"touch {stagedir}/__init__.py")
        run_system_cmd(f"touch training/__init__.py")

    # update the target conf
    import yaml

    with open(os.path.join("conf", "config.yaml")) as f:
        conf = yaml.safe_load(f)
    stage_conf = {k: v for k, v in conf.items()}
    for k, v in conf["modulus_project"]["training"]["stages"][stage]["data"].items():
        if "__selected__" in v:
            del v["__selected__"]
        stage_conf[k] = v
    src = ""
    for s, e in conf["modulus_project"]["training"]["stage-dag"]:
        if e == stage:
            src = os.path.join("training", s, "outputs")
            run_system_cmd(f"mkdir training/{stage}/outputs")
            print("copying NN models from", s)
            run_system_cmd(
                f"cp {src}/*pth training/{stage}/outputs; rm training/{stage}/outputs/optim*"
            )
    with open(os.path.join("training", stage, "conf", "config.yaml"), "w") as f:
        yaml.safe_dump(stage_conf, f)

    python_start_str = "python"
    if ngpus > 1:
        python_start_str = f"mpirun --allow-run-as-root -np {ngpus} python"
    if sampled:
        print(f"[mtc] starting pre-sampled training session in: {stagedir}")
        if not os.path.exists(os.path.join(stagedir, "samples")):
            print("Need to sample first, run:\n mtc sample --stage", stage)
        else:
            run_system_cmd(f"cd {stagedir}; {python_start_str} train_sampled.py")
    else:
        run_cmd = f"cd {stagedir}; {python_start_str} train.py"
        print(f"[mtc] starting training session in: {stagedir} | {run_cmd}")
        run_system_cmd(run_cmd)


@cli.command()
@click.option(
    "--only-first-order-ufunc/--not-only-first-order-ufunc",
    default=False,
    help="ensure that only first order derivatives are taken for all unknown functions (will introduce auxiliary ufuncs if necessary)",
)
@click.option("--max-steps", default=1000, help="Max training steps (default=1000)")
def init_conf(only_first_order_ufunc, max_steps):
    """DESTRUCTIVE! Initialize configuration file for problem. Run every time a new constraint, sub-model, or neural network is introduced."""
    os.system(
        f'python -c "from problem import p; p.init_config(only1storder={only_first_order_ufunc}, max_steps={max_steps})"'
    )


# @cli.command()
# @click.option("--port", default=7777, help="default=7777")
# def configurator(port):
#     """Start the Modulus Project Configurator server"""
#     os.system(f"sh $MPC_PATH/start-app.sh `pwd` {port}")


# @cli.command()
# @click.option(
#     "--stage",
#     default="stage1",
#     help="Use a stage ID (like 'stage1') to target a specific stage",
# )
# @click.option("--port", default=7777, help="default=7777")
# @click.option(
#     "--compile/--no-compile",
#     default=True,
#     help="problem.py is compiled into infer.py by default, use this to avoid compilation",
# )
# def inference_server(stage, port, compile):
#     "start an inference server"

#     if compile:
#         os.system(f"mtc compile --target inference --stage {stage}")

#     # start server
#     os.system(f"cd training/{stage}; python -m mpc.rest_server.start {stage}")


@cli.command()
@click.option(
    "--static-doc-dir",
    default=os.path.join(MTC_ROOT, "docs", "static"),
    help=f"Location to start serving docs (default={MTC_ROOT})",
)
@click.option("--port", default=7777, help="default=7777")
def docs(static_doc_dir, port):
    "Modulus Simplified API Docs Server"

    print("[mtc] serving docs from", static_doc_dir)
    os.system(f"cd {static_doc_dir}; python -m http.server {port}")


@cli.command()
@click.option(
    "--stage",
    default="all",
    help="Use a stage ID (like 'stage1') to target a specific stage. Default=all and removes all training.",
)
def clean(stage):
    "DESTRUCTIVE! Remove training data for all stages (default) or for a specific stage using the optional --stage stageX"
    if stage == "all":
        os.system("rm -rf training")
    else:
        path = f"training/{stage}"
        if os.path.exists(path):
            os.system(f"rm -rf {path}")
        else:
            print(f"Stage [{stage}] does not exist or has not been used in training.")


@cli.command(help="Prints the version string")
def version():
    import os

    fname = os.path.join(os.environ["MTC_PATH"], "MTC_VERSION")
    with open(fname, "r") as f:
        print(f.read())


# @cli.command()
# def problem_to_first_order():
#     "Transforms problem to ensure that only first order derivatives of the unknown functions are used"

#     os.system(f'python -c "from problem import p; p.compile_to_firstorder()"')


if __name__ == "__main__":
    cli()
