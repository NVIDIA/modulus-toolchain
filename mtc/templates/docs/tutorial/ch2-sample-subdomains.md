# Chapter 2: Inspecting the Point Cloud
[index](index.md)

Create a new project and put the following in `problem.py`

---
```python
from cfg import *

[x, y], [u] = p.add_neural_network(name="NN", inputs=["x", "y"], outputs=["u"])

r1 = p.Rectangle("r1", (0,0),(1,1))
r2 = p.Rectangle("r2", (0.2,0.2),(1,0.4))

geo = p.GeometryDifference("geo", r1,r2)

interior = p.add_interior_subdomain("interior", geom=geo)
boundary = p.add_boundary_subdomain("boundary", geom=geo)

p.add_constraint("wave_equation", enforce(equation=Eq(u,0), on_domain=interior))
p.add_constraint("BC", enforce(equation=Eq(u,0), on_domain=boundary))
```
---

Then initialize the configuration with `mtc init-conf` and sample the subdomains with `mtc sample`. The files for each of the constraints reside in `training/stage1/samples` and that directory should look like:
```
# ll training/stage1/samples
total 109400
drwxr-xr-x 2 root root     4096 Oct  1 20:00 ./
drwxr-xr-x 4 root root     4096 Oct  1 19:55 ../
-rw-r--r-- 1 root root 40004096 Oct  1 20:00 BC.hdf5
-rw-r--r-- 1 root root 32004096 Oct  1 20:00 wave_equation.hdf5
```

Now create a notebook at the root of the project directory and with kernel type `modulus-python`, and run the following cell:

---
```python
import matplotlib.pyplot as plt
import h5py

with h5py.File("training/stage1/samples/interior.hdf5") as f:
    print(f.keys())  
    plt.plot(f['x'], f['y'], '.')
with h5py.File("training/stage1/samples/boundary.hdf5") as f:
    print(f.keys())
    plt.plot(f['x'], f['y'], 'o')
```
---

The plot should look like this:

![p](figs/mtc-sample-example-01.png)


Now modify the `boundary` subdomain in the problem definintion by adding the criterion that `x<1` as follows:

---
```python
boundary = p.add_boundary_subdomain("boundary", geom=geo, criteria=x<1)
```
---

And run `mtc sample` followed by a rerun of the cell in the notebook. The resulting figure should remove all boundary (green) points from the right hand side.

![p](figs/mtc-sample-example-nox-1.png)

Procedure

1. Run `mtc sample` in root project directory (as always) creates HDF5 files inside `training/stage1/samples` for each subdomain.

    Note: `mtc sample --stage stageX` is the command to sample for a specific stage; by default `stage1` will be used if not specified. See `mtc sample --help` for more.


An alternative way to display all sub-domains is this

---
```python
import numpy as np
import matplotlib.pyplot as plt
import h5py, os
dpath = 'training/stage1/samples/'
n=1000 # number of points from each sub-domain
for fname in os.listdir(dpath):
    with h5py.File(dpath+fname) as f:
        idx=np.random.choice(np.arange(f['x'].shape[0]), size=n)
        idx=np.sort(idx)
        plt.plot(f['x'][:][idx], f['y'][:][idx], '.', label=fname)
        
plt.legend()
# plt.legend(ncol=1, loc=2,bbox_to_anchor=(1.0, 1))
```
---
