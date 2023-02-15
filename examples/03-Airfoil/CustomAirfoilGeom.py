import numpy as np
from modulus.geometry.primitives_2d import Channel2D, Rectangle


import matplotlib.pyplot as plt

import warp as wp
import numpy as np

import os

wp.init()

# Naca implementation modified from https://stackoverflow.com/questions/31815041/plotting-a-naca-4-series-airfoil
# https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
# @wp.func
# def camber_line(x, m, p, c):
#     cl = []
#     for xi in x:
#         cond_1 = Heaviside(xi, 0) * Heaviside((c * p) - xi, 0)
#         cond_2 = Heaviside(-xi, 0) + Heaviside(xi - (c * p), 0)
#         v_1 = m * (xi / p ** 2) * (2.0 * p - (xi / c))
#         v_2 = m * ((c - xi) / (1 - p) ** 2) * (1.0 + (xi / c) - 2.0 * p)
#         cl.append(cond_1 * v_1 + cond_2 * v_2)
#     return cl


@wp.func
def Heaviside(a: float, h: float):
    if a == 0:
        return h
    elif a < 0:
        return float(0)
    elif a > 0:
        return float(1)


@wp.func
def camber_line(xi: float, m: float, p: float, c: float):
    z = float(0.0)
    cond_1 = Heaviside(xi, z) * Heaviside((c * p) - xi, z)
    cond_2 = Heaviside(-xi, z) + Heaviside(xi - (c * p), z)
    v_1 = m * (xi / p**2.0) * (2.0 * p - (xi / c))
    v_2 = m * ((c - xi) / (1.0 - p) ** 2.0) * (1.0 + (xi / c) - 2.0 * p)
    r = cond_1 * v_1 + cond_2 * v_2
    return r


@wp.func
def thickness(xi: float, t: float, c: float):
    term1 = 0.2969 * (sqrt(xi / c))
    term2 = -0.1260 * (xi / c)
    term3 = -0.3516 * (xi / c) ** 2.0
    term4 = 0.2843 * (xi / c) ** 3.0
    term5 = -0.1015 * (xi / c) ** 4.0
    r = 5.0 * t * c * (term1 + term2 + term3 + term4 + term5)
    return r


# signed sphere
@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r


# signed box
@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):

    qx = wp.abs(p[0]) - upper[0]
    qy = wp.abs(p[1]) - upper[1]
    qz = wp.abs(p[2]) - upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def sdf_plane(p: wp.vec3, plane: wp.vec4):
    return plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]


# union
@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)


# subtraction
@wp.func
def op_subtract(d1: float, d2: float):
    return wp.max(-d1, d2)


# intersection
@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)


# simple scene
@wp.func
def sdf(p: wp.vec3):

    # intersection of two spheres
    sphere_1 = wp.vec3(0.0, 0.0, 0.0)
    sphere_2 = wp.vec3(0.0, 0.75, 0.0)

    d = op_subtract(sdf_sphere(p - sphere_2, 0.75), sdf_box(wp.vec3(1.0, 0.5, 0.5), p))
    # sdf_sphere(p + sphere_1, 1.0))

    # ground plane
    d = op_union(d, sdf_plane(p, wp.vec4(0.0, 1.0, 0.0, 1.0)))

    return d


@wp.func
def normal(p: wp.vec3):

    eps = 1.0e-5

    # compute gradient of the SDF using finite differences
    dx = sdf(p + wp.vec3(eps, 0.0, 0.0)) - sdf(p - wp.vec3(eps, 0.0, 0.0))
    dy = sdf(p + wp.vec3(0.0, eps, 0.0)) - sdf(p - wp.vec3(0.0, eps, 0.0))
    dz = sdf(p + wp.vec3(0.0, 0.0, eps)) - sdf(p - wp.vec3(0.0, 0.0, eps))

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def shadow(ro: wp.vec3, rd: wp.vec3):

    t = float(0.0)
    s = float(1.0)

    for i in range(64):

        d = sdf(ro + t * rd)
        t = t + wp.clamp(d, 0.0001, 2.0)

        h = wp.clamp(4.0 * d / t, 0.0, 1.0)
        s = wp.min(s, h * h * (3.0 - 2.0 * h))

        if t > 8.0:
            return 1.0

    return s


@wp.func
def channel_sdf(x: float, y: float, ytop: float, ybot: float):
    if wp.abs(ytop - y) < wp.abs(ybot - y):
        return y - ytop
    else:
        return ybot - y


@wp.func
def naca_sdf(N: int, m: float, p: float, t: float, c: float, tx: float, ty: float):

    dx = float(1.0) / float(N)

    d = float(1e9)
    for pxid in range(N):
        xi = float(pxid) * dx
        xi_1 = float(pxid) * dx + dx

        cli = camber_line(xi, m, p, c)
        cli_1 = camber_line(xi_1, m, p, c)

        px = xi
        py = cli
        yt = thickness(xi, t, c)
        pd = wp.sqrt((px - tx) * (px - tx) + (py - ty) * (py - ty))
        d = wp.min(pd - yt, d)

    return d


@wp.func
def naca_bdry(N: int, m: float, p: float, t: float, c: float, tx: float, ty: float):

    dx = float(1.0) / float(N)

    d = float(1e9)
    d0 = float(1e9)
    xr = float(1e9)
    yr = float(1e9)
    bd = float(1e9)
    for pxid in range(N):
        xi = float(pxid) * dx
        xi_1 = float(pxid) * dx + dx

        cli = camber_line(xi, m, p, c)
        cli_1 = camber_line(xi_1, m, p, c)

        px = xi
        py = cli
        yt = thickness(xi, t, c)
        pd = wp.sqrt((px - tx) * (px - tx) + (py - ty) * (py - ty))
        d = wp.min(pd - yt, d)

        if d < d0:
            xr = px
            yr = py
            d0 = d
            bd = yt

            if pd < 0.1:
                bd = 0.0

    nx = (tx - xr) / (d0 + bd)
    ny = (ty - yr) / (d0 + bd)
    xx = bd * nx + xr
    yy = bd * ny + yr

    l = 1.0
    if bd == 0:
        nx = 0.0
        ny = 0.0

    else:
        eps = 1e-5
    #         d = naca_sdf(N, m,p,t,c, xx, yy)
    #         dx = naca_sdf(N, m,p,t,c, xx+eps, yy)
    #         dy = naca_sdf(N, m,p,t,c, xx, yy+eps)

    #         nx = dx-d
    #         ny = dy-d
    #         l = wp.sqrt(nx*nx+ny*ny)

    return wp.vec4(xx, yy, nx / l, ny / l)


@wp.kernel
def draw(
    N: int,
    m: float,
    p: float,
    t: float,
    c: float,
    width: int,
    height: int,
    pixels: wp.array(dtype=float),
):
    tid = wp.tid()
    x = float(tid % width) / float(width) * 2.0
    x = x - 0.5
    y = float(tid // width) / float(height) * 2.0
    y = y - 1.0

    d = naca_sdf(N, m, p, t, c, x, y)
    cd = channel_sdf(x, y, 0.5, -0.5)
    pixels[tid] = op_subtract(d, cd)


@wp.kernel
def sample_interior(
    rand_seed: int,
    N: int,
    m0: float,
    m1: float,
    p: float,
    t0: float,
    t1: float,
    c: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rot0: float,
    rot1: float,
    xa: wp.array(dtype=float),
    ya: wp.array(dtype=float),
    sdf: wp.array(dtype=float),
    sdf_x: wp.array(dtype=float),
    sdf_y: wp.array(dtype=float),
    par_rot: wp.array(dtype=float),
    par_t: wp.array(dtype=float),
    par_m: wp.array(dtype=float),
):
    tid = wp.tid()
    rstate = wp.rand_init(rand_seed, tid)

    x = wp.randf(rstate, x0, x1)
    y = wp.randf(rstate, y0, y1)

    # apply rotation
    rot = -wp.randf(rstate, rot0, rot1)
    par_rot[tid] = rot
    xx = wp.cos(rot) * x - wp.sin(rot) * y
    yy = wp.sin(rot) * x + wp.cos(rot) * y

    xa[tid] = x
    ya[tid] = y

    m = wp.randf(rstate, m0, m1)
    par_m[tid] = m
    t = wp.randf(rstate, t0, t1)
    par_t[tid] = t

    d = naca_sdf(N, m, p, t, c, xx, yy)
    cd = channel_sdf(x, y, y1, y0)
    d = op_subtract(d, cd)
    sdf[tid] = d

    eps = 1e-5
    xx = wp.cos(rot) * (x + eps) - wp.sin(rot) * y
    yy = wp.sin(rot) * (x + eps) + wp.cos(rot) * y
    cd = channel_sdf(x + eps, y, y1, y0)
    dx = naca_sdf(N, m, p, t, c, xx, yy)
    dx = op_subtract(dx, cd)

    xx = wp.cos(rot) * x - wp.sin(rot) * (y + eps)
    yy = wp.sin(rot) * x + wp.cos(rot) * (y + eps)
    cd = channel_sdf(x, y + eps, y1, y0)
    dy = naca_sdf(N, m, p, t, c, xx, yy)
    dy = op_subtract(dy, cd)

    nx = dx - d
    ny = dy - d
    l = wp.sqrt(nx * nx + ny * ny)

    sdf_x[tid] = -nx / l
    sdf_y[tid] = -ny / l


@wp.kernel
def sample_boundary(
    rand_seed: int,
    N: int,
    m0: float,
    m1: float,
    p: float,
    t0: float,
    t1: float,
    c: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rot0: float,
    rot1: float,
    xa: wp.array(dtype=float),
    ya: wp.array(dtype=float),
    nx: wp.array(dtype=float),
    ny: wp.array(dtype=float),
    par_rot: wp.array(dtype=float),
    par_t: wp.array(dtype=float),
    par_m: wp.array(dtype=float),
):
    tid = wp.tid()
    rstate = wp.rand_init(rand_seed, tid)

    x = wp.randf(rstate, x0, x1)
    y = wp.randf(rstate, y0, y1)

    m = wp.randf(rstate, m0, m1)
    par_m[tid] = m

    t = wp.randf(rstate, t0, t1)
    par_t[tid] = t

    ret = naca_bdry(N, m, p, t, c, x, y)
    x = ret[0]
    y = ret[1]

    # apply rotation to points
    rot = wp.randf(rstate, rot0, rot1)
    par_rot[tid] = rot

    xx = wp.cos(rot) * x - wp.sin(rot) * y
    yy = wp.sin(rot) * x + wp.cos(rot) * y

    # rotate the normals as well
    xa[tid] = xx
    ya[tid] = yy

    x = ret[2]
    y = ret[3]
    nxx = wp.cos(rot) * x - wp.sin(rot) * y
    nyy = wp.sin(rot) * x + wp.cos(rot) * y
    nx[tid] = nxx
    ny[tid] = nyy

    # sdf[tid] = op_subtract(d,cd)


class Example:
    def __init__(self, nr=1000000):

        self.nr = nr
        self.xa = wp.zeros(nr, dtype=float)
        self.ya = wp.zeros(nr, dtype=float)
        self.nxa = wp.zeros(nr, dtype=float)
        self.nya = wp.zeros(nr, dtype=float)
        self.sdf = wp.zeros(nr, dtype=float)
        self.sdfx = wp.zeros(nr, dtype=float)
        self.sdfy = wp.zeros(nr, dtype=float)

        self.par_rot = wp.zeros(nr, dtype=float)
        self.par_t = wp.zeros(nr, dtype=float)
        self.par_m = wp.zeros(nr, dtype=float)

    def render(self, m, p, t, c, N, is_live=False):

        with wp.ScopedTimer(f"airfoil SDF [{self.width*self.height:,} pts]"):

            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[N, m, p, t, c, self.width, self.height, self.pixels],
            )

            wp.synchronize_device()

        parr = self.pixels.numpy().reshape((self.height, self.width))
        return parr

    def sample_interior(self, x0, y0, x1, y1, m0, m1, p, t0, t1, c, N, rot0, rot1):
        import time

        with wp.ScopedTimer(f"Sample Interior [{self.nr:,} pts]"):
            rand_seed = int(time.time())
            wp.launch(
                kernel=sample_interior,
                dim=self.nr,
                inputs=[
                    rand_seed,
                    N,
                    m0,
                    m1,
                    p,
                    t0,
                    t1,
                    c,
                    x0,
                    y0,
                    x1,
                    y1,
                    rot0,
                    rot1,
                    self.xa,
                    self.ya,
                    self.sdf,
                    self.sdfx,
                    self.sdfy,
                    self.par_rot,
                    self.par_t,
                    self.par_m,
                ],
            )

            wp.synchronize_device()

        sdf = -self.sdf.numpy()
        sel = sdf > 0
        rot = self.par_rot.numpy()[sel]
        t = self.par_t.numpy()[sel]
        m = self.par_m.numpy()[sel]
        return (
            self.xa.numpy()[sel],
            self.ya.numpy()[sel],
            sdf[sel],
            self.sdfx.numpy()[sel],
            self.sdfy.numpy()[sel],
            rot,
            t,
            m,
        )

    def sample_boundary(self, x0, y0, x1, y1, m0, m1, p, t0, t1, c, N, rot0, rot1):
        import time

        with wp.ScopedTimer(f"Sample Boundary [{self.nr:,} pts]"):
            rand_seed = int(time.time())

            wp.launch(
                kernel=sample_boundary,
                dim=self.nr,
                inputs=[
                    rand_seed,
                    N,
                    m0,
                    m1,
                    p,
                    t0,
                    t1,
                    c,
                    x0,
                    y0,
                    x1,
                    y1,
                    rot0,
                    rot1,
                    self.xa,
                    self.ya,
                    self.nxa,
                    self.nya,
                    self.par_rot,
                    self.par_t,
                    self.par_m,
                ],
            )

            wp.synchronize_device()

        nx = self.nxa.numpy()
        ny = self.nya.numpy()
        sel = (nx != 0) & (ny != 0)

        rot = self.par_rot.numpy()[sel]
        t = self.par_t.numpy()[sel]
        m = self.par_m.numpy()[sel]
        return self.xa.numpy()[sel], self.ya.numpy()[sel], nx[sel], ny[sel], rot, t, m


from modulus.geometry.parameterization import Parameterization, Parameter


class AirfoilInChannel:
    def __init__(self, ll, ur, params={}, include_channel_boundary=False):
        self.x0, self.y0 = ll
        self.x1, self.y1 = ur

        self.ch = Channel2D(ll, ur)
        self.include_channel_boundary = include_channel_boundary

        self.params = Parameterization(
            {
                Parameter("rot"): float(0.0),
                Parameter("m"): float(0.02),
                Parameter("t"): float(0.12),
            }
        )

    def _get_param_min_max(self, param, parameterization=None):
        if parameterization is None:
            parameterization = {}
        else:
            parameterization = parameterization.param_ranges

        ps = param
        rot = parameterization.get(ps, self.params.param_ranges[ps])
        if isinstance(rot, float):
            rot0, rot1 = rot, rot
        else:
            rot0, rot1 = min(rot), max(rot)
        return rot0, rot1

    def sample_boundary(
        self,
        nr_points: int,
        criteria=None,
        parameterization=None,
        quasirandom: bool = False,
    ):
        sb = self.ch.sample_boundary(
            nr_points, criteria=criteria, parameterization=parameterization
        )

        ps = Parameter("rot")
        rot0, rot1 = self._get_param_min_max(ps, parameterization)

        ps = Parameter("m")
        m0, m1 = self._get_param_min_max(ps, parameterization)

        ps = Parameter("t")
        t0, t1 = self._get_param_min_max(ps, parameterization)

        xa = np.zeros((nr_points, 1))
        ya = np.zeros((nr_points, 1))
        nx = np.zeros((nr_points, 1))
        ny = np.zeros((nr_points, 1))
        par_rot = np.zeros((nr_points, 1))
        par_t = np.zeros((nr_points, 1))
        par_m = np.zeros((nr_points, 1))

        gotn = 0

        while gotn < nr_points:
            e = Example(nr=nr_points)
            m = 0.02 * 1
            p = 0.4
            t = 0.12
            c = 1.0

            N = 501
            # m0, m1 = m*9, m*9
            # t0, t1 = t*1.9, t*1.9
            # rot1, rot0= -0.31, -0.31
            x, y, nxt, nyt, rott, tt, mm = e.sample_boundary(
                self.x0,
                self.y0,
                self.x1,
                self.y1,
                m0,
                m1,
                p,
                t0,
                t1,
                c,
                int(N),
                rot0,
                rot1,
            )

            e = min(len(x), nr_points - gotn)
            if e > 0:
                xa[gotn : gotn + e] = x[:e].reshape(-1, 1)
                ya[gotn : gotn + e] = y[:e].reshape(-1, 1)
                nx[gotn : gotn + e] = nxt[:e].reshape(-1, 1)
                ny[gotn : gotn + e] = nyt[:e].reshape(-1, 1)
                par_rot[gotn : gotn + e] = rott[:e].reshape(-1, 1)
                par_t[gotn : gotn + e] = tt[:e].reshape(-1, 1)
                par_m[gotn : gotn + e] = mm[:e].reshape(-1, 1)

                gotn += e

        if self.include_channel_boundary:
            idx = np.random.choice(np.arange(2 * nr_points), nr_points)
            xa = np.vstack([xa, sb["x"]])[idx]
            ya = np.vstack([ya, sb["y"]])[idx]
            nx = np.vstack([-nx, sb["normal_x"]])[idx]
            ny = np.vstack([-ny, sb["normal_y"]])[idx]
            return {
                "x": xa,
                "y": ya,
                "normal_x": -nx,
                "normal_y": -ny,
                "rot": par_rot,
                "t": par_t,
                "m": par_m,
                "area": sb["area"],
            }
        else:
            return {
                "x": xa,
                "y": ya,
                "normal_x": -nx,
                "normal_y": -ny,
                "rot": par_rot,
                "t": par_t,
                "m": par_m,
                "area": sb["area"],
            }

    def sample_interior(
        self,
        nr_points: int,
        bounds=None,
        criteria=None,
        parameterization=None,
        compute_sdf_derivatives: bool = False,
        quasirandom: bool = False,
    ):
        si = self.ch.sample_interior(
            nr_points, criteria=criteria, parameterization=parameterization
        )

        ps = Parameter("rot")
        rot0, rot1 = self._get_param_min_max(ps, parameterization)

        ps = Parameter("m")
        m0, m1 = self._get_param_min_max(ps, parameterization)

        ps = Parameter("t")
        t0, t1 = self._get_param_min_max(ps, parameterization)

        xa = np.zeros((nr_points, 1))
        ya = np.zeros((nr_points, 1))
        sdf = np.zeros((nr_points, 1))
        sdf__x = np.zeros((nr_points, 1))
        sdf__y = np.zeros((nr_points, 1))
        par_rot = np.zeros((nr_points, 1))
        par_t = np.zeros((nr_points, 1))
        par_m = np.zeros((nr_points, 1))

        gotn = 0

        while gotn < nr_points:
            e = Example(nr=nr_points)
            m = 0.02 * 1
            p = 0.4
            t = 0.12
            c = 1.0

            N = 501
            # m0, m1 = m*3, m*3
            # t0, t1 = t*1.9, t*1.9
            # rot1, rot0= -0.31, -0.931
            x, y, sdft, sdfx, sdfy, rott, tt, mm = e.sample_interior(
                self.x0,
                self.y0,
                self.x1,
                self.y1,
                m0,
                m1,
                p,
                t0,
                t1,
                c,
                int(N),
                rot0,
                rot1,
            )

            e = min(len(x), nr_points - gotn)
            xa[gotn : gotn + e] = x[:e].reshape(-1, 1)
            ya[gotn : gotn + e] = y[:e].reshape(-1, 1)
            sdf[gotn : gotn + e] = sdft[:e].reshape(-1, 1)
            sdf__x[gotn : gotn + e] = sdfx[:e].reshape(-1, 1)
            sdf__y[gotn : gotn + e] = sdfy[:e].reshape(-1, 1)
            par_rot[gotn : gotn + e] = rott[:e].reshape(-1, 1)
            par_t[gotn : gotn + e] = tt[:e].reshape(-1, 1)
            par_m[gotn : gotn + e] = mm[:e].reshape(-1, 1)

            gotn += e

        return {
            "x": xa,
            "y": ya,
            "sdf": sdf,
            "rot": par_rot,
            "t": par_t,
            "m": par_m,
            "sdf__x": sdf__x,
            "sdf__y": sdf__y,
            "area": si["area"],
        }

    @property
    def dims(self):
        return ["x", "y"]
