import warp as wp

wp.init()

## Custom Code (from problem.py)

{{custom_warp_code}}

## SDF Helpers

# subtraction
@wp.func
def op_subtract(d1: float, d2: float):
    return -wp.min(-d1, d2)

# intersection
@wp.func
def op_intersect(d1: float, d2: float):
    return wp.max(d1, d2)

# union
@wp.func
def op_union(d1: float, d2: float):
    return wp.min(d1, d2)

# signed sphere 
@wp.func
def sdf_circle(p: wp.vec2, r: float):
    return wp.length(p) - r

# signed box 
@wp.func
def sdf_rectangle(upper: wp.vec2, p: wp.vec2):

    qx = wp.abs(p[0])-upper[0]
    qy = wp.abs(p[1])-upper[1]

    e = wp.vec2(wp.max(qx, 0.0), wp.max(qy, 0.0))
    
    return wp.length(e) + wp.min( wp.max(qy, qx), 0.0)

@wp.func
def v3abs(v: wp.vec2):
    return wp.vec2(wp.abs(v[0]), wp.abs(v[1]))
##########
#
########## -- Geometry SDF wp.func's
{% for geom in geometries %}
@wp.func
def _sdf_{{geom.name}}(p: wp.vec2, {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    # type: {{geom.g.type}}
    {% if geom.g.type=="GeometryDifference" %}
    return op_subtract(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="GeometryUnion" %}
    return op_union(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="GeometryIntersection" %}
    return op_intersect(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="Rectangle" %}
    p1 = wp.vec2({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    p2 = wp.vec2({% for v in geom.g.args[1]%}float({{v}}),{% endfor %})
    side = p2-p1
    center = (p1+p2)*0.5
    {% if geom.g.rotate %}
    angle = {{geom.g.rotate[0]}}
    c = wp.cos(angle)
    s = wp.sin(angle)
    x = p[0]-center[0]
    y = p[1]-center[1]
    z = p[2]-center[2]
    p = wp.vec2(c*x +s*y+center[0], -s*x +c*y+center[1], p[2])
    {% endif %}

    sdf = sdf_rectangle((p2-p1)/2.0, p-center)
    return sdf
    {% elif geom.g.type=="Circle" %}
    c = wp.vec2({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    r = float({{geom.g.args[1]}})
    return sdf_circle(p-c, r)
    {% elif geom.g.type=="Channel2D" %}
    point_1 = wp.vec2({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    point_2 = wp.vec2({% for v in geom.g.args[1]%}float({{v}}),{% endfor %})
    dist_x = point_2[0] - point_1[0]
    dist_y = point_2[1] - point_1[1]
    center_y = point_1[1] + (dist_y) / 2.0
    y_diff = wp.abs(p[1] - center_y) - (point_2[1] - center_y)
    outside_distance = wp.sqrt(wp.max(y_diff, 0.0) ** 2.0)
    inside_distance = wp.min(y_diff, 0.0)
    sdf = (outside_distance + inside_distance)
    return sdf
    {% elif geom.g.type=="Polygon" %}
    {% if geom.g.rotate %}
    angle = {{geom.g.rotate[0]}}
    c = wp.cos(angle)
    s = wp.sin(angle)
    center = wp.vec2(0.0,0.0)
    x = p[0]#-center[0]
    y = p[1]#-center[1]
    p = wp.vec2(c*x +s*y+center[0], -s*x +c*y+center[1])
    {% endif %}      
    {% for pp in geom.g.args[0] %}
    p{{loop.index0}} = wp.vec2({% for v in pp %}float({{v}}),{% endfor %}){% endfor %}
    # Distance to line segment involves distance to line from a point
    # t = (normalized) tangent between p0 and p1
    # n = normal (perp to t)
    #
    # Solve [ t_x -n_x ] [s] = p_x - p0_x
    #       [ t_y -n_y ] [t] = p_y - p0_y
    # inverting the matrix
    sdf = wp.length(p0-p)
    {% for pp in geom.g.args[0] %}
    sdf = wp.min(wp.length(p{{loop.index0}}-p), sdf){% endfor %}
    sign = 1.0{% for pp in geom.g.args[0] %}
    {% if loop.last %}po = p0{% else %}
    po = p{{loop.index}}{% endif %}
    sdf = wp.min(wp.length(p{{loop.index0}}-p), sdf)
    tangent = po-p{{loop.index0}}
    t = wp.normalize(tangent)
    n = wp.vec2(t[1], -t[0])
    det = 1.0 #/ (-t[0]*n[1]+t[1]*n[0])
    vx = p[0] - p{{loop.index0}}[0]
    vy = p[1] - p{{loop.index0}}[1]
    s = det * ((-n[1])*vx+n[0]*vy)
    d = det * ((-t[1])*vx+t[0]*vy)
    if s>=0. and s <= wp.length(tangent):
        sdf = wp.min(sdf, wp.abs(d))
        if sdf == wp.abs(d):
            sign = wp.sign(d)
    {% endfor %}
    return sign*sdf
    {% elif geom.g.type=="GeometryCustomWarp" %}
    sdf = {{geom.g.func}}(p, {% for v in geom.g.args %}float({{v}}),{% endfor %})
    return sdf
    {% else %}
    return 0.0
    {% endif %}
{% endfor %}


{% for geom in geometries %}
@wp.kernel
def _kernel_sdf_{{geom.name}}(points: wp.array(dtype=wp.vec2), 
    sdf: wp.array(dtype=float), # return values
    {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    i = wp.tid()
    p = points[i]
    sdf[i] = _sdf_{{geom.name}}(p{{paramcallstr}})
{% endfor %}

{% for geom in geometries %}
@wp.kernel
def _kernel_sample_interior_{{geom.name}}(rand_seed: int, 
    points: wp.array(dtype=wp.vec2), 
    bbox: wp.vec4,
    {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    tid = wp.tid()
    rstate = wp.rand_init(rand_seed, tid)
    p = wp.vec2(wp.randf(rstate, bbox[0], bbox[1]), wp.randf(rstate, bbox[2], bbox[3]))
    sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
    count = int(0)
    while count < 1_000_000 and sdf>0.0:
        p = wp.vec2(wp.randf(rstate, bbox[0], bbox[1]), wp.randf(rstate, bbox[2], bbox[3]))
        sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
        count += 1
    points[tid] = p
{% endfor %}

{% for geom in geometries %}
@wp.kernel
def _kernel_sample_boundary_{{geom.name}}(rand_seed: int, 
    points: wp.array(dtype=wp.vec2), 
    bbox: wp.vec4,
    tol: float,
    {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    tid = wp.tid()
    rstate = wp.rand_init(rand_seed, tid)
    p = wp.vec2(wp.randf(rstate, bbox[0], bbox[1]), wp.randf(rstate, bbox[2], bbox[3]))
    sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
    count = int(0)
    while count < 1_000_000 and (sdf<-tol or sdf>0):
        p = wp.vec2(wp.randf(rstate, bbox[0], bbox[1]), wp.randf(rstate, bbox[2], bbox[3]))
        sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
        count += 1

    # compute gradient of the SDF using finite differences
    eps = 1.e-4

    dx = _sdf_{{geom.name}}(p + wp.vec2(eps, 0.0){{paramcallstr}}) - _sdf_{{geom.name}}(p - wp.vec2(eps, 0.0){{paramcallstr}})
    dy = _sdf_{{geom.name}}(p + wp.vec2(0.0, eps){{paramcallstr}}) - _sdf_{{geom.name}}(p - wp.vec2(0.0, eps){{paramcallstr}})

    normal = wp.normalize(wp.vec2(dx, dy))
    sdf = _sdf_{{geom.name}}(p{{paramcallstr}})

    points[tid] = p- normal*sdf
{% endfor %}


class Geometry:

    def __init__(self, dim=128):

        self.dim = dim
        self.max_verts = 10**6
        self.max_tris = 10**6

        self.sdf_return = wp.zeros(shape=(dim*10,), dtype=float)
        self.sample_points = wp.zeros(shape=(10,), dtype=wp.vec2)

        self._geoms = [{% for geom in geometries %}"{{geom.name}}",{% endfor %}]

    def list_geometries(self):
        return self._geoms.copy()

    def sdf(self, geom_name, xy{{paramstr}}):
        assert geom_name in self._geoms

        dim = xy.shape[0]
        if self.sdf_return.shape[0] < xy.shape[0]:
            self.sdf_return = wp.zeros(shape=(xy.shape[0],), dtype=float)
        kernel = globals()["_kernel_sdf_"+geom_name]
        with wp.ScopedTimer(f"SDF compute for {geom_name}"):
            wp.launch(kernel, 
            dim=dim, inputs=[wp.array(xy, dtype=wp.vec2), self.sdf_return, {% for k,v in params.items() %}{{k}}, {% endfor %}])
            wp.synchronize()
        return self.sdf_return.numpy()[:dim]

    def sample_interior(self, geom_name, n, bbox=[-2.0, 2.0, -2.0, 2.0]{{paramstr}}):
        if self.sample_points.shape[0] < n:
            self.sample_points = wp.zeros(shape=(n,), dtype=wp.vec2)
        kernel = globals()["_kernel_sample_interior_"+geom_name]
        with wp.ScopedTimer(f"Sampling interior of {geom_name}"):
            import numpy as np
            rand_seed = np.random.randint(1_000_000_000)
            bbox = wp.vec4(*bbox)
            wp.launch(kernel, 
            dim=n, inputs=[rand_seed, self.sample_points, bbox,
                    {% for k,v in params.items() %}{{k}}, {% endfor %}])
            wp.synchronize()
        
        return self.sample_points.numpy()[:n]

    def sample_boundary(self, geom_name, n, bbox=[-2.0, 2.0, -2.0, 2.0], tol=1e-2{{paramstr}}):
        if self.sample_points.shape[0] < n:
            self.sample_points = wp.zeros(shape=(n,), dtype=wp.vec2)
        kernel = globals()["_kernel_sample_boundary_"+geom_name]
        with wp.ScopedTimer(f"Sampling boundary of {geom_name}"):
            import numpy as np
            rand_seed = np.random.randint(1_000_000_000)
            bbox = wp.vec4(*bbox)
            wp.launch(kernel, 
            dim=n, inputs=[rand_seed, self.sample_points, bbox, tol,
                    {% for k,v in params.items() %}{{k}}, {% endfor %}])
            wp.synchronize()
        
        return self.sample_points.numpy()[:n]





