import warp as wp

wp.init()

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
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r

# signed box 
@wp.func
def sdf_box(upper: wp.vec3, p: wp.vec3):

    qx = wp.abs(p[0])-upper[0]
    qy = wp.abs(p[1])-upper[1]
    qz = wp.abs(p[2])-upper[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))
    
    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)

@wp.func
def v3abs(v: wp.vec3):
    return wp.vec3(wp.abs(v[0]), wp.abs(v[1]),wp.abs(v[2]))
@wp.func
def v3max(v: wp.vec3):
    return wp.max(wp.max(v[0], v[1]),v[2])
@wp.func
def v3min(v: wp.vec3):
    return wp.min(wp.min(v[0], v[1]),v[2])
##########
#
########## -- Geometry SDF wp.func's
{% for geom in geometries %}
@wp.func
def _sdf_{{geom.name}}(p: wp.vec3, {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    # type: {{geom.g.type}}
    {% if geom.g.type=="GeometryDifference" %}
    return op_subtract(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="GeometryUnion" %}
    return op_union(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="GeometryIntersection" %}
    return op_intersect(_sdf_{{geom.g.g}}(p{{paramcallstr}}), _sdf_{{geom.g.og}}(p{{paramcallstr}}))
    {% elif geom.g.type=="Cylinder" %}
    center = wp.vec3({% for v in geom.g.args[0] %}float({{v}}),{% endfor %})
    radius = {{geom.g.args[1]}}
    height = {{geom.g.args[2]}}
    {% if geom.g.rotate %}
    angle = {{geom.g.rotate[0]}}
    c = wp.cos(angle)
    s = wp.sin(angle)
    x = p[0]-center[0]
    y = p[1]-center[1]
    z = p[2]-center[2]
    {% if geom.g.rotate[1]=='x' %}
    p = wp.vec3(p[0], c*y +s*z+center[1], -s*y +c*z+center[2])
    {% elif geom.g.rotate[1]=='y' %}
    p = wp.vec3(c*x +s*z+center[0], p[1], -s*x +c*z+center[2])
    {% else %}
    p = wp.vec3(c*x +s*y+center[0], -s*x +c*y+center[1], p[2])
    {% endif %}
    {% endif %}   
    r_dist = wp.sqrt((p[0] - center[0])*(p[0] - center[0]) + (p[1] - center[1])*(p[1] - center[1]))
    z_dist = wp.abs(p[2] - center[2])
    outside = wp.sqrt(wp.min(0.0, radius-r_dist)**2.0 + wp.min(0.0, 0.5*height-z_dist)**2.0)
    inside = -wp.min(wp.abs(wp.min(0.0, r_dist-radius)), wp.abs(wp.min(0.0, z_dist-0.5*height)))
    sdf = (outside+inside)
    return sdf
    {% elif geom.g.type=="ModBox" %}
    p1 = wp.vec3({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    p2 = wp.vec3({% for v in geom.g.args[1]%}float({{v}}),{% endfor %})
    side = p2-p1
    center = (p1+p2)/2.0
    {% if geom.g.rotate %}
    angle = {{geom.g.rotate[0]}}
    c = wp.cos(angle)
    s = wp.sin(angle)
    x = p[0]-center[0]
    y = p[1]-center[1]
    z = p[2]-center[2]
    {% if geom.g.rotate[1]=='x' %}
    p = wp.vec3(p[0], c*y +s*z+center[1], -s*y +c*z+center[2])
    {% elif geom.g.rotate[1]=='y' %}
    p = wp.vec3(c*x +s*z+center[0], p[1], -s*x +c*z+center[2])
    {% else %}
    p = wp.vec3(c*x +s*y+center[0], -s*x +c*y+center[1], p[2])
    {% endif %}
    {% endif %}

    c_dist = v3abs(p - center) - 0.5 * side

    ov = wp.vec3(wp.max(c_dist[0], 0.), wp.max(c_dist[1], 0.), wp.max(c_dist[2], 0.))
    # outside = wp.min(c_dist[0], 0.)**2. + wp.min(c_dist[1], 0.)**2. + wp.min(c_dist[2], 0.)**2.
    outside = wp.length(ov) #wp.sqrt(outside)
    inside = wp.min(v3max(c_dist), 0.0)
    sdf = (outside + inside)
    # sdf = inside #outside
    return sdf
    # upper = (p2-p1)*0.5
    # return sdf_box(upper, p-upper-p1)
    {% elif geom.g.type=="Box" %}
    p1 = wp.vec3({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    p2 = wp.vec3({% for v in geom.g.args[1]%}float({{v}}),{% endfor %})
    side = p2-p1
    center = (p1+p2)*0.5
    {% if geom.g.rotate %}
    angle = {{geom.g.rotate[0]}}
    c = wp.cos(angle)
    s = wp.sin(angle)
    x = p[0]-center[0]
    y = p[1]-center[1]
    z = p[2]-center[2]
    {% if geom.g.rotate[1]=='x' %}
    p = wp.vec3(p[0], c*y +s*z+center[1], -s*y +c*z+center[2])
    {% elif geom.g.rotate[1]=='y' %}
    p = wp.vec3(c*x +s*z+center[0], p[1], -s*x +c*z+center[2])
    {% else %}
    p = wp.vec3(c*x +s*y+center[0], -s*x +c*y+center[1], p[2])
    {% endif %}
    {% endif %}

    sdf = sdf_box((p2-p1)/2.0, p-center)
    return sdf
    # upper = (p2-p1)*0.5
    # return sdf_box(upper, p-upper-p1)
    {% elif geom.g.type=="Sphere" %}
    c = wp.vec3({% for v in geom.g.args[0]%}float({{v}}),{% endfor %})
    r = float({{geom.g.args[1]}})
    return sdf_sphere(p-c, r)
    {% else %}
    return 0.0
    {% endif %}
{% endfor %}


{% for geom in geometries %}
@wp.kernel
def _kernel_{{geom.name}}(field: wp.array3d(dtype=float), 
        dim: float, 
        scale: float,
        {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    i, j, k = wp.tid()
    p = wp.vec3(float(i), float(j), float(k))
    p = (p/dim - wp.vec3(1.0,1.0,1.0)*0.5)*scale

    sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
    field[i,j,k] = sdf
{% endfor %}

{% for geom in geometries %}
@wp.kernel
def _kernel_adjust_points_{{geom.name}}(points: wp.array(dtype=wp.vec3), 
        dim: float, 
        scale: float,
        {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    i = wp.tid()
    p = points[i]
    p = (p/dim - wp.vec3(1.0,1.0,1.0)*0.5)*scale

    eps = 1.e-5

    # compute gradient of the SDF using finite differences
    dx = _sdf_{{geom.name}}(p + wp.vec3(eps, 0.0, 0.0){{paramcallstr}}) - _sdf_{{geom.name}}(p - wp.vec3(eps, 0.0, 0.0){{paramcallstr}})
    dy = _sdf_{{geom.name}}(p + wp.vec3(0.0, eps, 0.0){{paramcallstr}}) - _sdf_{{geom.name}}(p - wp.vec3(0.0, eps, 0.0){{paramcallstr}})
    dz = _sdf_{{geom.name}}(p + wp.vec3(0.0, 0.0, eps){{paramcallstr}}) - _sdf_{{geom.name}}(p - wp.vec3(0.0, 0.0, eps){{paramcallstr}})

    normal = wp.normalize(wp.vec3(dx, dy, dz))
    sdf = _sdf_{{geom.name}}(p{{paramcallstr}})
    points[i] = p - normal*sdf
{% endfor %}


{% for geom in geometries %}
@wp.kernel
def _kernel_sdf_{{geom.name}}(points: wp.array(dtype=wp.vec3), 
    sdf: wp.array(dtype=float), # return values
    {% for k,v in params.items() %}{{k}}: float, {% endfor %}):
    i = wp.tid()
    p = points[i]
    sdf[i] = _sdf_{{geom.name}}(p{{paramcallstr}})
{% endfor %}

class Geometry:

    def __init__(self, dim=128):

        self.dim = dim
        self.max_verts = 10**6
        self.max_tris = 10**6

        self.time = 0.0

        self.field = wp.zeros(shape=(self.dim, self.dim, self.dim), dtype=float)
        self.sdf_return = wp.zeros(shape=(dim*10,), dtype=float)
        
        self.iso = wp.MarchingCubes(nx=self.dim,
                                    ny=self.dim,
                                    nz=self.dim,
                                    max_verts=self.max_verts,
                                    max_tris=self.max_tris)

        self._geoms = [{% for geom in geometries %}"{{geom.name}}",{% endfor %}]

    def list_geometries(self):
        return self._geoms.copy()

    def update(self):
        pass

    def render_sdf(self, geom_name, scale=3.0{{paramstr}}):
        assert geom_name in self._geoms

        kernel = globals()["_kernel_"+geom_name]
        self.scale = scale
        with wp.ScopedTimer(f"Updated SDF volume with {geom_name}"):
            wp.launch(kernel, 
            dim=self.field.shape, inputs=[self.field, self.dim, scale, {% for k,v in params.items() %}{{k}}, {% endfor %}])
            self._last_kernel = geom_name

    def adjust_points(self{{paramstr}}):
        kernel = globals()["_kernel_adjust_points_"+self._last_kernel]
        with wp.ScopedTimer(f"Adjusted mesh points from {self._last_kernel}"):
            wp.launch(kernel, 
            dim=self.iso.verts.shape,
            inputs=[self.iso.verts, self.dim, self.scale, {% for k,v in params.items() %}{{k}}, {% endfor %}])
            verts = self.iso.verts.numpy()#(self.iso.verts.numpy()/float(self.dim) - 0.5)*self.scale
            indices=self.iso.indices.numpy()
        print(f"geometry: {self._last_kernel} | {verts.shape[0]:,} verts | {indices.shape[0]:,} tris")
        return {"verts": verts, "indices":indices}

    def sdf(self, geom_name, xyz{{paramstr}}):
        assert geom_name in self._geoms

        dim = xyz.shape[0]
        if self.sdf_return.shape[0] < xyz.shape[0]:
            self.sdf_return = wp.zeros(shape=(xyz.shape[0],), dtype=float)
        kernel = globals()["_kernel_sdf_"+geom_name]
        with wp.ScopedTimer(f"SDF compute for {geom_name}"):
            wp.launch(kernel, 
            dim=dim, inputs=[wp.array(xyz, dtype=wp.vec3), self.sdf_return, {% for k,v in params.items() %}{{k}}, {% endfor %}])
        return self.sdf_return.numpy()[:dim]

    def get_mesh_data(self):
        with wp.ScopedTimer(f"get_mesh_data [geometry: {self._last_kernel}]"):
            self.iso.surface(field=self.field, threshold=0)
            verts = (self.iso.verts.numpy()/float(self.dim) - 0.5)*self.scale
            indices=self.iso.indices.numpy()
        print(f"geometry: {self._last_kernel} | {verts.shape[0]:,} verts | {indices.shape[0]:,} tris")
        return {"verts": verts, "indices":indices}


