from cfg import *

# Select Problem Type
p = PINN
# p = FNO # uncomment to select a PINO/FNO problem type
# -------------------


# Suggested structure (PINN)
#
# 1. Define problem variables and unknown functions; e.g.,
#    [x, y], [u] = p.add_neural_network(name="NN", inputs=["x", "y"], outputs=["u"])
#
# 2. Define geometries; e.g.,
#    geo = p.Line1D("geom", 0,1)
#
# 3. Define sub-domains (geometries plus non-geometric variables goe here); e.g.
#
# interior = p.add_interior_subdomain("interior",
#                                     geom=geo,
#                                     params={t:(0,2*L)})
# boundary = p.add_boundary_subdomain("boundary",
#                                     geom=geo,
#                                     params={t:(0,2*L)})
#
# 4. (Optionally) Sub-models; e.g,
#    g_air = p.add_submodel("g_air", u_air * zf + airT)
#
#
# 5. Define constraints; e.g.,
#
#     wave_eq = Eq(u.diff(t, 2), (c**2 * u.diff(x)).diff(x))
#     p.add_constraint("wave_equation", enforce(equation=wave_eq, on_domain=interior))
#     p.add_data_constraint(name="calculated", model=u, data_fname=hdf_fname)
#
#
# 6. (optionally) Define piecewise-models (collecting submodels)
#
# p.set_model(
#     "T",
#     [
#         {"func": g_metal, "on": And(y > mny, y < mxy)},
#         {"func": g_air, "on": ~And(y > mny, y < mxy)},
#     ],
# )

# Suggested structure (FNO)
#
# 1. Define the discrete spatial domain (the extent, the grid, and grid spacing)
#    x, y, dx, dy = p.set_grid_domain(N=100, extent={"x": (0.0, 1.0), "y": (0.0, 1.0)})
#
#    Note that the spatial variables are defined in this way (x and y are sympy.Symbol) and the
#    grid spacing is now computed based on the number of grid points N and returned for each variable.
#
# 2. Define problem input and output functions; e.g.,
#    [K, ],[X,Y],  [U] = p.add_neural_network(name="NN", inputs=["K"], outputs=["U"])
#
# 3. If a data driven FNO, add input/output distribution
#     ddata = p.add_distribution(name="Ddata", {"K": Kvals, "U": Uvals})
#
# 4. If PINO, an input distribution may be sufficient (i.e. with no output values)
#     dinput = p.add_distribution(name="Dinputs", {"K": Kvals})
#
# 5. If data-driven, add a data constraint
#     p.add_data_constraint("data", over=ddata) # input is a distribution that has input/output entries
#
# 6. If PINO, add an interior and boundary equation constraint (interior/boundary grid points)
#     p.add_interior_constraint("heat eq", equation=Eq(U.diff(x,x)+U.diff(y, y), 1), over=dinput)
#     p.add_boundary_constraint("bdry", equation=Eq(U=0), over=dinput)
