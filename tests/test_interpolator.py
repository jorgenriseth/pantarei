
import dolfin as df
import numpy as np
from pantarei.interpolator import function_interpolator
from pantarei.mms import MMSDomain
from pantarei.timekeeper import TimeKeeper

domain = MMSDomain(10)
V = df.FunctionSpace(domain, "CG", 1)
d0 = df.Function(V)
d1 = df.Expression("x[0] < 0 ? x[0]*x[0] : 0.0", degree=2)
d1 = df.interpolate(d1, V)
d2 = df.Expression("x[0] > 0 ? x[0]*x[0] : 0.0", degree=2)
d2 = df.interpolate(d2, V)

data = [d0, d1, d2]
times = np.array([0.0, 1.0, 2.0])
d = function_interpolator(data, times)

vmin = min([ci.vector().min() for ci in data])
vmax = max([ci.vector().max() for ci in data])

u = df.Function(V, name="data")
with df.XDMFFile("data.xdmf") as xdmf:
    for ti, di in zip(times, data):
        u.assign(di)
        xdmf.write(u, ti)

u.rename("interpolated", "interpolated")
with df.XDMFFile("interpolated.xdmf") as xdmf:
    dt = 0.1
    T = 3.0
    t = 0.0
    while t <= T:
        u.assign(d(t))
        xdmf.write(u, t)
        t += dt

V = data[-1].function_space()
dt = times[-1] / 40
T = times[-1]
time = TimeKeeper(dt,T)
bdry = df.Function(V)
bcs = [df.DirichletBC(V, bdry, "on_boundary")]
u = df.TrialFunction(V)
v = df.TestFunction(V)
a = df.inner(df.grad(u), df.grad(v)) * df.dx
L = df.Constant(0.0) * v * df.dx
f = df.Function(V,name="solution")
A = df.assemble(a)
b = df.assemble(L)
xdmf = df.XDMFFile("solution.xdmf")
for t in time:
    bdry.assign(d(float(t)))
    for bc in bcs:
        bc.apply(A, b)
    df.solve(A, f.vector(), b)
    xdmf.write(f, float(t))