import subprocess
from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt 

from pantarei.io.fenicsstorage import FenicsStorage


def store_function():
    comm = df.MPI.comm_world
    domain = df.RectangleMesh(comm, df.Point(-1, -1), df.Point(1, 1), 10, 10)
    element = df.FiniteElement("Lagrange", domain.ufl_cell(), 2)
    V = df.FunctionSpace(domain, element)
    p_expr = df.Expression("-(pow(x[0], 2)+pow(x[1], 2))", degree=2, domain=domain)
    v_expr = df.grad(p_expr)
    
    W = df.FunctionSpace(domain, df.VectorElement(element, dim=2))
    v = df.project(v_expr, W)
    file = FenicsStorage("test.h5", "w")
    file.write_function(v, "velocity")
    file.close()


def solve_convection_diffusion_stationary():
    readfile = FenicsStorage("test.h5", "r")
    domain = readfile.read_domain()
    vel = readfile.read_function("velocity", domain)
    readfile.close()
    del readfile
    
    element = df.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = df.FunctionSpace(domain, element)

    from dolfin import grad, inner
    dx = df.Measure("dx", domain=domain)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    D = 1.0
    a = -inner(-D*grad(u) + u * vel, grad(v)) * dx  # type: ignore
    L = 1 * v * dx
    bcs = df.DirichletBC(V, df.Constant(0.0), "on_boundary")

    u = df.Function(V, name="stationary")
    df.solve(a == L, u, bcs=bcs)

    storefile =  FenicsStorage("test.h5", "a")
    storefile.write_function(u, "stationary")
    storefile.close()


def solve_convection_diffusion_time_dependent():
    readfile = FenicsStorage("test.h5", "r")
    domain = readfile.read_domain()
    vel = readfile.read_function("velocity", domain)
    readfile.close()
    del readfile
    
    element = df.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = df.FunctionSpace(domain, element)

    from dolfin import grad, inner
    dx = df.Measure("dx", domain=domain)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    u0 = df.Function(V, name="init")

    dt = 0.5
    T = 1.0
    D = 1.0

    a = (u*v - dt * inner(-D*grad(u) + u * vel, grad(v))) * dx  # type: ignore
    L = (u0 + dt * df.Constant(1.0)) * v * dx 
    bdry_val = df.Constant(0.0)
    bcs = df.DirichletBC(V, bdry_val, "on_boundary")

    u = df.Function(V, name="concentration")
    u.assign(u0)
    storefile =  FenicsStorage("test.h5", "a")
    storefile.write_checkpoint(u, "concentration", 0.0)
    t = 0.0
    while t <= T - dt + df.DOLFIN_EPS:
        bdry_val.assign(0.3*t)
        t += dt
        df.solve(a == L, u, bcs=bcs)
        u0.assign(u)
        storefile.write_checkpoint(u, "concentration", t)
    storefile.close()

def read_and_plot():
    readfile = FenicsStorage("test.h5", "r")
    vel = readfile.read_function("velocity")
    u = readfile.read_function("stationary")

    plt.figure()
    c = df.plot(u)
    df.plot(vel)
    plt.colorbar(c)
    plt.title("Stationary solution")
    plt.show(block=False)

    ncounts = readfile.hdf.attributes("/concentration")["count"]
    time = readfile.read_timevector("concentration")
    u = readfile.read_function("concentration")
    for idx in range(ncounts):
        u = readfile.read_checkpoint(u, "concentration", idx)
        plt.figure()
        c = df.plot(u)#, vmin=0, vmax=2)
        plt.title(f"Time-dependent solution at t={time[idx]:.2f}")
        plt.colorbar(c)

    readfile.close()
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("funccall", default="")
    args = parser.parse_args()

    if args.funccall == "store":
        store_function()
    elif args.funccall == "solve":
        solve_convection_diffusion_stationary()
    elif args.funccall == "time":
        solve_convection_diffusion_time_dependent()
    elif args.funccall == "read":
        read_and_plot()
    elif args.funccall == "all":
        import subprocess
        subprocess.run("mpirun -n 3 python test_fenicsstorage.py store", shell=True)
        subprocess.run("mpirun -n 4 python test_fenicsstorage.py time", shell=True)
        subprocess.run("mpirun -n 4 python test_fenicsstorage.py solve", shell=True)
        subprocess.run("python test_fenicsstorage.py read", shell=True)
    else:
        raise ValueError("Unknown function call")
