from pathlib import Path
import os

import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from pantarei.fenicsstorage import FenicsStorage

comm = df.MPI.comm_world
rank = comm.rank

print(f"Hello from process {rank}/{comm.size}")
domain = df.UnitSquareMesh(comm, 10, 10)
u_expr = df.Expression("x[0]*x[0]", degree=2, domain=domain)
element = df.FiniteElement("Lagrange", domain.ufl_cell(), 1)
V = df.FunctionSpace(domain, element)
u = df.project(u_expr, V)

file = df.HDF5File(df.MPI.comm_world, "test.h5", "w")
file.write(u, "u", 0.0)
u_expr = df.Expression("x[1]*x[1]", degree=2, domain=domain)
u = df.project(u_expr, V)
file.write(u, "u", 10.0)
file.close()
exit()

domain = df.UnitSquareMesh(df.MPI.comm_world, 15, 15)
u_expr = df.Expression("x[0]*x[0]", degree=2, domain=domain)
V = df.FunctionSpace(domain, "CG", 1)
u = df.project(u_expr, V)
u.rename("u", "")

filepath = Path("storetest.hdf5")

store = FenicsStorage(filepath, "w")
store.write_function(u)
store.close()

df.plot(u)
plt.show()
