from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt
from pantarei.io.fenicsstorage import FenicsStorage

comm = df.MPI.comm_world
rank = comm.rank

print(f"Hello from process {rank}/{comm.size}")
domain = df.UnitSquareMesh(comm, 10, 10)
element = df.FiniteElement("Lagrange", domain.ufl_cell(), 1)
V = df.FunctionSpace(domain, element)
u = df.Function(V, name="u")

file = df.HDF5File(df.MPI.comm_world, "test.h5", "r")
file.read(u, "/u/vector_0")
df.plot(u)
plt.show(block=False)

file.read(u, "/u/vector_1")
file.close()
del file
plt.figure()
df.plot(u)
plt.show()
exit()

store = FenicsStorage(Path("storetest.hdf5"), "r")
u = store.read_function("u") 
store.close()

df.plot(u)
plt.show()
