import dolfin as df

comm = df.MPI.comm_world
rank = comm.rank

print(f"Hello from process {rank}/{comm.size}")
domain = df.UnitSquareMesh(comm, 10, 10)
element = df.FiniteElement("Lagrange", domain.ufl_cell(), 1)
V = df.FunctionSpace(domain, element)
u = df.Function(V, name="u")

file = df.HDF5File(df.MPI.comm_world, "test.h5", "w")
file.write(u, "/u", 0.1)
file.write(u, "/u", 0.2)
file.close()