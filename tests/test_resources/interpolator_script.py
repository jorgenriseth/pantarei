import dolfin as df
import pantarei as pr
import numpy as np
import matplotlib.pyplot as plt


from pantarei import subspace_local_dofs

domain = pr.MMSSquare(200)
el = df.FiniteElement("CG", domain.ufl_cell(), degree=2)
element = df.VectorElement(el.family(), el.cell(), el.degree(), dim=2)
W = df.FunctionSpace(domain, element)
V = df.FunctionSpace(domain, el)

exp1 = df.Expression("+(x[0]*x[0] + x[1]*x[1])", degree=2)
exp2 = df.Expression("-(x[0]*x[0] + x[1]*x[1])", degree=2)

compartments = [1, 2]
exps = {1: exp1, 2: exp2}
phi = {1: 0.5, 2: 0.5}
u = pr.assign_mixed_function(exps, W, compartments)
uT = df.Function(W.sub(0).collapse())
N = len(compartments)
dofs = [subspace_local_dofs(W, idx) for idx in range(N)]
uT.vector().set_local(
    sum(
        (phi[i] * u.vector().get_local(dofs[idx]) for idx, i in enumerate(compartments))
    )
)
uT.vector().apply("insert")
assert np.allclose(uT.vector().vec().array, 0.0)
