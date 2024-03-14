import sympy as sp
import pantarei as pr
import dolfin as df
import matplotlib.pyplot as plt

from pantarei.mms import sdiv, sgrad
from ufl import inner, grad


def poisson_form(V, coefficients, boundaries):
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    dx = df.Measure("dx", V.mesh())
    f = coefficients["source"] if "source" in coefficients else 0
    return (
        inner(grad(u), grad(v)) * dx
        - f * v * dx
        - pr.process_boundary_forms(u, v, boundaries)
    )


def test_poisson():
    x, y = sp.symbols("x y ")
    u = x**2 + y**2
    f = -sdiv(sgrad(u))

    degree = 2
    aR = 1.0
    domain = pr.MMSDomain(2)
    uN = pr.sp_neumann_boundary(u, domain.normals)
    uR = pr.sp_robin_boundary(u, aR, domain.normals)

    element = df.FiniteElement("CG", domain.ufl_cell(), degree=2)
    boundaries = [
        pr.DirichletBoundary(pr.expr(u, degree), 1),
        pr.DirichletBoundary(pr.expr(u, degree), 2),
        pr.RobinBoundary(aR, pr.expr(uR[3], degree), 3),
        pr.NeumannBoundary(pr.expr(uN[4], degree), 4),
    ]

    coefficients = {"source": pr.expr(f, degree=degree)}

    uh = pr.solve_stationary(
        domain,
        element=element,
        coefficients=coefficients,
        form=poisson_form,
        boundaries=boundaries,
        solver=pr.StationaryProblemSolver(),
        name="Poisson",
    )

    assert df.errornorm(pr.expr(u, 5), uh, norm_type="H1") < 1e-12
