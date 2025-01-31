from typing import Any, Callable, Dict, List, TypeAlias

from dolfin import dx  # type: ignore
from dolfin import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    Measure,
    TestFunction,
    TrialFunction,
    assemble,
    grad,
    inner,
    project,
    solve,
)
from ufl import Coefficient

from pantarei.utils import assign_mixed_function


def smoothing_projector(h1_weight):
    def projector(u, V, bcs):
        """Projects the function u onto a function space V with
        Dirichlet boundary conditions given by bcs with a weighted H1-norm
        i.e.  a weight coefficient for the 1-st derivative on the norm in which
        the minimization problem is solved."""
        u_ = TrialFunction(V)
        v = TestFunction(V)

        # TODO: What if this projection is what causes the oscillations? Should be interp?
        u0_ = project(u, V)  # Projection of u onto V, without the bcs.
        a0 = inner(u_, v) * dx + h1_weight * inner(grad(u_), grad(v)) * dx
        L0 = inner(u0_, v) * dx + h1_weight * inner(grad(u0_), grad(v)) * dx
        u1 = Function(V)
        A = assemble(a0)
        b = assemble(L0)
        for bc in bcs:
            bc.apply(A, b)
        solve(A, u1.vector(), b)
        return u1

    return projector


class NeumannProjector:
    def __init__(self, uN):
        self.uN = uN

    def project(self, u0: Coefficient, V: FunctionSpace, _):
        # Project u0 to have Dirichlet boundary equal to g0.
        dx = Measure("dx", domain=V.mesh())
        u = TrialFunction(V)
        v = TestFunction(V)
        a0 = u * v * dx  # type: ignore (Argument * Argument is most def. allowed.)
        L0 = u0 * v * dx  # type: ignore (Coefficient * Argument, same.)
        u0 = Function(V)
        solve(a0 == L0, u0)
        return u0


def rescale_function(u: Function, value: float) -> Function:
    v = u.vector()
    v *= value / assemble(u * dx)
    return u


Projector: TypeAlias = Callable[[Any, FunctionSpace, List[DirichletBC]], Function]


def mixed_space_projector(
    labels: List[str], projector: Projector
) -> Callable[[Any, FunctionSpace, List[DirichletBC]], Function]:
    def project(
        u0: Dict[str, Coefficient], V: FunctionSpace, bcs: List[DirichletBC]
    ) -> Function:
        U = assign_mixed_function(u0, V, labels)
        return projector(U, V, bcs)

    return project
