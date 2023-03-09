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
    project,
    solve,
)
from ufl import Coefficient


class BaseProjector:
    def project(self, expression, space):
        return project(expression, space)


class DirichletProjector(BaseProjector):
    def __init__(self, uD):
        self.uD = uD

    def project(self, u0: Coefficient, V: FunctionSpace):
        # Project u0 to have Dirichlet boundary equal to g0.
        u = TrialFunction(V)
        v = TestFunction(V)
        a0 = u * v * dx  # type: ignore
        L0 = u0 * v * dx  # type: ignore
        u1 = Function(V)
        A = assemble(a0)
        b = assemble(L0)
        DirichletBC(V, self.uD, "on_boundary").apply(A, b)
        solve(A, u1.vector(), b)
        return u1


class HomogeneousDirichletProjector(DirichletProjector):
    def __init__(self):
        super().__init__(Constant(0.0))


class AveragingDirichletProjector(DirichletProjector):
    def __init__(self):
        super().__init__(Constant(0.0))

    def project(self, u0, V):
        ds = Measure("ds", domain=V.mesh())
        surface_area = assemble(1.0 * ds)
        self.uD = assemble(u0 * ds) / surface_area
        return super().project(u0, V)


class NeumannProjector(BaseProjector):
    def __init__(self, uN):
        self.uN = uN
        super().__init__()

    def project(self, u0: Coefficient, V: FunctionSpace):
        # Project u0 to have Dirichlet boundary equal to g0.
        dx = Measure("dx", domain=V.mesh())
        u = TrialFunction(V)
        v = TestFunction(V)
        a0 = u * v * dx  # type: ignore (Argument * Argument is most def. allowed.)
        L0 = u0 * v * dx  # type: ignore (Coefficient * Argument, same.)
        u0 = Function(V)
        solve(a0 == L0, u0)
        return u0


def rescale_function(u: Function, value: float):
    v = u.vector()
    v *= value / assemble(u * dx)
    return u
