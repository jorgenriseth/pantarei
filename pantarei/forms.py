from typing import Callable, Dict, List, TypeAlias

from dolfin import (
    Form,
    FunctionSpace,
    Measure,
    TestFunction,
    TrialFunction,
    grad,
    inner,
)
from ufl import Coefficient

from pantarei.boundary import BoundaryData, process_boundary_forms

AbstractForm: TypeAlias = Callable[
    [FunctionSpace, Dict[str, Coefficient], List[BoundaryData]], Form
]


def poisson_form() -> AbstractForm:
    def abstract_form(
        V: FunctionSpace,
        coefficients: Dict[str, Coefficient],
        boundaries: List[BoundaryData],
    ) -> Form:
        domain = V.mesh()
        u = TrialFunction(V)
        v = TestFunction(V)
        D = coefficients["D"]
        f = coefficients["source"]
        dx = Measure("dx", V.mesh())
        return (
            inner(D * grad(u), grad(v)) - f * v  # type: ignore
        ) * dx + process_boundary_forms(u, v, domain, boundaries)

    return abstract_form


def diffusion_form() -> AbstractForm:
    def abstract_form(
        V: FunctionSpace,
        coefficients: Dict[str, Coefficient],
        boundaries: List[BoundaryData],
    ) -> Form:
        u = TrialFunction(V)
        v = TestFunction(V)
        D = coefficients["D"]
        dt = coefficients["dt"]
        u0 = coefficients["u0"]
        f = coefficients["source"]
        dx = Measure("dx", V.mesh())
        F = (
            (u - u0) * v + dt * (inner(D * grad(u), grad(v)) - f * v)  # type: ignore
        ) * dx + dt * process_boundary_forms(u, v, V.mesh(), boundaries)
        return F

    return abstract_form
