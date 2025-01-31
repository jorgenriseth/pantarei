from typing import Callable, Dict, List, TypeAlias

import dolfin as df
from dolfin import grad, inner

from pantarei.boundary import BoundaryData, process_boundary_forms
from pantarei.timekeeper import TimeDelta
from pantarei.utils import CoefficientsDict

StationaryForm: TypeAlias = Callable[
    [df.FunctionSpace, CoefficientsDict, List[BoundaryData]], df.Form
]
TimedependentForm: TypeAlias = Callable[
    [df.FunctionSpace, CoefficientsDict, List[BoundaryData], df.Function, TimeDelta],
    df.Form,
]
AbstractForm: TypeAlias = StationaryForm | TimedependentForm


def poisson_form() -> AbstractForm:
    def abstract_form(
        V: df.FunctionSpace,
        coefficients: CoefficientsDict,
        boundaries: List[BoundaryData],
    ) -> df.Form:
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        D = coefficients["D"]
        f = coefficients["source"]
        dx = df.Measure("dx", V.mesh())
        return (
            inner(D * grad(u), grad(v)) - f * v  # type: ignore
        ) * dx + process_boundary_forms(u, v, boundaries)

    return abstract_form


def diffusion_form() -> AbstractForm:
    def abstract_form(
        V: df.FunctionSpace,
        coefficients: CoefficientsDict,
        boundaries: List[BoundaryData],
        u0: df.Function,
        dt: TimeDelta,
    ) -> df.Form:
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        D = coefficients["D"]
        f = coefficients["source"] if "source" in coefficients else None
        dx = df.Measure("dx", V.mesh())
        F = (
            (u - u0) * v + dt * (inner(D * grad(u), grad(v)) - (f is not None) * f * v)  # type: ignore
        ) * dx + dt * process_boundary_forms(u, v, boundaries)
        return F

    return abstract_form
