from abc import ABC, abstractmethod
from typing import Dict, List
from ufl import Coefficient
from dolfin import FunctionSpace, TrialFunction, TestFunction, Form, inner, grad, Measure

from pantarei.boundary import BoundaryData, process_boundary_forms
from pantarei.domain import Domain

class AbstractForm(ABC):
    @staticmethod
    @abstractmethod
    def create_fem_form(V: FunctionSpace, coefficients: Dict[str, Coefficient], boundaries: List[BoundaryData]) -> Form:
        pass
    
    
class PoissonForm:
    @staticmethod
    def create_fem_form(V: FunctionSpace, coefficients: Dict[str, Coefficient], boundaries: List[BoundaryData]) -> Form:
        domain = V.mesh()
        u = TrialFunction(V)
        v = TestFunction(V)
        D = coefficients["D"]
        f = coefficients["source"]
        dx = Measure("dx", V.mesh())
        return ((inner(D * grad(u), grad(v)) - f * v) * dx  # type: ignore
                + process_boundary_forms(u, v, domain, boundaries))


class DiffusionForm:
    @staticmethod
    def create_fem_form(V: FunctionSpace, coefficients: Dict[str, Coefficient], boundaries: List[BoundaryData]) -> Form:
        u = TrialFunction(V)
        v = TestFunction(V)
        D = coefficients["D"]
        dt = coefficients["dt"]
        u0 = coefficients["u0"]
        f = coefficients["source"]
        dx = Measure("dx", V.mesh())
        F = (((u - u0) * v + dt * (inner(D*grad(u), grad(v)) - f * v)) * dx  # type: ignore
                + dt * process_boundary_forms(u, v, V.mesh(), boundaries))
        return F