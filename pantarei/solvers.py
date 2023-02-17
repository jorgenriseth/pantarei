from typing import Any, Callable, Optional, List, Dict, TypeAlias, TypeVar
from dataclasses import dataclass

import dolfin as df
from dolfin import FunctionSpace, Form, DirichletBC, Function, assemble, solve, lhs, rhs, Mesh
from ufl import Coefficient
from ufl.finiteelement.finiteelementbase import FiniteElementBase

from pantarei.boundary import process_dirichlet, BoundaryData
from pantarei.computers import BaseComputer
from pantarei.forms import AbstractForm
from pantarei.timekeeper import TimeKeeper
from pantarei.io.timeseriesstorage import TimeSeriesStorage


class ProblemSolver:
    pass

DolfinMatrix: TypeAlias = df.cpp.la.Matrix
DolfinVector: TypeAlias = df.cpp.la.Vector

class StationaryProblemSolver(ProblemSolver):
    def __init__(self, method: str = "lu", preconditioner: str = "none"):
        self._method = method
        self._precond = preconditioner

    def solve(self, u: Function, A: DolfinMatrix, b: DolfinVector, dirichlet_bcs: List[DirichletBC]) -> Function:
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        solve(A, u.vector(), b, self._method, self._precond)
        return u

class QuasiStationarySolver(ProblemSolver):
    def __init__(self,
        solver: StationaryProblemSolver,
        storage: Optional[TimeSeriesStorage] = None,
        computer: Optional[BaseComputer] = None):

        self._solver = solver
        self._storage = storage
        self._computer = computer

    def solve(self, V: FunctionSpace, time: TimeKeeper) -> Function:
        pass



class TimeDependentSolver(ProblemSolver):
    def __init__(self,
        solver: StationaryProblemSolver,
        storage: Optional[TimeSeriesStorage] = None,
        computer: Optional[BaseComputer] = None):
        self._solver = solver
        self._storage = storage
        self._computer = computer

    def solve(self, V: FunctionSpace, time: TimeKeeper) -> df.Function:
        pass


def solve_stationary(
    domain: Mesh,
    element: FiniteElementBase,
    coefficients: Dict[str, Coefficient],
    form: AbstractForm,
    boundaries: List[BoundaryData],
    solver: StationaryProblemSolver,
    name: Optional[str] = None
) -> Function:
    V = FunctionSpace(domain, element)
    F = form.create_fem_form(V, coefficients, boundaries)
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    a: Form = lhs(F) #  type: ignore (lhs/rhs allow too many return-types.)
    l: Form = rhs(F) #  type: ignore (lhs/rhs allow too many return-types.)
    A = assemble(a)
    b = assemble(l)
    u = Function(V, name=name)
    return solver.solve(u, A, b, dirichlet_bcs)


T = TypeVar("T")
def set_optional(argument: Optional[T], classname: Callable[[Any], T], *args, **kwargs) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument


class ProblemUpdater:
    def update(self, u: Function, time: TimeKeeper, coefficients) -> None:
        pass


def solve_time_dependent(
    domain: Mesh,
    element: FiniteElementBase,
    coefficients: Dict[str, Coefficient],
    form: AbstractForm,
    boundaries: List[BoundaryData],
    initial_condition,
    time: TimeKeeper,
    solver: StationaryProblemSolver,
    storage_path: Optional[str] = None,
    name: Optional[str] = None,
    computer: Optional[BaseComputer] = None,
    updater: Optional[ProblemUpdater] = None,
) -> BaseComputer:
    """Solve a time-dependent problem"""
    time.reset()
    V = FunctionSpace(domain, element)
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    storage = TimeSeriesStorage("w", storage_path, mesh=domain, V=V)
    computer = set_optional(computer, BaseComputer, {})
    updater = set_optional(updater, ProblemUpdater)

    # Prepare initial conditions
    u0 = df.project(initial_condition, V, bcs=dirichlet_bcs)
    # TODO: Allow different name for coefficient.
    if "u0" in coefficients:
        raise ValueError(f"Coefficient list already has initial condition entry.")
    else:
        coefficients["u0"] = u0
    u = Function(V, name=name)
    u.assign(u0)

    F = form.create_fem_form(V, coefficients, boundaries)
    a: Form = lhs(F) #  type: ignore (lhs/rhs allow too many return-types.)
    l: Form = rhs(F) #  type: ignore (lhs/rhs allow too many return-types.)
    A = assemble(a)

    computer.compute(time, u)
    updater.update(u, time, coefficients)
    storage.write(u, float(time))
    for idx, ti in enumerate(time):
        b = assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)
        computer.compute(ti, u)
        updater.update(u, ti, coefficients)
        storage.write(u, float(ti))
        coefficients["u0"].assign(u)

    storage.close()
    return computer

@dataclass
class StationaryProblem:
    domain: Mesh
    element: FiniteElementBase
    coefficients: Dict[str, Coefficient]
    form: AbstractForm
    boundaries: List[BoundaryData]
    solver: StationaryProblemSolver
    name: Optional[str] = None

    def solve(self):
        return solve_stationary(self.domain, self.element, self.coefficients, self.form, self.boundaries, self.solver, self.name)
