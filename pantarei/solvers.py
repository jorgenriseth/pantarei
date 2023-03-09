from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeAlias, TypeVar

import dolfin as df
from dolfin import (
    DirichletBC,
    Form,
    Function,
    FunctionSpace,
    Mesh,
    assemble,
    lhs,
    rhs,
    solve,
)
from ufl import Coefficient
from ufl.finiteelement.finiteelementbase import FiniteElementBase

from pantarei.boundary import (
    BoundaryData,
    process_boundary_forms,
    process_dirichlet,
)
from pantarei.computers import BaseComputer
from pantarei.forms import AbstractForm
from pantarei.io.timeseriesstorage import TimeSeriesStorage
from pantarei.timekeeper import TimeKeeper


class ProblemSolver:
    pass


DolfinMatrix: TypeAlias = df.cpp.la.Matrix
DolfinVector: TypeAlias = df.cpp.la.Vector


class StationaryProblemSolver(ProblemSolver):
    def __init__(self, method: str = "lu", preconditioner: str = "none"):
        self._method = method
        self._precond = preconditioner

    def solve(
        self,
        u: Function,
        A: DolfinMatrix,
        b: DolfinVector,
        dirichlet_bcs: List[DirichletBC],
    ) -> Function:
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        solve(A, u.vector(), b, self._method, self._precond)
        return u


class QuasiStationarySolver(ProblemSolver):
    def __init__(
        self,
        solver: StationaryProblemSolver,
        storage: Optional[TimeSeriesStorage] = None,
        computer: Optional[BaseComputer] = None,
    ):

        self._solver = solver
        self._storage = storage
        self._computer = computer

    def solve(self, V: FunctionSpace, time: TimeKeeper) -> Function:
        pass


class TimeDependentSolver(ProblemSolver):
    def __init__(
        self,
        solver: StationaryProblemSolver,
        storage: Optional[TimeSeriesStorage] = None,
        computer: Optional[BaseComputer] = None,
    ):
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
    name: Optional[str] = None,
) -> Function:
    V = FunctionSpace(domain, element)
    F = form.create_fem_form(
        V, coefficients, boundaries
    )  # + process_boundary_forms()
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    a: Form = lhs(F)  #  type: ignore (lhs/rhs allow too many return-types.)
    l: Form = rhs(F)  #  type: ignore (lhs/rhs allow too many return-types.)
    A = assemble(a)
    if not l.empty():
        b = assemble(l)
    else:
        b = df.cpp.la.Vector(domain.mpi_comm(), V.dim())
    u = Function(V, name=name)
    return solver.solve(u, A, b, dirichlet_bcs)


T = TypeVar("T")


def set_optional(
    argument: Optional[T], classname: Callable[[Any], T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument


class ProblemUpdater:
    def update(self, u: Function, time: TimeKeeper, coefficients) -> None:
        pass


def trial_test_functions(form: Form):
    return form.arguments()[1], form.arguments()[0]


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
    projector=None,
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
    u = Function(V, name=name)
    u.assign(u0)

    # TODO:
    # 1) Switch form to a closure/callable.
    # 2) Change "process" to take in the form, rather than test/trialfunctions.
    F = form.create_fem_form(V, coefficients, boundaries)
    a: Form = lhs(F)  #  type: ignore (lhs/rhs allow too many return-types.)
    l: Form = rhs(F)  #  type: ignore (lhs/rhs allow too many return-types.

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
        return solve_stationary(
            self.domain,
            self.element,
            self.coefficients,
            self.form,
            self.boundaries,
            self.solver,
            self.name,
        )
