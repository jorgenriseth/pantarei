from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

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

from pantarei.boundary import BoundaryData, process_dirichlet
from pantarei.computers import BaseComputer
from pantarei.fenicsstorage import FenicsStorage
from pantarei.forms import AbstractForm
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
    F = form(V, coefficients, boundaries)
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    a = lhs(F)
    l = rhs(F)
    A = assemble(a)
    if l.empty():  # type: ignore
        b = Function(V).vector()
    else:
        b = assemble(l)

    u = Function(V, name=name)
    return solver.solve(u, A, b, dirichlet_bcs)


T = TypeVar("T")


def set_optional(
    argument: Optional[T], classname: Callable[..., T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument


def solve_time_dependent(
    domain: Mesh,
    element: FiniteElementBase,
    coefficients: Dict[str, Coefficient],
    form: AbstractForm,
    boundaries: List[BoundaryData],
    initial_condition: Callable[[FunctionSpace, List[BoundaryData]], Function],
    time: TimeKeeper,
    solver: StationaryProblemSolver,
    storage: FenicsStorage,
    name: Optional[str] = None,
    computer: Optional[BaseComputer] = None,
) -> BaseComputer:
    """Solve a time-dependent problem"""
    computer = set_optional(computer, BaseComputer, {})
    name = set_optional(name, str)

    V = FunctionSpace(domain, element)
    u = Function(V, name=name)

    coefficients["u0"] = initial_condition(V, boundaries)
    if not isinstance(coefficients["u0"], Function):
        coefficients["u0"] = df.project(coefficients["u0"], V)
    u.assign(coefficients["u0"])
    computer.compute(time, u)
    storage.write_function(u, name)

    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    F = form(V, coefficients, boundaries)
    a = lhs(F)
    l = rhs(F)
    A = assemble(a)

    for ti in time:
        print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        b = assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)
        computer.compute(ti, u)
        storage.write_checkpoint(u, name, float(time))
        coefficients["u0"].assign(u)

    storage.close()
    return computer


def print_progress(t, T, rank=0):
    if rank != 0:
        return
    progress = int(20 * t / T)
    print(f"[{'=' * progress}{' ' * (20 - progress)}]", end="\r", flush=True)


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


def trial_test_functions(form: Form):
    return form.arguments()[1], form.arguments()[0]
