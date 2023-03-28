from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeAlias, TypeVar, Union

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
    F = form.create_fem_form(
        V, coefficients, boundaries
    ) 
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)
    a: Form = lhs(F)  #  type: ignore (lhs/rhs allow too many return-types.)
    l: Form = rhs(F)  #  type: ignore (lhs/rhs allow too many return-types.)
    A = assemble(a)
    if not l.empty():
        b = assemble(l)
    else:
        b = Function(V).vector() 
    u = Function(V, name=name)
    return solver.solve(u, A, b, dirichlet_bcs)


T = TypeVar("T")


def set_optional(
    argument: Optional[T], classname: Callable[..., T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument


def trial_test_functions(form: Form):
    return form.arguments()[1], form.arguments()[0]


StrPath: TypeAlias = Union[str, Path]


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
    projector=None,
) -> BaseComputer:
    """Solve a time-dependent problem"""
    computer = set_optional(computer, BaseComputer, {})
    # projector = set_optional(projector, lambda: df.project)
    name = set_optional(name, str)

    time.reset()  # TODO: Should this be the user's responsibility?
    V = FunctionSpace(domain, element)
    dirichlet_bcs = process_dirichlet(V, domain, boundaries)

    u0 = initial_condition(V, boundaries)
    coefficients["u0"] = u0  # type: ignore
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
    storage.write_function(u, name)
    for idx, ti in enumerate(time):
        print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        b = assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)
        computer.compute(ti, u)
        storage.write_checkpoint(u, name, float(time))
        coefficients["u0"].assign(u)  # type: ignore

    storage.close()
    return computer

def print_progress(t, T, rank=0):
    if rank !=0:
        return
    progress = int(20 * t / T)
    print(
        f"[{'=' * progress}{' ' * (20 - progress)}]", end="\r", flush=True
    )

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
