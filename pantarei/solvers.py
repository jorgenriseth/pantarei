import logging
import time as pytime
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, TypeAlias

import dolfin as df
from ufl.finiteelement.finiteelementbase import FiniteElementBase

import pantarei as pr
from pantarei.boundary import BoundaryData, process_dirichlet
from pantarei.computers import BaseComputer
from pantarei.fenicsstorage import FenicsStorage
from pantarei.forms import StationaryForm, TimedependentForm
from pantarei.timekeeper import TimeKeeper
from pantarei.utils import CoefficientsDict, set_optional, single_logger

logger = logging.getLogger(__name__)
log = partial(single_logger, logger)


DolfinMatrix: TypeAlias = df.cpp.la.Matrix
DolfinVector: TypeAlias = df.cpp.la.Vector
InitialCondition: TypeAlias = Callable[
    [df.FunctionSpace, List[BoundaryData]], df.Function
]


class ProblemSolver:
    pass


class StationaryProblemSolver(ProblemSolver):
    def __init__(self, method: str = "lu", preconditioner: str = "none"):
        self._method = method
        self._precond = preconditioner

    def solve(
        self,
        u: df.Function,
        A: DolfinMatrix,
        b: DolfinVector,
        dirichlet_bcs: List[df.DirichletBC],
    ) -> df.Function:
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        df.solve(A, u.vector(), b, self._method, self._precond)
        return u


def solve_stationary(
    domain: df.Mesh,
    element: FiniteElementBase,
    coefficients: CoefficientsDict,
    form: StationaryForm,
    boundaries: List[BoundaryData],
    solver: StationaryProblemSolver,
    name: Optional[str] = None,
) -> df.Function:
    V = df.FunctionSpace(domain, element)
    F = form(V, coefficients, boundaries)
    dirichlet_bcs = process_dirichlet(V, boundaries)
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)
    if l.empty():  # type: ignore
        b = df.Function(V).vector()
    else:
        b = df.assemble(l)

    u = df.Function(V, name=name)
    return solver.solve(u, A, b, dirichlet_bcs)


def solve_time_dependent(
    V: df.FunctionSpace,
    form: TimedependentForm,
    coefficients: CoefficientsDict,
    initial_condition: InitialCondition,
    boundaries: List[BoundaryData],
    time: TimeKeeper,
    solver: StationaryProblemSolver,
    storage: FenicsStorage,
    name: Optional[str] = None,
    computer: Optional[BaseComputer] = None,
) -> BaseComputer:
    """Solve a time-dependent problem"""
    computer = set_optional(computer, BaseComputer, {})
    name = set_optional(name, str)

    u = df.Function(V, name=name)
    u0 = initial_condition(V, boundaries)
    u.assign(u0)
    computer.compute(time, u)
    storage.write_function(u, name)

    dirichlet_bcs = process_dirichlet(V, boundaries)
    F = form(V, coefficients, boundaries, u0, time.dt)
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)

    tic = pytime.time()
    for ti in time:
        print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        b = df.assemble(l)
        solver.solve(u, A, b, dirichlet_bcs)
        computer.compute(ti, u)
        storage.write_checkpoint(u, name, float(time))
        u0.assign(u)
    logger.info("Time loop finished.")
    toc = pytime.time()
    df.MPI.comm_world.barrier()
    log("info", f"Elapsed time in loop: {toc - tic:.2f} seconds.")
    storage.close()
    return computer


@dataclass
class StationaryProblem:
    domain: df.Mesh
    element: FiniteElementBase
    coefficients: CoefficientsDict
    form: StationaryForm
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
