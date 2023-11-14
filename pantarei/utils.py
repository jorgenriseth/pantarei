import logging
from typing import Callable, Dict, Optional, TypeAlias, TypeVar

import dolfin as df
import ufl

DolfinMatrix: TypeAlias = df.cpp.la.Matrix
DolfinVector: TypeAlias = df.cpp.la.Vector
FormCoefficient: TypeAlias = float | ufl.Coefficient
CoefficientsDict: TypeAlias = Dict[str, FormCoefficient]
T = TypeVar("T")


def assign_mixed_function(p, V, compartments):
    """Create a function in a mixed function-space with sub-function being
    assigned from a dictionray of functions living in the subspaces."""
    P = df.Function(V)
    for j in compartments:
        if not j in p:
            raise KeyError(f"Missing key {j} in p; p.keys() = {p.keys()}")

    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [
        df.interpolate(p[j], subspaces[idx])
        for idx, j in enumerate(compartments)
    ]
    assigner = df.FunctionAssigner(V, subspaces)
    assigner.assign(P, Pint)
    return P


def rescale_function(u: df.Function, value: float):
    """Rescale a function u to have integral value"""
    v = u.vector()
    v *= value / df.assemble(u * df.dx)
    return u


def trial_test_functions(form: df.Form):
    """Get the test and trial function present in a variational form."""
    return form.arguments()[1], form.arguments()[0]


def single_logger(logger: logging.Logger, logfunc: str, logstring: str):
    if df.MPI.comm_world.rank == 0:
        getattr(logger, logfunc)(logstring)

def rank_logger(logger: logging.Logger, logfunc: str, logstring: str):
    getattr(logger, logfunc)(f"Process {df.MPI.comm_world.rank}: {logstring}")


def mpi_single_process_logger(logger: logging.Logger):
    return lambda logfunc, logstring: single_logger(logger, logfunc, logstring)


def set_optional(
    argument: Optional[T], classname: Callable[..., T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument
