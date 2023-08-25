import dolfin as df
import ufl
from typing import TypeAlias, Dict, TypeVar, Optional, Callable


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


def set_optional(
    argument: Optional[T], classname: Callable[..., T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument