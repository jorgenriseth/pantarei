import logging
from typing import Callable, Dict, Optional, TypeAlias, TypeVar

import dolfin as df
import ufl
import numpy as np

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
        if j not in p:
            raise KeyError(f"Missing key {j} in p; p.keys() = {p.keys()}")

    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [df.interpolate(p[j], subspaces[idx]) for idx, j in enumerate(compartments)]
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


def print_mesh_quality_info(mesh: df.Mesh) -> dict[str, int | float]:
    num_cells = mesh.num_cells()
    num_vertices = mesh.num_vertices()
    min_diameter = 2 * mesh.rmin()
    max_diameter = 2 * mesh.rmax()

    print(f"{'Mesh Quality Information':^35}")
    print("-" * 35)
    print(f"{'Number of cells':<20} : {num_cells}")
    print(f"{'Number of vertices':<20} : {num_vertices}")
    print(f"{'Minimum cell diameter':<20} : {min_diameter:.6f}")
    print(f"{'Maximum cell diameter':<20} : {max_diameter:.6f}")
    return {
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
    }


def set_optional(
    argument: Optional[T], classname: Callable[..., T], *args, **kwargs
) -> T:
    if argument is None:
        argument = classname(*args, **kwargs)
    return argument


def mesh_info(mesh: df.Mesh) -> dict[str, int | float]:
    num_cells = mesh.num_cells()
    num_vertices = mesh.num_vertices()
    min_diameter = 2 * mesh.rmin()
    max_diameter = 2 * mesh.rmax()
    return {
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
    }


def print_mesh_info(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
) -> None:
    print(f"{'Mesh Quality Information':^35}")
    print("-" * 35)
    print(f"{'Number of cells':<20} : {num_cells}")
    print(f"{'Number of vertices':<20} : {num_vertices}")
    print(f"{'Minimum cell diameter':<20} : {min_diameter:.{decimals}g}")
    print(f"{'Maximum cell diameter':<20} : {max_diameter:.{decimals}g}")


def as_latex_table(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
    indentsize: int = 4,
) -> None:
    indent = " " * indentsize
    str_out = (
        latex_header()
        + 3 * indent
        + latex_row(num_cells, num_vertices, min_diameter, max_diameter, decimals)
        + latex_footer()
    )
    return str_out


def latex_header(index_header: Optional[str] = None):
    index_headers = [index_header] * (index_header is not None) + [
        "# Cells",
        "# Vertices",
        "$h_{min}$",
        "$h_{max}$",
    ]
    formatted_index_headers = [r"\textbf{" + x + "}" for x in index_headers]
    return rf"""
\begin{{table}}[h!]
    \centering 
    \begin{{tabular}}{{lcccc}}
        \hline
        {" & ".join(formatted_index_headers)} \\
        \hline
"""


def latex_row(
    num_cells: int,
    num_vertices: int,
    min_diameter: float,
    max_diameter: float,
    decimals: int,
    index: Optional[str] = None,
) -> str:
    str_formatter = [index] * (index is not None) + [
        str(num_cells),
        str(num_vertices),
        f"{min_diameter:.{decimals}f}",
        f"{max_diameter:.{decimals}f}",
    ]
    return " & ".join(str_formatter) + r" \\"


def latex_footer():
    return r"""
        \hline
    \end{tabular}
    \caption{Mesh Information}
    \label{table: mesh info}
\end{table}
"""


def print_progress(t, T, rank=0):
    if rank != 0:
        return
    progress = int(20 * t / T)
    print(f"[{'=' * progress}{' ' * (20 - progress)}]", end="\r", flush=True)


def subspace_local_dofs(W: df.FunctionSpace, idx: int):
    mesh = W.mesh()
    dofs = W.sub(idx).dofmap().entity_closure_dofs(mesh, mesh.topology().dim())
    dofs = np.sort(np.unique(dofs))
    return dofs


def total_concentration(
    W: df.FunctionSpace, phi: dict[str, float], compartments: list[str]
):
    uT = df.Function(W.sub(0).collapse())
    N = len(compartments)
    dofs = [subspace_local_dofs(W, idx) for idx in range(N)]

    def call(u: df.Function):
        uT.vector().set_local(
            sum(
                (
                    phi[i] * u.vector().get_local(dofs[idx])
                    for idx, i in enumerate(compartments)
                )
            )
        )
        uT.vector().apply("insert")
        return uT

    return call
