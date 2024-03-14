import time as pytime
from pathlib import Path
from typing import Any
import pytest

import sympy as sp
import numpy as np
import dolfin as df
import ufl

from ufl import inner, grad


import pantarei as pr
from pantarei.boundary import RobinBoundary, DirichletBoundary, NeumannBoundary
from pantarei.mms import (
    expr,
    MMSSquare,
    SympyDiffOperators,
    sp_robin_boundary,
    sp_neumann_boundary,
)


def strong_form(c, coefficients):
    Sp = SympyDiffOperators("xy")
    phi, D, beta, r = tuple(coefficients[x] for x in ["phi", "D", "beta", "r"])
    compartments = list(c.keys())
    s = {
        j: sum(beta * (c[i] - c[j]) for i in compartments if i != j)
        for j in compartments
    }
    f = {
        j: phi[j] * (Sp.dt(c[j]) - Sp.div(D[j] * Sp.grad(c[j]))) + r[j] * c[j] - s[j]
        for j in compartments
    }
    return f


def multicomp_diffusion_form(
    V: df.FunctionSpace,
    coefficients: pr.CoefficientsDict,
    boundaries: list[pr.BoundaryData],
    c0: df.Function,
    dt: float,
    compartments: list[str],
) -> df.Form:
    dx = df.Measure("dx", domain=V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    source = (
        coefficients["source"]
        if "source" in coefficients
        else {j: 0 for j in compartments}
    )
    cellform = (
        sum(
            (
                compartment_form(
                    idx_j,
                    u,
                    v,
                    c0,
                    coefficients,
                    compartments,
                    dt,
                    source,
                )
                for idx_j, _ in enumerate(compartments)
            )
        )
        * dx
    )
    return cellform + pr.process_boundary_forms(u, v, boundaries)


def multicomp_errornorm(u: dict[str, Any], uh, compartments, norm="H1", high_degree=5):
    """Compute the Sobolev-norm error between a target u and approximation uh
    over a set of compartments. u is a dictionary containing the targets, uh is a
    mixed-dimensional function returned from the solver."""
    Vhigh = df.FunctionSpace(uh.function_space().mesh(), "CG", high_degree)
    return np.sqrt(
        sum(
            [
                df.errornorm(df.interpolate(u[j], Vhigh), uh.sub(idx), norm) ** 2
                for idx, j in enumerate(compartments)
            ]
        )
    )


def temporal_errornorm_from_vec(tvec: np.ndarray, errorvec: np.ndarray):
    """Computes the Bochner space norm based on a trapezoid quadrature
    between the squared Sobolev norms at each time-step."""
    return np.sqrt(np.trapz(errorvec**2, tvec))


def compartment_form(
    idx_j: int,
    u: ufl.Argument,
    v: ufl.Argument,
    c0: df.Function,
    coefficients: pr.CoefficientsDict,
    compartments: list[str],
    dt: float,
    source: dict[str, float],
) -> df.Form:
    j = compartments[idx_j]
    phi, D, beta, r = tuple(coefficients[x] for x in ["phi", "D", "beta", "r"])
    sj = sum(
        [
            beta * (u[idx_i] - u[idx_j])
            for idx_i, i in enumerate(compartments)
            if idx_i != idx_j
        ]
    )
    return (
        phi[j]
        * (
            (u[idx_j] - c0[idx_j]) / dt * v[idx_j]
            + inner(D[j] * grad(u[idx_j]), grad(v[idx_j]))
        )
        + (r[j] * u[idx_j] - sj) * v[idx_j]
        - (source[j] * v[idx_j])
    )


def test_mms_multicompartment_diffusion():
    # Exact coefficient values are irrelevant, as we're using P2-MMS solution.
    coefficients = {
        "phi": {"ecs": 0.14, "pvs": 0.1},
        "D": {"ecs": 0.1, "pvs": 1.0},
        "r": {"ecs": 0.0, "pvs": 3.1},
        "beta": 5.0,
        "robin_coeff": 0.3,
    }
    # coefficients = {
    #     "phi": {"ecs": 0.1, "pvs": 0.1},
    #     "D": {"ecs": 1.0, "pvs": 1.0},
    #     "r": {"ecs": 1.0, "pvs": 1.0},
    #     "beta": 1.0,
    #     "robin_coeff": 2.0,
    # }

    compartments = ["ecs", "pvs"]
    t, x, y = sp.symbols("t x y")
    Sp = SympyDiffOperators("xy")

    domain = MMSSquare(2)
    P2 = df.FiniteElement("CG", domain.ufl_cell(), degree=2)
    W = df.FunctionSpace(domain, df.MixedElement([P2] * len(compartments)))

    expression_degree = 2
    a = {"ecs": 0.9, "pvs": 0.1}
    c = {j: a[j] * t * (x**2 + y**2) for j in compartments}
    D = coefficients["D"]
    phi = coefficients["phi"]
    flux = {j: -phi[j] * D[j] * Sp.grad(c[j]) for j in compartments}
    f = strong_form(c, coefficients)

    dt = 0.1
    T = 1.0
    time = pr.TimeKeeper(dt, T)

    c_expr = {j: expr(c[j], degree=expression_degree, t=time) for j in c}
    coefficients["source"] = {
        j: expr(f[j], degree=expression_degree, t=time) for j in c
    }

    gR = {
        j: sp_robin_boundary(c[j], flux[j], domain.normals, coefficients["robin_coeff"])
        for j in compartments
    }
    gN = {j: sp_neumann_boundary(c[j], flux[j], domain.normals) for j in compartments}

    boundaries = {
        idx: [
            DirichletBoundary(c_expr[j], 1),
            DirichletBoundary(c_expr[j], 2),
            RobinBoundary(
                coefficients["robin_coeff"],
                expr(gR[j][3], expression_degree, t=time),
                3,
            ),
            NeumannBoundary(expr(gN[j][4], expression_degree, t=time), 4),
        ]
        for idx, j in enumerate(compartments)
    }
    # ... and convert to a mixed-dimensional friendly format
    boundaries = pr.indexed_boundary_conditions(boundaries)
    dirichlet_bcs = pr.process_dirichlet(W, boundaries)

    u0 = pr.assign_mixed_function(c_expr, W, compartments)
    F = multicomp_diffusion_form(W, coefficients, boundaries, u0, time.dt, compartments)

    bilinear_form = df.lhs(F)
    linear_form = df.rhs(F)
    A = df.assemble(bilinear_form)
    u = df.Function(W)

    E = {
        "time": np.nan * np.empty(len(time)),
        "multicomp_error": np.nan * np.empty(len(time)),
    }
    E["time"][0] = float(time)
    E["multicomp_error"][0] = multicomp_errornorm(c_expr, u, compartments)

    for idx, ti in enumerate(time, start=1):
        pr.print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        b = df.assemble(linear_form)
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        df.solve(A, u.vector(), b, "lu")
        E["time"][idx] = float(ti)
        E["multicomp_error"][idx] = multicomp_errornorm(
            c_expr, u, compartments, high_degree=5
        )
        u0.assign(u)

    df.MPI.comm_world.barrier()
    t = E["time"][~np.isnan(E["time"])]
    errors = E["multicomp_error"][~np.isnan(E["multicomp_error"])]
    assert temporal_errornorm_from_vec(t, errors) == pytest.approx(0)
