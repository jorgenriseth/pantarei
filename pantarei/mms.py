from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import dolfin as df
import numpy as np
import sympy as sp
import ufl
import ulfy
from dolfin import DOLFIN_EPS, div, grad, inner

from pantarei.boundary import (
    BoundaryData,
    BoundaryTag,
    DirichletBoundary,
    NeumannBoundary,
    RobinBoundary,
)
from pantarei.domain import Domain
from pantarei.timekeeper import TimeKeeper
from pantarei.utils import CoefficientsDict, FormCoefficient

# 3D-mesh = df.BoxMesh(df.Point(-1, -1, -1), df.Point(1, 1, 1), 10, 10, 10)


class MMSDomain(Domain):
    def __init__(self, N: int):
        subdomains = {
            1: df.CompiledSubDomain("x[1] <= +tol", tol=DOLFIN_EPS),
            2: df.CompiledSubDomain("x[1] >= -tol", tol=DOLFIN_EPS),
        }
        subboundaries = {
            1: df.CompiledSubDomain("near(x[0], -1) && on_boundary"),
            2: df.CompiledSubDomain("near(x[0], 1) && on_boundary"),
            3: df.CompiledSubDomain("near(x[1], -1) && on_boundary"),
            4: df.CompiledSubDomain("near(x[1], 1) && on_boundary"),
        }
        normals = {
            1: np.array([-1.0, +0.0]),
            2: np.array([+1.0, +0.0]),
            3: np.array([+0.0, -1.0]),
            4: np.array([+0.0, +1.0]),
        }

        mesh = df.RectangleMesh(
            df.Point(-1, -1), df.Point(1, 1), N, N, diagonal="crossed"
        )
        subdomain_tags = mark_subdomains(subdomains, mesh, 0, default_value=1)
        boundary_tags = mark_subdomains(subboundaries, mesh, 1)

        super().__init__(mesh, subdomain_tags, boundary_tags)
        self.normals = normals


def mark_subdomains(
    subdomains: Dict[int, df.CompiledSubDomain],
    mesh: df.Mesh,
    codim: int,
    default_value: int = 0,
):
    dim = mesh.topology().dim() - codim
    subdomain_tags = df.MeshFunction(
        "size_t", mesh, dim=dim, value=default_value
    )
    for tag, subd in subdomains.items():
        subd.mark(subdomain_tags, tag)
    return subdomain_tags


def sp_grad(u, variables) -> np.ndarray:
    return np.array([u.diff(xi) for xi in "".join(variables.split())])


def sp_div(u, variables):
    return sum(
        [u[i].diff(xi) for i, xi in enumerate("".join(variables.split()))]
    )


def sp_jacobian(u, variables):
    return np.array([sp_grad(ui, variables) for ui in u])


ddt = lambda u: u.diff("t")
sgrad = lambda u: sp_grad(u, "xy")
sdiv = lambda u: sp_div(u, "xy")


def sp_robin_boundary(u, alpha, normals):
    return {
        tag: (u - 1.0 / alpha * np.dot(sgrad(u), n)) for tag, n in normals.items()
    }

def sp_neumann_boundary(u, normals):
    return {tag: np.dot(sgrad(u), n) for tag, n in normals.items()}


def mms_placeholder():
    mesh_: df.Mesh = df.UnitIntervalMesh(1)
    V_ = df.FunctionSpace(mesh_, "CG", 1)
    return df.Function(V_)


def expr(exp, degree, **kwargs) -> FormCoefficient:
    """Helper function to create 2D dolfin-expression of specific degree
    from a sympy-expression."""
    v = mms_placeholder()
    return ulfy.Expression(v, subs={v: exp}, degree=degree, **kwargs)  # type: ignore

