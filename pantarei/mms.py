from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import sympy as sp
import ulfy
from dolfin import (
    DOLFIN_EPS,
    CompiledSubDomain,
    Constant,
    Function,
    FunctionSpace,
    Mesh,
    MeshFunction,
    Point,
    RectangleMesh,
    UnitCubeMesh,
    UnitIntervalMesh,
    UnitSquareMesh,
    div,
    grad,
    inner,
)
from ufl import Coefficient

from pantarei.boundary import (
    BoundaryData,
    DirichletBoundary,
    NeumannBoundary,
    RobinBoundary,
)
from pantarei.domain import Domain
from pantarei.timekeeper import TimeKeeper


class MMSDomain(Domain):
    def __init__(self, N: int):
        subdomains = {
            1: CompiledSubDomain("x[1] <= +tol", tol=DOLFIN_EPS),
            2: CompiledSubDomain("x[1] >= -tol", tol=DOLFIN_EPS),
        }
        subboundaries = {
            1: CompiledSubDomain("near(x[0], -1) && on_boundary"),
            2: CompiledSubDomain("near(x[0], 1) && on_boundary"),
            3: CompiledSubDomain("near(x[1], -1) && on_boundary"),
            4: CompiledSubDomain("near(x[1], 1) && on_boundary"),
        }
        normals = {
            1: Constant((-1, 0)),
            2: Constant((1, 0)),
            3: Constant((0, -1)),
            4: Constant((0, 1)),
        }

        mesh = RectangleMesh(
            Point(-1, -1), Point(1, 1), N, N, diagonal="crossed"
        )
        subdomain_tags = mark_subdomains(subdomains, mesh, 0, default_value=1)
        boundary_tags = mark_subdomains(subboundaries, mesh, 1)

        super().__init__(mesh, subdomain_tags, boundary_tags)
        self.normals: Dict[int, Constant] = normals


def mark_subdomains(
    subdomains: Dict[int, CompiledSubDomain],
    mesh: Mesh,
    codim: int,
    default_value: int = 0,
):
    dim = mesh.topology().dim() - codim
    subdomain_tags = MeshFunction("size_t", mesh, dim=dim, value=default_value)
    for tag, subd in subdomains.items():
        subd.mark(subdomain_tags, tag)
    return subdomain_tags


def mms_placeholder(dim: int):
    if dim == 1:
        mesh_: Mesh = UnitIntervalMesh(1)
    elif dim == 2:
        mesh_: Mesh = UnitSquareMesh(1, 1)
    elif dim == 3:
        mesh_: Mesh = UnitCubeMesh(1, 1)
    else:
        raise ValueError(f"dim should be 1, 2 or 3 got {dim}")
    V_ = FunctionSpace(mesh_, "CG", 1)
    return Function(V_)


class MMSModelSystem(ABC):
    @staticmethod
    @abstractmethod
    def strong_form(
        u: Coefficient, coefficients: Dict[str, Coefficient], **kwargs
    ) -> Coefficient:
        pass

    @staticmethod
    @abstractmethod
    def flux_density(
        u: Coefficient, coefficients: Dict[str, Coefficient], **kwargs
    ) -> Coefficient:
        pass


class PoissonModelSystem(MMSModelSystem):
    @staticmethod
    def strong_form(
        u: Coefficient, coefficients: Dict[str, Coefficient], **kwargs
    ) -> Coefficient:
        D = coefficients["D"]
        return div(-D * grad(u))  # type: ignore

    @staticmethod
    def flux_density(
        u: Coefficient, coefficients: Dict[str, Coefficient], **kwargs
    ) -> Coefficient:
        D = coefficients["D"]
        return -D * grad(u)  # type: ignore


class DiffusionModelSystem(MMSModelSystem):
    @staticmethod
    def strong_form(
        u: Coefficient, coefficients: Dict[str, Coefficient], dudt: Coefficient
    ) -> Coefficient:
        D = coefficients["D"]
        return dudt + div(-D * grad(u))  # type: ignore

    @staticmethod
    def flux_density(
        u: Coefficient, coefficients: Dict[str, Coefficient], **kwargs
    ) -> Coefficient:
        D = coefficients["D"]
        return -D * grad(u)  # type: ignore


class MMSBoundaryBase(ABC):
    def __init__(self, tag: Union[str, int]):
        self.tag = tag

    @abstractmethod
    def get(
        self,
        u: Function,
        system: MMSModelSystem,
        normals: Dict[int, Constant],
        coefficients: Dict[str, Coefficient],
        subs: Dict[Function, sp.Expr],
        degree: int,
        time: TimeKeeper,
    ) -> BoundaryData:
        pass


class MMSDirichletBoundary(MMSBoundaryBase):
    def get(
        self,
        u: Function,
        system: MMSModelSystem,
        normals: Dict[int | str, Constant],
        coefficients: Dict[str, Coefficient],
        subs: Dict[Function, sp.Expr],
        degree: int,
        time: TimeKeeper,
    ) -> BoundaryData:
        uD = ulfy.Expression(u, subs=subs, degree=degree, t=time)
        return DirichletBoundary(uD, self.tag)


class MMSNeumannBoundary(MMSBoundaryBase):
    def get(
        self,
        u: Function,
        system: MMSModelSystem,
        normals: Dict[int | str, Constant],
        coefficients: Dict[str, Coefficient],
        subs: Dict[Function, sp.Expr],
        degree: int,
        time: TimeKeeper,
    ) -> BoundaryData:
        n = normals[self.tag]
        Fu = system.flux_density(u, coefficients)
        g = ulfy.Expression(inner(Fu, n), subs=subs, degree=degree - 1, t=time)
        return NeumannBoundary(g, self.tag)


class MMSRobinBoundary(MMSBoundaryBase):
    def __init__(self, transfer_coefficient, tag):
        super().__init__(tag)
        self.a = transfer_coefficient

    def get(
        self, u, system, normals, coefficients, subs, degree, time
    ) -> BoundaryData:
        n = normals[self.tag]
        Fu = system.flux_density(u, coefficients)
        g = ulfy.Expression(
            u - inner(Fu, n) / self.a, subs=subs, degree=degree, t=time
        )
        return RobinBoundary(self.a, g, self.tag)


@dataclass
class MMSModelCoefficients:
    source: Coefficient
    boundaries: List[BoundaryData]


def setup_mms_coefficients(
    u_sympy: sp.Expr,
    boundaries: List[MMSBoundaryBase],
    domain: MMSDomain,
    system: MMSModelSystem,
    coefficients: Dict[str, Coefficient],
    degree: int,
    time: Optional[TimeKeeper] = None,
) -> MMSModelCoefficients:
    if time is None:
        time = TimeKeeper(dt=1, endtime=1)

    dim = domain.topology().dim()
    u = mms_placeholder(dim)
    dudt = mms_placeholder(dim)

    t = sp.symbols("t")
    subs = {u: u_sympy, dudt: sp.diff(u_sympy, t)}

    source = ulfy.Expression(
        system.strong_form(u, coefficients, dudt=dudt),
        subs=subs,
        degree=degree,
        t=time,
    )
    boundary_data = [
        boundary.get(
            u, system, domain.normals, coefficients, subs, degree, time=time
        )
        for boundary in boundaries
    ]
    return MMSModelCoefficients(source, boundary_data)
