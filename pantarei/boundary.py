from abc import ABC, abstractmethod
from typing import Dict, Union, List

import dolfin as df
from dolfin import DirichletBC, Form, FunctionSpace, Measure, Mesh
from dolfin.function.argument import Argument
from ufl import inner, Coefficient, FacetNormal

from pantarei.domain import Domain


class BoundaryData(ABC):
    def __init__(self, condition_type: str, tag: Union[int, str]):
        self.type = condition_type
        self.tag = tag

    @abstractmethod
    def process(self):
        pass


class IndexedBoundaryData:
    pass


class DirichletBoundary(BoundaryData):
    def __init__(self, value, tag: Union[int, str], **kwargs):
        self.uD = value
        super().__init__("Dirichlet", tag=tag, **kwargs)

    def process(self, space: FunctionSpace, domain: Mesh) -> DirichletBC:
        if self.tag == "everywhere":
            return DirichletBC(space, self.uD, "on_boundary")
        return DirichletBC(space, self.uD, domain.boundaries, self.tag)


class IndexedDirichletBoundary:
    def __init__(self, index: int, bc: DirichletBoundary):
        self.idx = index
        self.bc = bc

    def process(self, space: FunctionSpace, domain: Domain):
        return self.bc.process(space.sub(self.idx), domain)


class VariationalBoundary(BoundaryData):
    @abstractmethod
    def variational_boundary_form(self, u: Argument, v: Argument, n: FacetNormal, ds: Measure) -> Form:
        pass

    def process(self, u: Argument, v: Argument, domain: Mesh) -> Form:
        n = FacetNormal(domain)
        if hasattr(domain, "boundaries") and domain.boundaries is not None:
            ds = Measure("ds", domain=domain, subdomain_data=domain.boundaries)
        else:
            ds = Measure("ds", domain=domain)
        return self.variational_boundary_form(u, v, n, ds)


class IndexedVariationalBoundary:
    def __init__(self, index: int, bc: VariationalBoundary):
        self.idx = index
        self.bc = bc

    def process(self, U: List[Argument], V: List[Argument], domain: Mesh) -> Form:
        return self.bc.process(U, V[self.idx], domain)  # type: ignore


class NeumannBoundary(VariationalBoundary):
    def __init__(self, value: Coefficient, tag: Union[int, str], **kwargs):
        self.g = value
        super().__init__("Neumann", tag=tag, **kwargs)

    def variational_boundary_form(self, _: Argument, v: Argument, n: FacetNormal, ds: Measure) -> Form:
        return inner(self.g, v) * ds(self.tag)  # type: ignore (seemingly wrong)


class RobinBoundary(VariationalBoundary):
    def __init__(self, coeff, value, tag, **kwargs):
        self.a = coeff
        self.g = value
        super().__init__("Robin", tag=tag, **kwargs)

    def variational_boundary_form(self, u: Argument, v: Argument, _: FacetNormal, ds: Measure) -> Form:
        return self.a * (u - self.g) * v * ds(self.tag)


class TractionBoundary(VariationalBoundary):
    def __init__(self, value, tag: int, **kwargs):
        self.g = value
        super().__init__("Traction", tag=tag, **kwargs)

    def variational_boundary_form(self, _: Argument, test: Argument, n: FacetNormal, ds: Measure):
        return inner(self.g, test) * ds(self.tag)


def process_dirichlet(space: FunctionSpace, domain: Mesh, boundaries: List[BoundaryData]) -> List[DirichletBC]:
    return [
        bc.process(space, domain) for bc in boundaries if isinstance(bc, (DirichletBoundary, IndexedDirichletBoundary))
    ]


def process_boundary_forms(trial: Argument, test: Argument, domain: Mesh, boundaries: List[BoundaryData]) -> Form:
    return sum([bc.process(trial, test, domain) for bc in boundaries if isinstance(bc, (VariationalBoundary, IndexedVariationalBoundary))])  # type: ignore


def indexed_boundary_conditions(bcs: Dict[int, List[BoundaryData]]) -> List[IndexedBoundaryData]:
    bcs_out = []
    for idx, idx_bcs in bcs.items():
        bcs_out += [IndexedDirichletBoundary(idx, bc) for bc in idx_bcs if isinstance(bc, DirichletBoundary)]
        bcs_out += [IndexedVariationalBoundary(idx, bc) for bc in idx_bcs if isinstance(bc, VariationalBoundary)]
    return bcs_out
