from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, TypeAlias

import dolfin as df
from dolfin import inner
from dolfin.function.argument import Argument as Argument

from pantarei.domain import Domain
from pantarei.utils import FormCoefficient

BoundaryTag: TypeAlias = int | Literal["everywhere"]


class BoundaryData(ABC):
    def __init__(self, condition_type: str, tag: BoundaryTag):
        self.type = condition_type
        self.tag = tag

    @abstractmethod
    def process(self):
        pass


class IndexedBoundaryData:
    pass


class DirichletBoundary(BoundaryData):
    def __init__(self, value: FormCoefficient, tag: BoundaryTag):
        self.uD = value
        super().__init__("Dirichlet", tag=tag)

    def process(self, space: df.FunctionSpace) -> df.DirichletBC:
        if self.tag == "everywhere":
            return df.DirichletBC(space, self.uD, "on_boundary")
        return df.DirichletBC(space, self.uD, space.mesh().boundaries, self.tag)


class IndexedDirichletBoundary(IndexedBoundaryData):
    def __init__(self, index: int, bc: DirichletBoundary):
        self.idx = index
        self.bc = bc

    def process(self, space: df.FunctionSpace):
        return self.bc.process(space.sub(self.idx))


class VariationalBoundary(BoundaryData):
    @abstractmethod
    def variational_boundary_form(
        self, u: Argument, v: Argument, ds: df.Measure
    ) -> df.Form:
        pass

    def process(
        self, u: Argument, v: Argument, V: Optional[df.FunctionSpace] = None
    ) -> df.Form:
        if V is None:
            V = u.function_space()
        domain = V.mesh()
        if isinstance(domain, Domain):
            ds = df.Measure("ds", domain=domain, subdomain_data=domain.boundaries)
        else:
            ds = df.Measure("ds", domain=domain)
        return self.variational_boundary_form(u, v, ds)


class IndexedVariationalBoundary(IndexedBoundaryData):
    def __init__(self, index: int, bc: VariationalBoundary):
        self.idx = index
        self.bc = bc

    def process(self, U: list[Argument], V: list[Argument]) -> df.Form:
        return self.bc.process(U[self.idx], V[self.idx], V=U.function_space())

    def __call__(self, U: list[Argument], V: list[Argument]):
        return self.process(U, V)


def process_dirichlet(
    space: df.FunctionSpace, boundaries: List[BoundaryData]
) -> List[df.DirichletBC]:
    return [
        bc.process(space)
        for bc in boundaries
        if isinstance(bc, (DirichletBoundary, IndexedDirichletBoundary))
    ]


def process_boundary_forms(
    trial: Argument,
    test: Argument,
    boundaries: List[BoundaryData],
) -> df.Form:
    return sum(
        [
            bc.process(trial, test)
            for bc in boundaries
            if isinstance(bc, (VariationalBoundary, IndexedVariationalBoundary))
        ],
    )  # type: ignore


def indexed_boundary_conditions(
    bcs: Dict[int, List[BoundaryData]],
) -> List[IndexedBoundaryData]:
    bcs_out = []
    for idx, idx_bcs in bcs.items():
        bcs_out += [
            IndexedDirichletBoundary(idx, bc)
            for bc in idx_bcs
            if isinstance(bc, DirichletBoundary)
        ]
        bcs_out += [
            IndexedVariationalBoundary(idx, bc)
            for bc in idx_bcs
            if isinstance(bc, VariationalBoundary)
        ]
    return bcs_out


class NeumannBoundary(VariationalBoundary):
    def __init__(self, value: FormCoefficient, tag: BoundaryTag, **kwargs):
        self.uN = value
        super().__init__("Neumann", tag=tag, **kwargs)

    def variational_boundary_form(
        self, _: Argument, v: Argument, ds: df.Measure
    ) -> df.Form:
        return inner(self.uN, v) * ds(self.tag)  # type: ignore


class RobinBoundary(VariationalBoundary):
    def __init__(
        self,
        coeff: FormCoefficient,
        uR: FormCoefficient,
        tag: BoundaryTag,
        **kwargs,
    ):
        self.a = coeff
        self.uR = uR
        super().__init__("Robin", tag=tag, **kwargs)

    def variational_boundary_form(
        self, u: Argument, v: Argument, ds: df.Measure
    ) -> df.Form:
        return inner(inner(self.a, u - self.uR), v) * ds(self.tag)  # type: ignore
