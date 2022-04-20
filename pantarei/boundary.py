from fenics import DirichletBC
from ufl import inner
from abc import ABC, abstractmethod
from dolfin import TestFunction, FacetNormal, Measure
from typing import Union


class BoundaryData(ABC):
    def __init__(self, condition_type: str, idx: Union[int, str], boundary_name: str = None):
        self.type = condition_type
        self.idx = idx
        if boundary_name is None:
            self.name = "boundary_" + str(idx)
        else:
            self.name = boundary_name

    @abstractmethod
    def process(self, domain, space):
        pass


class DirichletBoundary(BoundaryData):
    def __init__(self, value, idx: Union[int, str], **kwargs):
        self.uD = value
        super().__init__("Dirichlet", idx=idx, **kwargs)

    def process(self, domain, space):
        if self.idx == "everywhere":
            return DirichletBC(space, self.uD, "on_boundary")
        return DirichletBC(space, self.uD, domain.boundaries, self.idx)


class VariationalBoundary(BoundaryData):
    @abstractmethod
    def variational_boundary_form(self, n, v, ds):
        pass

    def process(self, domain, space):
        v = TestFunction(space)
        n = FacetNormal(domain.mesh)
        ds = Measure("ds", domain=domain.mesh, subdomain_data=domain.boundaries)
        return self.variational_boundary_form(n, v, ds)


class TractionBoundary(VariationalBoundary):
    def __init__(self, value, idx: int, **kwargs):
        self.g = value
        super().__init__("Traction", idx=idx, **kwargs)

    def variational_boundary_form(self, n, v, ds):
        return inner(self.g * n, v) * ds(self.idx)


def process_dirichlet(domain, space, boundaries):
    return [bc.process(domain, space) for bc in boundaries if isinstance(bc, DirichletBoundary)]


def process_boundary_forms(domain, space, boundaries):
    return sum([bc.process(domain, space) for bc in boundaries if isinstance(bc, VariationalBoundary)])
