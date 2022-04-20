from dataclasses import dataclass
from dolfin import Mesh
from dolfin.cpp.mesh import MeshFunctionSizet


@dataclass
class Domain:
    mesh: Mesh
    subdomains: MeshFunctionSizet
    boundaries: MeshFunctionSizet


def unpack_domain(domain: Domain):
    return domain.mesh, domain.subdomains, domain.boundaries

