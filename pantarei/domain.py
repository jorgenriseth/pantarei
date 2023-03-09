from dolfin import Mesh, MeshFunction


class Domain(Mesh):
    def __init__(
        self, mesh: Mesh, subdomains: MeshFunction, boundaries: MeshFunction
    ):
        super().__init__(mesh)
        self.subdomains = transfer_meshfunction(self, subdomains)
        self.boundaries = transfer_meshfunction(self, boundaries)


def transfer_meshfunction(
    newmesh: Mesh, meshfunc: MeshFunction
) -> MeshFunction:
    newtags = MeshFunction("size_t", newmesh, dim=meshfunc.dim())  # type: ignore
    newtags.set_values(meshfunc)  # type: ignore
    return newtags
