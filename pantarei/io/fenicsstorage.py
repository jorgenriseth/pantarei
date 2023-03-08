import os
import dolfin as df
import logging
import re
from typing import Union
from pathlib import Path

import numpy as np
import ufl

from pantarei.domain import Domain

logger = logging.getLogger(__name__)
StrPath = Union[str, Path]


class FenicsStorage:
    def __init__(self, filepath: StrPath, mode: str):
        self.filepath = Path(filepath).resolve()
        self.mode = mode
        if mode == "w" or mode == "a":
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            df.MPI.comm_world.barrier()
        self.hdf = df.HDF5File(df.MPI.comm_world, str(self.filepath), mode)

    def write_domain(self, domain: df.Mesh):
        self.hdf.write(domain, "/domain/mesh")
        if isinstance(domain, Domain):
            self.hdf.write(domain.subdomains, "/domain/subdomains")
            self.hdf.write(domain.boundaries, "/domain/boundaries")

    def read_domain(self):
        mesh = df.Mesh(df.MPI.comm_world)
        self.hdf.read(mesh, "/domain/mesh", False)  # Parallell might cause troubles.
        n = mesh.topology().dim()
        subdomains = df.MeshFunction("size_t", mesh, n)
        boundaries = df.MeshFunction("size_t", mesh, n - 1)
        if self.hdf.has_dataset("/domain/subdomains"):
            self.hdf.read(subdomains, "/domain/subdomains")
        if self.hdf.has_dataset("/domain/boundaries"):
            self.hdf.read(boundaries, "/domain/boundaries")
        return Domain(mesh, subdomains, boundaries)

    def read_element(self, function_name):
        signature = self.hdf.attributes(function_name)["signature"]
        return read_signature(signature)

    def write_function(self, function):
        if not self.hdf.has_dataset("/domain"):
            self.write_domain(function.function_space().mesh())
        self.hdf.write(function, f"{function.name()}")

    def read_function(self, name, domain=None):
        if domain is None:
            domain = self.read_domain()
        element = self.read_element(name)
        V = df.FunctionSpace(domain, element)
        u = df.Function(V, name=name)
        self.hdf.read(u, f"{name}/vector_{0}")
        return u

    def write_checkpoint(self, function: df.Function, name: str, t: float):
        self.hdf.write(function, name, t)

    def read_checkpoint(self, u: df.Function, name: str, idx: int):
        self.hdf.read(u, f"{name}/vector_{idx}")
        return u

    def read_checkpoint_time(self, name: str, idx: int):
        return self.hdf.attributes(f"{name}/vector_{idx}")["timestamp"]

    def read_timevector(self, function_name):
        num_entries = self.hdf.attributes("/timevector")["count"]
        time = np.zeros(num_entries)
        for i in range(num_entries):
            time[i] = self.read_checkpoint_time(function_name, i)
        return time

    def close(self):
        logger.info(
            f"Process {df.MPI.comm_world.rank} waiting for other\
                    processes before closing file."
        )
        df.MPI.comm_world.barrier()
        self.hdf.close()


def read_signature(signature):
    # Imported here since the signature require functions without namespace
    # but we want to avoid them in global scope.
    from dolfin import MixedElement, FiniteElement, VectorElement, TensorElement
    from dolfin import interval, triangle, tetrahedron, quadrilateral, hexahedron
    return eval(signature)
