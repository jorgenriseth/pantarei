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

class FenicsStorage:
    def __init__(self, filepath: Union[str, Path], mode: str):
        self.filepath = Path(filepath).resolve()
        self.mode = mode
        if mode == "w":
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            # FIXME: May cause problems with parallell
            if self.filepath.exists():
                logger.info(f"Replacing existing file {self.filepath}")
                os.remove(self.filepath)
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

    def write_function_element(self, function):
        signature = function.function_space().element().signature()
        petsc_signature = encode_signature(signature)
        self.hdf.write(petsc_signature, f"{function.name()}/element")

    def read_element(self, function_name, mpi_wrapper):
        petsc_signature = df.Vector(df.MPI.comm_self)
        self.hdf.read(petsc_signature, f"{function_name}/element", False)
        signature = decode_signature(petsc_signature)
        return read_signature(signature)

    def write_function(self, function):
        if not self.hdf.has_dataset("/domain"):
            self.write_domain(function.function_space().mesh())
        self.write_function_element(function)
        self.hdf.write(function.vector(), f"{function.name()}/vector_{0}")

    def read_function(self, name, domain=None):
        if domain is None:
            domain = self.read_domain()
        element = self.read_element(name, domain.mpi_comm())
        V = df.FunctionSpace(domain, element)
        u = df.Function(V, name=name)
        self.hdf.read(u.vector(), f"{name}/vector_{0}", False)
        return u

    def write_checkpoint(self, function, idx):
        if idx == 0:
            self.write_function(function)
        else:
            self.hdf.write(function.vector(), f"{function.name()}/vector_{idx}")

    def read_checkpoint(self, u, idx):
        self.hdf.read(u.vector(), f"{u.name()}/vector_{idx}", False)
        return u

    def write_timevector(self, timevector, function_name=None):
        vec = df.Vector(df.MPI.comm_self, timevector.size)
        vec[:] = timevector
        if function_name is None:
            self.hdf.write(vec, "/timevector")
        else:
            self.hdf.write(vec, f"{function_name}/timevector")

    def read_timevector(self, function_name=None):
        if function_name is None:
            timevector = df.Vector()
            self.hdf.read(timevector, "/timevector", False)
        else:
            timevector = df.Vector()
            self.hdf.read(timevector, f"{function_name}/timevector", False)
        return timevector[:]

    def close(self):
        self.hdf.close()


def encode_signature(signature: str):
    arr = np.array(bytearray(signature.encode()))
    petv = df.Vector(df.MPI.comm_self, arr.size)
    petv[:] = arr
    return petv

def decode_signature(petv: df.Vector):
    return bytearray(petv[:].astype(np.uint8)).decode()

def read_signature(signature):
    # Imported here since the signature require functions without namespace
    # but we want to avoid them in global scope.
    from dolfin import MixedElement, FiniteElement, VectorElement, TensorElement
    from dolfin import interval, triangle, tetrahedron, quadrilateral, hexahedron
    return eval(signature)


def signature_to_element(signature: str) -> df.FiniteElement:
    # Write a fucntion
    match = re.match(r"FiniteElement\('([a-zA-Z ]+)', ([a-z]+), (\d)\)", signature)
    if match is None:
        raise ValueError(f"Could not parse signature {signature}")
    element_family, cell_type, degree = match.groups()
    cell = ufl.Cell(cell_type)
    return df.FiniteElement(element_family, cell, int(degree))

