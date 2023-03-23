import logging
import os
import re
from pathlib import Path
from typing import List, Union

import dolfin as df
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
            if mode == "w" and  df.MPI.comm_world.rank == 0:
                self.filepath.unlink(missing_ok=True)
            df.MPI.comm_world.barrier()
        self.hdf = df.HDF5File(df.MPI.comm_world, str(self.filepath), mode)

    def write_domain(self, domain: df.Mesh):
        self.hdf.write(domain, "/domain/mesh")
        if isinstance(domain, Domain):
            self.hdf.write(domain.subdomains, "/domain/subdomains")
            self.hdf.write(domain.boundaries, "/domain/boundaries")

    def read_domain(self):
        mesh = df.Mesh(df.MPI.comm_world)
        self.hdf.read(
            mesh, "/domain/mesh", False
        )  # Parallell might cause troubles.
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

    def write_function(self, function: df.Function, name: str):
        if not self.hdf.has_dataset("/domain"):
            self.write_domain(function.function_space().mesh())
        self.hdf.write(function, f"{name}", 0.0)

    def read_function(self, name, domain=None, idx: int = 0):
        if domain is None:
            domain = self.read_domain()
        element = self.read_element(name)
        V = df.FunctionSpace(domain, element)
        u = df.Function(V, name=name)
        self.hdf.read(u, f"{name}/vector_{idx}")
        return u

    def write_checkpoint(self, function: df.Function, name: str, t: float):
        self.hdf.write(function, name, t)

    def read_checkpoint(self, u: df.Function, name: str, idx: int):
        self.hdf.read(u, f"{name}/vector_{idx}")
        return u

    def read_checkpoint_time(self, name: str, idx: int) -> float:
        return self.hdf.attributes(f"{name}/vector_{idx}")["timestamp"]

    def read_timevector(self, function_name: str) -> np.ndarray:
        num_entries = self.hdf.attributes(function_name)["count"]
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

    def to_xdmf(self, funcname: str, subnames: Union[str, List[str]]):
        """FIXME: Rewrite as external function taking in a FenicsStorage object."""
        xdmfs = {
            name: df.XDMFFile(
                df.MPI.comm_world,
                str(self.filepath.parent / "visual_{}.xdmf".format(name)),
            )
            for name in flat(subnames)
        }
        times = self.read_timevector(funcname)
        ui = self.read_function(funcname)
        for idx, ti in enumerate(times):
            ui = self.read_checkpoint(ui, funcname, idx)
            write_to_xdmf(xdmfs, ui, ti, subnames)
        for xdmf in xdmfs.values():
            xdmf.close()


def read_signature(signature):
    # Imported here since the signature require functions without namespace
    # but we want to avoid them in global scope.
    from dolfin import (
        FiniteElement,
        MixedElement,
        TensorElement,
        VectorElement,
        hexahedron,
        interval,
        quadrilateral,
        tetrahedron,
        triangle,
    )

    return eval(signature)


def write_to_xdmf(xdmfs, u, t, names):
    if isinstance(names, str):
        u.rename(names, "")
        xdmfs[names].write(u, t)
    else:
        for uj, name in zip(u.split(deepcopy=True), names):
            write_to_xdmf(xdmfs, uj, t, name)


def flat(pool):
    if isinstance(pool, str):
        return [pool]
    res = []
    for v in pool:
        if isinstance(v, str):
            res.append(v)
        else:
            res += flat(v)
    return res
