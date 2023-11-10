import logging
from pathlib import Path
from typing import List, Optional, Union, Callable

import dolfin as df
import h5py
import numpy as np
import ufl

from pantarei.domain import Domain

logger = logging.getLogger(__name__)
StrPath = str | Path


class FenicsStorage:
    def __init__(self, filepath: str | Path, mode: str):
        self.filepath = Path(filepath)
        self.mode = mode
        if mode == "w" or mode == "a":
            self.filepath.parent.mkdir(exist_ok=True, parents=True)
            if mode == "w" and df.MPI.comm_world.rank == 0:
                self.filepath.unlink(missing_ok=True)
            if mode == "a" and not self.filepath.exists():
                self.mode = "w"
            df.MPI.comm_world.barrier()
        self.hdf = df.HDF5File(df.MPI.comm_world, str(self.filepath), self.mode)

    def write_domain(self, domain: df.Mesh):
        self.hdf.write(domain, "/domain/mesh")
        if isinstance(domain, Domain):
            self.hdf.write(domain.subdomains, "/domain/subdomains")
            self.hdf.write(domain.boundaries, "/domain/boundaries")

    def read_domain(self) -> Domain:
        mesh = df.Mesh(df.MPI.comm_world)
        self.hdf.read(mesh, "/domain/mesh", False)
        n = mesh.topology().dim()
        subdomains = df.MeshFunction("size_t", mesh, n)
        boundaries = df.MeshFunction("size_t", mesh, n - 1)
        if self.hdf.has_dataset("/domain/subdomains"):
            self.hdf.read(subdomains, "/domain/subdomains")
        if self.hdf.has_dataset("/domain/boundaries"):
            self.hdf.read(boundaries, "/domain/boundaries")
        return Domain(mesh, subdomains, boundaries)

    def read_element(self, function_name: str) -> ufl.FiniteElementBase:
        signature = self.hdf.attributes(function_name)["signature"]
        return read_signature(signature)

    def write_function(
        self, function: df.Function, name: str, overwrite: bool = False
    ):
        if overwrite:
            delete_dataset(self, name)
        if not self.hdf.has_dataset("/domain"):
            self.write_domain(function.function_space().mesh())
        self.hdf.write(function, name, 0.0)

    def read_function(
        self, name: str, domain: Optional[df.Mesh] = None, idx: int = 0
    ):
        if domain is None:
            domain = self.read_domain()
        element = self.read_element(name)
        V = df.FunctionSpace(domain, element)
        u = df.Function(V, name=name)
        self.hdf.read(u, f"{name}/vector_{idx}")
        return u

    def write_checkpoint(self, function: df.Function, name: str, t: float):
        self.hdf.write(function, name, t)

    def read_checkpoint(
        self, u: df.Function, name: str, idx: int
    ) -> df.Function:
        self.hdf.read(u, f"{name}/vector_{idx}")
        return u

    def read_checkpoint_time(self, name: str, idx: int) -> float:
        return self.hdf.attributes(f"{name}/vector_{idx}")["timestamp"]

    def read_timevector(self, function_name: str) -> np.ndarray:
        num_entries = int(self.hdf.attributes(function_name)["count"])
        time = np.zeros(num_entries)
        for i in range(num_entries):
            time[i] = self.read_checkpoint_time(function_name, i)
        return time

    def close(self):
        worldsize = df.MPI.comm_world.size
        if worldsize > 1:
            logger.info(
                f"Process {df.MPI.comm_world.rank}/{worldsize-1} waiting for other\
                        processes before closing file."
            )
        df.MPI.comm_world.barrier()
        self.hdf.close()
        if df.MPI.comm_world.rank == 0:
            logger.info(f"File closed, continuing.")

    def to_xdmf(
        self,
        funcname: str,
        subnames: Union[str, List[str]],
        outputpattern: Optional[Path | Callable[[str], Path]] = None,
    ):
        fenicsstorage2xdmf(self, funcname, subnames, outputpattern)

    def to_pvd(
        self,
        funcname: str,
        subnames: Union[str, List[str]],
        outputpattern: Optional[Callable[[str], Path]] = None,
    ):
        fenicsstorage2pvd(self, funcname, subnames, outputpattern)


def fenicsstorage2xdmf(
    storage: FenicsStorage,
    funcname: str,
    subnames: str | list[str],
    outputpattern: Optional[Path | Callable[[str], Path]] = None,
):
    if outputpattern is None:
        outputpattern = (
            lambda x: storage.filepath.parent / f"{funcname}_{x}.xdmf"
        )
    elif isinstance(outputpattern, Path) or isinstance(outputpattern, str):
        temppath = Path(outputpattern)
        outputpattern = lambda x: temppath / f"{funcname}_{x}.xdmf"

    flattened = flat(subnames)
    xdmfs = {}
    for name in flattened:
        filename = outputpattern(name)
        filename.parent.mkdir(exist_ok=True, parents=True)
        xdmfs[name] = df.XDMFFile(df.MPI.comm_world, str(filename))

    times = storage.read_timevector(funcname)
    ui = storage.read_function(funcname)
    for idx, ti in enumerate(times):
        ui = storage.read_checkpoint(ui, funcname, idx)
        write_to_xdmf(xdmfs, ui, ti, subnames)

    for xdmf in xdmfs.values():
        xdmf.close()


def fenicsstorage2pvd(
    storage: FenicsStorage,
    funcname: str,
    subnames: str | list[str],
    outputpattern: Optional[Callable[[str], Path]] = None,
):
    if outputpattern is None:
        outputpattern = (
            lambda x: storage.filepath.parent / f"{funcname}_{x}.xdmf"
        )
    flattened = flat(subnames)
    pvds = {}
    for name in flattened:
        filename = outputpattern(name)
        filename.parent.mkdir(exist_ok=True, parents=True)
        pvds[name] = df.XDMFFile(df.MPI.comm_world, str(filename))

    times = storage.read_timevector(funcname)
    ui = storage.read_function(funcname)
    for idx, ti in enumerate(times):
        ui = storage.read_checkpoint(ui, funcname, idx)
        write_to_pvd(pvds, ui, ti, subnames)

    for pvd in pvds.values():
        pvd.close()


def delete_dataset(store, dataset_name):
    if df.MPI.comm_world.rank == 0:
        if store.hdf.has_dataset(dataset_name):
            with h5py.File(store.filepath, "a") as f:
                print(
                    "Deleting dataset",
                    dataset_name,
                    "from file",
                    store.filepath,
                    flush=True,
                )
                del f[dataset_name]
    df.MPI.comm_world.barrier()
    print("Barrier passed", flush=True)


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


def write_to_pvd(pvds, t, u, names):
    if isinstance(names, str):
        u.rename(names, "")
        pvds[names] << (u, t)  # type: ignore
    else:
        for uj, name in zip(u.split(deepcopy=True), names):
            write_to_pvd(pvds, t, uj, name)


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


class NullStorage(FenicsStorage):
    def __init__(self):
        pass

    def write_function(
        self, function: df.Function, name: str, overwrite: bool = False
    ):
        pass

    def write_checkpoint(self, function: df.Function, name: str, t: float):
        pass

    def close(self):
        pass
