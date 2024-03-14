import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import dolfin as df
import numpy as np
import ufl

from pantarei.domain import Domain
from pantarei.io_utils import close as hdfclose
from pantarei.io_utils import (
    read_checkpoint,
    read_checkpoint_time,
    read_domain,
    read_element,
    read_function,
    read_timevector,
    write_checkpoint,
    write_domain,
    write_element,
    write_function,
)
from pantarei.utils import print_progress

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
        return write_domain(self.hdf, domain)

    def read_domain(self) -> Domain:
        return read_domain(self.hdf)

    def write_element(self, funcname: str, function_space: df.FunctionSpace) -> str:
        return write_element(self.hdf, funcname, function_space)

    def read_element(self, function_name: str) -> ufl.FiniteElementBase:
        return read_element(self.hdf, function_name)

    def write_function(self, function: df.Function, name: str, overwrite: bool = False):
        if overwrite:
            return write_function(self.hdf, function, name, self.filepath)
        return write_function(self.hdf, function, name)

    def read_function(self, name: str, domain: Optional[df.Mesh] = None, idx: int = 0):
        return read_function(self.hdf, name, domain, idx)

    def write_checkpoint(self, function: df.Function, name: str, t: float) -> None:
        write_checkpoint(self.hdf, function, name, t)

    def read_checkpoint(self, u: df.Function, name: str, idx: int) -> df.Function:
        return read_checkpoint(self.hdf, u, name, idx)

    def read_checkpoint_time(self, name: str, idx: int) -> float:
        return read_checkpoint_time(self.hdf, name, idx)

    def read_timevector(self, function_name: str) -> np.ndarray:
        return read_timevector(self.hdf, function_name)

    def close(self):
        hdfclose(self.hdf)

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
        outputpattern = lambda x: storage.filepath.parent / f"{funcname}_{x}.xdmf"
    elif isinstance(outputpattern, Path) or isinstance(outputpattern, str):
        temppath = Path(outputpattern)
        outputpattern = lambda x: temppath / f"{funcname}_{x}.xdmf"

    flattened = flat(subnames)
    xdmfs = {}
    for name in flattened:
        filename = outputpattern(name)
        filename.parent.mkdir(exist_ok=True, parents=True)
        xdmfs[name] = df.XDMFFile(df.MPI.comm_world, str(filename))

    try:
        times = storage.read_timevector(funcname)
    except RuntimeError as e:
        print(str())
        if "HDF5 attribute type unknown" in str(e):
            times = np.array([0.0])
        else:
            raise e
    ui = storage.read_function(funcname)
    for idx, ti in enumerate(times):
        print_progress(idx, len(times), rank=df.MPI.comm_world.rank)
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
        outputpattern = lambda x: storage.filepath.parent / f"{funcname}_{x}.xdmf"
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

    def write_function(self, function: df.Function, name: str, overwrite: bool = False):
        pass

    def write_checkpoint(self, function: df.Function, name: str, t: float):
        pass

    def close(self):
        pass
