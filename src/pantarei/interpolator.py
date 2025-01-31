from pathlib import Path
from typing import Callable

import dolfin as df
import numpy as np
from ufl.algebra import Sum

from pantarei.io_utils import read_function, read_timevector


def vectordata_interpolator(
    data: list[df.Function], times: np.ndarray
) -> Callable[[float], np.ndarray]:
    dt = times[1:] - times[:-1]
    dvec = [di.vector()[:] for di in data]
    dudt = [(d1 - d0) / dti for d0, d1, dti in zip(dvec[:-1], dvec[1:], dt)]

    def call(t: float) -> np.ndarray:
        if t <= times[0]:
            return dvec[0]
        if t >= times[-1]:
            return dvec[-1]
        bin = np.digitize(t, times) - 1
        return dvec[bin] + dudt[bin] * (t - times[bin])

    return call


def fenicsfunc_interpolator(
    data: list[df.Function], times: np.ndarray
) -> Callable[[float], Sum | df.Function]:
    dt = times[1:] - times[:-1]
    dudt = [(d1 - d0) / dti for d0, d1, dti in zip(data[:-1], data[1:], dt)]  # type: ignore

    def call(t: float) -> Sum | df.Function:
        if t <= times[0]:
            return data[0]
        if t >= times[-1]:
            return data[-1]
        bin = np.digitize(t, times) - 1
        return data[bin] + dudt[bin] * (t - times[bin])

    return call


class DataInterpolator(df.Function):
    def __init__(self, data, times):
        self.times = times.copy()
        self.data = data
        super().__init__(data[0].function_space())
        self.assign(data[0])
        self.interpolator = fenicsfunc_interpolator(data, times)

    def update(self, t: float) -> df.Function:
        self.assign(self.interpolator(t))
        return self


def interpolate_from_file(filepath: Path, funcname: str, t: float):
    with df.HDF5File(df.MPI.comm_world, str(filepath), "r") as hdf:
        tvec = read_timevector(hdf, funcname)
        bin = np.digitize(t, tvec) - 1
        C = [
            read_function(hdf, funcname, idx=i) for i in range(tvec.size)[bin : bin + 2]
        ]
    interpolator = vectordata_interpolator(C, tvec[bin : bin + 2])
    u = df.Function(C[0].function_space())
    u.vector()[:] = interpolator(t)
    return u
