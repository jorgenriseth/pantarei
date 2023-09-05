from typing import Callable
import dolfin as df
import numpy as np

from ufl.algebra import Sum


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
        super().__init__(data[0].function_space())
        self.interpolator = fenicsfunc_interpolator(data, times)

    def update(self, t: float) -> df.Function:
        self.assign(self.interpolator(t))
        return self
