import dolfin as df
import numpy as np


def function_interpolator(data, times):
    dt = times[1:] - times[:-1]
    dudt = [(d1 - d0) / dti for d0, d1, dti in zip(data[:-1], data[1:], dt)]

    def call(t):
        if t <= times[0]:
            return data[0]
        if t >= times[-1]:
            return data[-1]
        bin = np.digitize(t, times) - 1
        return data[bin] + dudt[bin] * (t - times[bin])

    return call


def vectordata_interpolator(data, times):
    dt = times[1:] - times[:-1]
    dudt = [
        (d1.vector() - d0.vector()) / dti
        for d0, d1, dti in zip(data[:-1], data[1:], dt)
    ]

    def call(t):
        if t <= times[0]:
            return data[0].vector()
        if t >= times[-1]:
            return data[-1].vector()
        bin = np.digitize(t, times) - 1
        return data[bin].vector() + dudt[bin].vector() * (t - times[bin])

    return call


class DataInterpolator(df.Function):
    def __init__(self, data, times):
        self.space = data[-1].function_space()
        super().__init__(self.space)
        self.data = data
        self.times = times
        dt = [t1 - t0 for t0, t1 in zip(times[:-1], times[1:])]
        self.dudt = [
            df.project((d1 - d0) / dt, self.space)
            for d0, d1, dt in zip(data[:-1], data[1:], dt)
        ]

    def update(self, t: float) -> df.Function:
        self.vector()[:] = (self.interpolate(t)).vector()
        # self.assign(self.interpolate(t))
        return self

    def interpolate(self, t):
        if t <= self.times[0]:
            return self.data[0]
        if t >= self.times[-1]:
            return self.data[-1]
        bin = np.digitize(t, self.times) - 1
        return df.project(
            self.data[bin] + self.dudt[bin] * (t - self.times[bin]), self.space
        )
