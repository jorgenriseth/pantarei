import numpy as np
from dolfin import Constant


class TimeKeeper(Constant):
    def __init__(self, dt: float, endtime: float):
        self.iter = 0
        self.dt = dt
        self.endtime = endtime
        super().__init__(0.0)

    def progress(self):
        self.assign(self + self.dt)  # type: ignore (Constant + float is indeed valid as assignment)
        self.iter += 1

    def is_valid(self) -> bool:
        return float(self) <= self.endtime

    def reset(self):
        self.assign(0.0)
        self.iter = 0

    def as_vector(self) -> np.ndarray:
        return np.linspace(0, (len(self) - 1) * self.dt, len(self))

    def __len__(self):
        return int(np.ceil(self.endtime / self.dt) + 1)

    def __iter__(self):
        self.reset()
        for _ in range(len(self) - 1):
            self.progress()
            yield self
