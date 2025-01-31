from typing import Callable, Dict

import numpy as np
from dolfin import Function

from pantarei.timekeeper import TimeKeeper


class NullComputer:
    def compute(self, *args, **kwargs):
        pass


class BaseComputer:
    """Class to perform basic computations during simulation of diffusion
    equation."""

    def __init__(self, function_dict: Dict[str, Callable]):
        self.functions = function_dict
        self.initiated = False
        self.values = {}

    def _create_value_dict(self, time: TimeKeeper) -> None:
        self.initiated = True
        timevec = time.as_vector()
        self.values = {
            "time": timevec,
            **{key: np.zeros(len(timevec)) for key in self.functions},
        }

    def init_from_vector(self, timevec: np.ndarray):
        self.initiated = True
        self.values = {
            "time": timevec,
            **{key: np.zeros(len(timevec)) for key in self.functions},
        }

    def reset(self, time: TimeKeeper) -> None:
        self._create_value_dict(time)

    def compute(self, time: TimeKeeper, u: Function, *args) -> None:
        if not self.initiated:
            self._create_value_dict(time)
        for key, function in self.functions.items():
            self.values[key][time.iter] = function(u, *args)

    def compute_from_index(self, iter: int, u: Function, *args):
        for key, function in self.functions.items():
            self.values[key][iter] = function(u, *args)

    def __getitem__(self, item):
        return self.values[item]
