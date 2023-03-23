from typing import Dict, Callable
from dolfin import Function
from numpy import zeros

from pantarei.timekeeper import TimeKeeper


class BaseComputer:
    """Class to perform basic computations during simulation of diffusion
    equation."""

    def __init__(self, function_dict: Dict[str, Callable]):
        self.functions = function_dict
        self.initiated = False
        self.values = {}

    def _create_value_dict(self, time: TimeKeeper) -> None:
        self.initiated = True
        self.values = {key: zeros(len(time)) for key in self.functions}

    def reset(self, time: TimeKeeper) -> None:
        self._create_value_dict(time)

    def compute(self, time: TimeKeeper, u: Function) -> None:
        if not self.initiated:
            self._create_value_dict(time)
        for key, function in self.functions.items():
            self.values[key][time.iter] = function(u)

    def __getitem__(self, item):
        return self.values[item]
