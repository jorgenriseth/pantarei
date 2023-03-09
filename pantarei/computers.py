from dolfin import Function
from numpy import zeros


class BaseComputer:
    """Class to perform basic computations during simulation of diffusion equation."""

    def __init__(self, function_dict):
        self.functions = function_dict
        self.initiated = False
        self.values = {}

    # def initiate(self, time, u):
    #     self.initiate = True
    #     self._create_value_dict(time)
    #     self.compute(time, u)

    def _create_value_dict(self, timekeeper):
        self.initiated = True
        self.values = {key: zeros(len(timekeeper)) for key in self.functions}

    def reset(self, timekeeper):
        self._create_value_dict(timekeeper)

    def compute(self, time, u: Function):
        if not self.initiated:
            self._create_value_dict(time)
        for key, function in self.functions.items():
            self.values[key][time.iter] = function(u)

    def __getitem__(self, item):
        return self.values[item]
