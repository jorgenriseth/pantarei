from ufl import Cell
from ufl.finiteelement import FiniteElement, MixedElement, VectorElement


class TaylorHood(MixedElement):
    def __init__(self, cell_type):
        QV = VectorElement("CG", Cell(cell_type), 2)
        LP = FiniteElement("CG", Cell(cell_type), 1)
        super().__init__([QV, LP])

        # Cache repr string
        if type(self) is TaylorHood:
            self._repr = "MixedElement(%s)" % (
                ", ".join(repr(e) for e in self._sub_elements),
            )
