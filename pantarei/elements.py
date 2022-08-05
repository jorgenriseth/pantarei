from ufl.finiteelement import FiniteElement, MixedElement, VectorElement


class TaylorHood(MixedElement):
    def __init__(self, mesh):
        QV = VectorElement("CG", mesh.ufl_cell(), 2)
        LP = FiniteElement("CG", mesh.ufl_cell(), 1)
        super().__init__([QV, LP])

        # Cache repr string
        if type(self) is TaylorHood:
            self._repr = "MixedElement(%s)" % (
                ", ".join(repr(e) for e in self._sub_elements),
            )
