import dolfin as df


def assign_mixed_function(p, V, compartments):
    """Create a function in a mixed function-space with sub-function being
    assigned from a dictionray of functions living in the subspaces."""
    P = df.Function(V)
    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [
        df.interpolate(p[j], Vj)
        for j, Vj in zip(compartments, subspaces)
    ]
    assigner = df.FunctionAssigner(V, subspaces)
    assigner.assign(P, Pint)
    return P


def rescale_function(u: df.Function, value: float):
    v = u.vector()
    v *= value / df.assemble(u * df.dx)
    return u
