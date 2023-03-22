import dolfin as df


def assign_mixed_function(p, V, compartments):
    """Create a function in a mixed function-space with sub-function being
    assigned from a dictionray of functions living in the subspaces."""
    P = df.Function(V)
    subspaces = [V.sub(idx).collapse() for idx, _ in enumerate(compartments)]
    Pint = [df.interpolate(p[idx_j], Vj) for idx_j, (_, Vj) in enumerate(zip(compartments, subspaces))]
    assigner = df.FunctionAssigner(V, subspaces)
    assigner.assign(P, Pint)
    return P
