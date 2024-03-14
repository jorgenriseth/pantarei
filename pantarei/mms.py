import dolfin as df
import numpy as np
import sympy
from pantarei.domain import Domain
from pantarei.utils import FormCoefficient

# 3D-mesh = df.BoxMesh(df.Point(-1, -1, -1), df.Point(1, 1, 1), 10, 10, 10)


class MMSInterval(Domain):
    def __init__(self, N: int):
        subdomains = {
            1: df.CompiledSubDomain("abs(x[0]) <= 0.8 + tol", tol=df.DOLFIN_EPS),
            2: df.CompiledSubDomain("abs(x[0]) >= 0.8 - tol", tol=df.DOLFIN_EPS),
        }
        subboundaries = {
            1: df.CompiledSubDomain("near(x[0], -1) && on_boundary"),
            2: df.CompiledSubDomain("near(x[0], 1) && on_boundary"),
        }
        normals = {1: -1.0, 2: 1.0}

        mesh = df.IntervalMesh(N, -1.0, 1.0)
        subdomain_tags = mark_subdomains(subdomains, mesh, 0, default_value=1)
        boundary_tags = mark_subdomains(subboundaries, mesh, 1)

        super().__init__(mesh, subdomain_tags, boundary_tags)
        self.normals = normals


class MMSSquare(Domain):
    def __init__(self, N: int):
        subdomains = {
            1: df.CompiledSubDomain(
                "max(abs(x[0]), abs(x[1])) <= 0.8 + tol", tol=df.DOLFIN_EPS
            ),
            2: df.CompiledSubDomain(
                "max(abs(x[0]), abs(x[1])) >= 0.8 - tol", tol=df.DOLFIN_EPS
            ),
        }
        subboundaries = {
            1: df.CompiledSubDomain("near(x[0], -1) && on_boundary"),
            2: df.CompiledSubDomain("near(x[0], 1) && on_boundary"),
            3: df.CompiledSubDomain("near(x[1], -1) && on_boundary"),
            4: df.CompiledSubDomain("near(x[1], 1) && on_boundary"),
        }
        normals = {
            1: np.array([-1.0, +0.0]),
            2: np.array([+1.0, +0.0]),
            3: np.array([+0.0, -1.0]),
            4: np.array([+0.0, +1.0]),
        }

        mesh = df.RectangleMesh(
            df.Point(-1, -1), df.Point(1, 1), N, N, diagonal="crossed"
        )
        subdomain_tags = mark_subdomains(subdomains, mesh, 0, default_value=1)
        boundary_tags = mark_subdomains(subboundaries, mesh, 1)

        super().__init__(mesh, subdomain_tags, boundary_tags)
        self.normals = normals


def mark_subdomains(
    subdomains: dict[int, df.CompiledSubDomain],
    mesh: df.Mesh,
    codim: int,
    default_value: int = 0,
):
    dim = mesh.topology().dim() - codim
    subdomain_tags = df.MeshFunction("size_t", mesh, dim=dim, value=default_value)
    for tag, subd in subdomains.items():
        subd.mark(subdomain_tags, tag)
    return subdomain_tags


def sp_grad(u, variables) -> np.ndarray:
    return np.array([u.diff(xi) for xi in "".join(variables.split())])


def sp_div(u, variables):
    return sum([u[i].diff(xi) for i, xi in enumerate("".join(variables.split()))])


def sp_jacobian(u, variables):
    return np.array([sp_grad(ui, variables) for ui in u])


class SympyDiffOperators:
    def __init__(self, variables: str):
        self.vars = variables

    def dt(self, u):
        return u.diff("t")

    def grad(self, u):
        return sp_grad(u, self.vars)

    def div(self, u):
        return sp_div(u, self.vars)


def sp_robin_boundary(u, flux, normals, k):
    return {tag: (u - 1.0 / k * np.dot(flux, n)) for tag, n in normals.items()}


def sp_neumann_boundary(u, flux, normals):
    return {tag: np.dot(flux, n) for tag, n in normals.items()}


def mms_placeholder():
    mesh_: df.Mesh = df.UnitSquareMesh(1, 1)
    V_ = df.FunctionSpace(mesh_, "CG", 1)
    return df.Function(V_)


def expr(expr, degree, subs=None, **kwargs) -> FormCoefficient:
    if subs is None:
        subs = {"x": ("x0", "x[0]"), "y": ("x1", "x[1]"), "z": ("x2", "x[2]")}
    print(sympy.printing.ccode(expr))
    code_expr = expr.subs({sym: sym_subs[0] for sym, sym_subs in subs.items()})
    code = sympy.printing.ccode(code_expr)
    for sym_subs in subs.values():
        code = code.replace(sym_subs[0], sym_subs[1])

    return df.Expression(code, degree=degree, **kwargs)
