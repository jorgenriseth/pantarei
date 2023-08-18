from dolfin import FunctionSpace, Measure
from dolfin import Function, TrialFunctions, TestFunctions, FacetNormal, solve
from dolfin import TimeSeries, XDMFFile
from ufl import lhs, rhs, grad, div, dx

from pantarei.meshprocessing import geo2hdf
from pantarei.elements import TaylorHood


class PulsatingPoiseuille2D(FenicsProblem):
    def __init__(self):
        self.load_mesh()
        self.define_functionspace()

    def define_functionspace(self):
        TH = TaylorHood(self.mesh)
        self.W = FunctionSpace(self.mesh, TH)

    def load_mesh(self):
        geofile = ".pipe2d.geo"
        hdffile = "./pipe2d.h5"
        geo2hdf(geofile, hdffile)
        self.mesh, self.subdomains, self.boundaries = hdf2fenics(hdffile)

    def process_boundaries(self, boundaries):
        v, _ = TestFunctions(self.W)
        n = FacetNormal(self.mesh)
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

        # Process Dirichlet boundaries
        self.bcs = [bc.process(self.W.sub(0), self.boundaries)
                    for bc in boundaries if bc.type == "Dirichlet"]

        # Process boundary integral terms
        self.boundary_forms = [bc.process(n, v, ds)
                               for bc in boundaries if bc.type == "Traction"]

    def variational_form(self):
        u, p = TrialFunctions(self.W)
        v, q = TestFunctions(self.W)
        self.F = (inner(grad(u), grad(v)) - p*div(v) - q*div(u)) * dx

    def solve(self, boundaries, time_end, dt):
        # Preparations
        self.variational_form()
        self.process_boundaries(boundaries)
        self.F += sum(self.boundary_forms)
        a, L = lhs(self.F), rhs(self.F)
        UP = Function(self.W)

        # Results storage
        timeseries_velocity = TimeSeries(
            "pulsating_poiseuille/velocity_series")

        xdmf_velocity = XDMFFile("pulsating_poiseuille/velocity.xdmf")
        xdmf_pressure = XDMFFile("pulsating_poiseuille/pressure.xdmf")

        t = 0.
        while t < time_end:
            solve(a == L, UP, self.bcs)

            # Store solutions
            u, p = UP.split(deepcopy=True)
            timeseries_velocity.store(u.vector(), t)
            xdmf_velocity.write(u, t)
            xdmf_pressure.write(p, t)

            t += dt
            self.set_time(boundaries, t)

        # Close
        xdmf_velocity.close()
        xdmf_pressure.close()

    def set_time(self, boundaries, t):
        for bc in boundaries:
            bc.set_time(t)


def solve_pulsating_poiseuille():
    # Pulsating Pressure
    p_inflow = Expression("10. * sin(t) + 5. * sin(3.*t)",
                          t=Constant(0.), degree=1)

    # Define boundary-conditions
    boundary_conditions = [
        DirichletBoundary(Constant((0., 0.)), 1, boundary_name="bottom wall"),
        DirichletBoundary(Constant((0., 0.)), 3, boundary_name="top wall"),
        TractionBoundary(Constant(0.), 2, boundary_name="outflow"),
        TractionBoundary(p_inflow, 4, boundary_name="inflow")
    ]

    # Define Initial Condition
    problem = PulsatingPoiseuille2D()
    problem.solve(boundary_conditions, 30, 0.05)
    problem.visualize()


if __name__ == "__main__":
    solve_pulsating_poiseuille()
