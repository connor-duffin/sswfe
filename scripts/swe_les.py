import fenics as fe


class LES:
    def __init__(self, mesh, fs, u, density, smagorinsky_coefficient):
        """ Smagorinsky Large Eddy Simulation, done in Fenics.

        Implementation uses the same code as in the LES module in
        firedrake-fluids: https://github.com/firedrakeproject/firedrake-fluids,
        and is inspired by similar in the LES module, given in Oasis:
        https://github.com/mikaem/Oasis

        To use this object, initialize it and extract the self.eddy_viscosity
        variable, to be used in Fenics forms. This variable is filled in after
        calling self.solve().
        """
        self.mesh = mesh
        self.function_space = fs

        self.eddy_viscosity = fe.Function(self.function_space)

        F = self.form(u, density, smagorinsky_coefficient)
        self.problem = fe.LinearVariationalProblem(fe.lhs(F),
                                                   fe.rhs(F),
                                                   self.eddy_viscosity,
                                                   bcs=[])
        self.solver = fe.LinearVariationalSolver(self.problem)
        self.solver.parameters["linear_solver"] = "gmres"
        self.solver.parameters["preconditioner"] = "jacobi"

    def strain_rate_tensor(self, u):
        S = 0.5 * (fe.grad(u) + fe.grad(u).T)
        return S

    def form(self, u, density, smagorinsky_coefficient):
        w = fe.TestFunction(self.function_space)
        eddy_viscosity = fe.TrialFunction(self.function_space)

        filter_width = (fe.CellVolume(self.mesh)) ** (1. / 2)

        strain_tensor = self.strain_rate_tensor(u)
        mag_S = fe.sqrt(2. * fe.inner(strain_tensor, strain_tensor))
        rhs = (smagorinsky_coefficient * filter_width)**2 * mag_S
        F = (fe.inner(w, eddy_viscosity) - fe.inner(w, rhs)) * fe.dx
        return F

    def solve(self):
        self.solver.solve()
        return self.eddy_viscosity
