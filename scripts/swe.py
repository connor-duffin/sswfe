""" Solve the Shallow-water equations in non-conservative form. """
import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser


class PiecewiseIC(fe.UserExpression):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def eval(self, values, x):
        if x[0] < self.L + fe.DOLFIN_EPS:
            values[0] = 5.
        else:
            values[0] = 0.


class ShallowOne:
    def __init__(self, control):
        # settings: L, nu, C
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.simulation = control["simulation"]

        if self.simulation == "dam_break":
            self.L = 2000
            self.nu = 1.0
            self.C = 0.
        elif self.simulation == "tidal_flow":
            self.L = 14_000
            self.nu = 1.0
            self.C = 0.
        else:
            print("simulation setup not recognised")
            raise ValueError

        # setup mesh and function spaces
        self.mesh = fe.IntervalMesh(self.nx, 0., self.L)
        self.x = fe.SpatialCoordinate(self.mesh)
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        U = fe.FiniteElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        self.x_coords = self.mesh.coordinates()
        self.x_dofs = self.W.tabulate_dof_coordinates()
        self.u_dofs = self.W.sub(0).dofmap().dofs()
        self.h_dofs = self.W.sub(1).dofmap().dofs()

        self.x_dofs_u = self.x_dofs[self.u_dofs, :]
        self.x_dofs_h = self.x_dofs[self.h_dofs, :]

        # HACK: introduce new function space to construct interpolant
        V = fe.FunctionSpace(self.mesh, "CG", 1)

        if self.simulation == "dam_break":
            self.H = fe.Constant(5.0)

            p = PiecewiseIC(self.L // 2)
            ic = fe.interpolate(p, V)
        elif self.simulation == "tidal_flow":
            H = fe.Expression(
                "50.5 - 40 * x[0] / L - 10 * sin(pi * (4 * x[0] / L - 0.5))",
                L=self.L,
                degree=2)
            self.H = fe.interpolate(H, V)
        else:
            print("simulation setup not currently enabled")
            raise ValueError

        self.du = fe.Function(W)
        u, h = fe.split(self.du)

        self.du_prev = fe.Function(W)
        u_prev, h_prev = fe.split(self.du_prev)

        v_u, v_h = fe.TestFunctions(W)

        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        C = fe.Constant(self.C)
        dt = fe.Constant(self.dt)

        self.theta = 1.0
        u_mid = self.theta * u + (1 - self.theta) * u_prev
        h_mid = self.theta * h + (1 - self.theta) * h_prev
        u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

        self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx +
                  fe.inner(h - h_prev, v_h) / dt * fe.dx +
                  u_prev * u_mid.dx(0) * v_u * fe.dx +
                  nu * 2 * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx +
                  g * h_mid.dx(0) * v_u * fe.dx + C * u_mag * u_mid * v_u /
                  (self.H + h_mid) * fe.dx -
                  ((self.H + h_mid) * u_mid * v_h.dx(0)) * fe.dx)
        self.J = fe.derivative(self.F, self.du)

        def _right(x, on_boundary):
            return x[0] >= (self.L - fe.DOLFIN_EPS)

        def _left(x, on_boundary):
            return x[0] <= fe.DOLFIN_EPS

        self._right = _right
        self._left = _left

        # set the IC's
        if self.simulation == "dam_break":
            for init in [self.du, self.du_prev]:
                fe.assign(init.sub(1), ic)

            h_left = fe.Constant(5.0)
            h_right = fe.Constant(0.0)

            bc0 = fe.DirichletBC(W.sub(1), h_left, self._left)
            bc1 = fe.DirichletBC(W.sub(1), h_right, self._right)

            # add in boundary terms to the weak form
            self.F += v_h * u_prev * (
                self.H + h_right) * fe.ds - v_h * u_prev * (self.H + h_left) * fe.ds
        elif self.simulation == "tidal_flow":
            u_right = fe.Constant(0.0)

            bc0 = fe.DirichletBC(W.sub(1), self.tidal_bc(0), self._left)
            bc1 = fe.DirichletBC(W.sub(0), u_right, self._right)

        self.bcs = [bc0, bc1]

    @staticmethod
    def tidal_bc(t):
        return 4 - 4 * np.sin(np.pi * ((4 * t) / 86_400 + 0.5))

    def solve(self, t):
        if self.simulation == "tidal_flow":
            self.bcs[0] = fe.DirichletBC(self.W.sub(1), self.tidal_bc(t),
                                         self._left)

        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)
        fe.assign(self.du_prev, self.du)


class ShallowTwo:
    def __init__(self, control):
        # settings: L, nu, C
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.theta = control["theta"]

        self.mesh = fe.UnitSquareMesh(self.nx, self.nx)
        self.x = fe.SpatialCoordinate(self.mesh)
        self.x_coords = self.mesh.coordinates()
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        # split up function spaces for interpolation
        self.U, self.H = W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.du = fe.Function(W)
        u, h = fe.split(self.du)

        self.du_prev = fe.Function(W)
        u_prev, h_prev = fe.split(self.du_prev)

        v_u, v_h = fe.TestFunctions(W)

        self.nu = 0.6
        self.C = 0.0025
        self.H = 50.

        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        C = fe.Constant(self.C)
        dt = fe.Constant(self.dt)

        f_u = fe.Function(self.U_space)
        f_h = fe.Function(self.H_space)

        # f_u_1 = "cos(x[0]) * (cos(x[1]) * (cos(x[1]) + sin(pow(x[0], 2))) + sin(x[1])*(11.0 - sin(x[0])*sin(x[1]) + (0.0025 * sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2)))/(50.0 + sin(x[0])*sin(x[1]))))"

        # f_u_2 = "-1.2*cos(pow(x[0], 2)) + 0.6*cos(x[1]) + 9.8*cos(x[1])*sin(x[0]) + 2.4 * pow(x[0], 2) * sin(pow(x[0], 2)) + 2*x[0]*cos(x[0])*cos(pow(x[0], 2))*sin(x[1]) - (cos(x[1]) + sin(pow(x[0], 2)))*sin(x[1]) + (0.0025*sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2))*(cos(x[1]) + sin(pow(x[0], 2))))/(50.0 + sin(x[0])*sin(x[1]))"

        # f_h = "cos(x[1]) * sin(x[0]) * (cos(x[1]) + sin(pow(x[0], 2))) - 50*(1 + sin(x[0]))*sin(x[1]) + (cos(2*x[0]) - sin(x[0]))*pow(sin(x[1]), 2)"

        f_u_exact = fe.Expression(("cos(x[0]) * (cos(x[1]) * (cos(x[1]) + sin(pow(x[0], 2))) + sin(x[1])*(11.0 - sin(x[0])*sin(x[1]) + (0.0025 * sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2)))/(50.0 + sin(x[0])*sin(x[1]))))",
                                   "-1.2*cos(pow(x[0], 2)) + 0.6*cos(x[1]) + 9.8*cos(x[1])*sin(x[0]) + 2.4 * pow(x[0], 2) * sin(pow(x[0], 2)) + 2*x[0]*cos(x[0])*cos(pow(x[0], 2))*sin(x[1]) - (cos(x[1]) + sin(pow(x[0], 2)))*sin(x[1]) + (0.0025*sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2))*(cos(x[1]) + sin(pow(x[0], 2))))/(50.0 + sin(x[0])*sin(x[1]))"
                                ), degree=4)
        f_h_exact = fe.Expression("cos(x[1]) * sin(x[0]) * (cos(x[1]) + sin(pow(x[0], 2))) - 50*(1 + sin(x[0]))*sin(x[1]) + (cos(2*x[0]) - sin(x[0]))*pow(sin(x[1]), 2)", degree=4)

        f_u.assign(f_u_exact)
        f_h.interpolate(f_h_exact)

        self.theta = 1.0
        u_mid = self.theta * u + (1 - self.theta) * u_prev
        h_mid = self.theta * h + (1 - self.theta) * h_prev
        u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

        self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx +
                  fe.inner(h - h_prev, v_h) / dt * fe.dx +
                  fe.inner(fe.dot(u_mid, fe.nabla_grad(u_mid)), v_u) * fe.dx  # advection
                  + nu * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx  # dissipation
                  + g * fe.inner(fe.grad(h_mid), v_u) * fe.dx  # surface term
                  + C * u_mag * fe.inner(u_mid, v_u) / (self.H + h_mid) * fe.dx  # friction term
                  - fe.inner((self.H + h_mid) * u_mid, fe.grad(v_h)) * fe.dx
                  - fe.inner(f_u, v_u) * fe.dx - fe.inner(f_h, v_h) * fe.dx)
        self.J = fe.derivative(self.F, self.du)

        self.u_exact = fe.Expression(("cos(x[0]) * sin(x[1])",
                                      "sin(pow(x[0], 2)) + cos(x[1])"), degree=4)
        self.h_exact = fe.Expression("sin(x[0]) * sin(x[1])", degree=4)

        def boundary(x, on_boundary):
            return on_boundary

        bc_u = fe.DirichletBC(self.W.sub(0), self.u_exact, boundary)
        bc_h = fe.DirichletBC(self.W.sub(1), self.h_exact, boundary)
        self.bcs = [bc_u, bc_h]

    def solve(self, t, check_steady_state=False):
        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)
        fe.assign(self.du_prev, self.du)

    @staticmethod
    def steady_state(u, u_prev, tol=1e-6):
        diff = fe.errornorm(u, u_prev)
        if diff <= tol:
            return True
        else:
            return False
