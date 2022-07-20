""" Solve the Shallow-water equations in non-conservative form. """
import h5py
import logging

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser
from numpy.testing import assert_allclose

logger = logging.getLogger(__name__)
fe.set_log_level(40)  # only print emergencies


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
            raise ValueError("Simulation steup not recognised")

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
                self.H + h_right) * fe.ds - v_h * u_prev * (self.H +
                                                            h_left) * fe.ds
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
    def __init__(self, mesh, control):
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.simulation = control["simulation"]
        self.integrate_continuity_by_parts = control["integrate_continuity_by_parts"]

        if type(mesh) == str:
            # read mesh from file
            self.mesh = fe.Mesh()
            f = fe.XDMFFile(mesh)
            f.read(self.mesh)
        else:
            self.mesh = mesh

        self.dx = self.mesh.hmax()
        self.x = fe.SpatialCoordinate(self.mesh)
        self.x_coords = self.mesh.coordinates()
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        # use P2-P1 elements only
        U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        # split up function spaces for later interpolation
        self.U, self.H = W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.du = fe.Function(W)
        u, h = fe.split(self.du)

        self.du_prev = fe.Function(W)
        u_prev, h_prev = fe.split(self.du_prev)

        v_u, v_h = fe.TestFunctions(W)

        # storage for later
        self.du_vertices = np.copy(self.du.compute_vertex_values())

        if self.simulation == "mms":
            self.nu = 0.6
            self.C = 0.0025
            self.H = 50.
        elif self.simulation in ["cylinder", "laminar"]:
            self.nu = 1e-6
            self.C = 0.
            self.H = 0.073

        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        C = fe.Constant(self.C)
        dt = fe.Constant(self.dt)

        self.f_u = fe.Function(self.U_space)
        self.f_h = fe.Function(self.H_space)

        u_mid = self.theta * u + (1 - self.theta) * u_prev
        h_mid = self.theta * h + (1 - self.theta) * h_prev
        u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

        self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx
                  + fe.inner(h - h_prev, v_h) / dt * fe.dx
                  + fe.inner(fe.dot(u_prev, fe.nabla_grad(u_mid)), v_u) * fe.dx  # advection
                  + nu * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx  # dissipation
                  + g * fe.inner(fe.grad(h_mid), v_u) * fe.dx  # surface term
                  + C * u_mag * fe.inner(u_mid, v_u) / (self.H + h_mid) * fe.dx  # friction term
                  - fe.inner(self.f_u, v_u) * fe.dx - fe.inner(self.f_h, v_h) * fe.dx)

        # add in continuity term
        if self.integrate_continuity_by_parts:
            self.F += -fe.inner((self.H + h_mid) * u_mid, fe.grad(v_h)) * fe.dx
        else:
            self.F += fe.inner(fe.div((self.H + h_mid) * u_mid), v_h) * fe.dx

        self.J = fe.derivative(self.F, self.du)

        if self.simulation == "mms":
            f_u_exact = fe.Expression((
                "cos(x[0]) * (cos(x[1]) * (cos(x[1]) + sin(pow(x[0], 2))) + sin(x[1])*(11.0 - sin(x[0])*sin(x[1]) + (0.0025 * sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2)))/(50.0 + sin(x[0])*sin(x[1]))))",
                "-1.2*cos(pow(x[0], 2)) + 0.6*cos(x[1]) + 9.8*cos(x[1])*sin(x[0]) + 2.4 * pow(x[0], 2) * sin(pow(x[0], 2)) + 2*x[0]*cos(x[0])*cos(pow(x[0], 2))*sin(x[1]) - (cos(x[1]) + sin(pow(x[0], 2)))*sin(x[1]) + (0.0025*sqrt(pow(cos(x[1]) + sin(pow(x[0], 2)), 2) + pow(cos(x[0])*sin(x[1]), 2))*(cos(x[1]) + sin(pow(x[0], 2))))/(50.0 + sin(x[0])*sin(x[1]))"),
                degree=4)
            f_h_exact = fe.Expression(
                "cos(x[1]) * sin(x[0]) * (cos(x[1]) + sin(pow(x[0], 2))) - 50*(1 + sin(x[0]))*sin(x[1]) + (cos(2*x[0]) - sin(x[0]))*pow(sin(x[1]), 2)",
                degree=4)

            self.f_u.assign(f_u_exact)
            self.f_h.interpolate(f_h_exact)

            self.u_exact = fe.Expression(
                ("cos(x[0]) * sin(x[1])", "sin(pow(x[0], 2)) + cos(x[1])"),
                degree=4)
            self.h_exact = fe.Expression("sin(x[0]) * sin(x[1])", degree=4)

            def boundary(x, on_boundary):
                return on_boundary

            bc_u = fe.DirichletBC(self.W.sub(0), self.u_exact, boundary)
            bc_h = fe.DirichletBC(self.W.sub(1), self.h_exact, boundary)
            self.bcs = [bc_u, bc_h]
        elif self.simulation in ["cylinder", "laminar"]:
            # basic BC's
            # TODO: take in mesh parameterisations as argument
            u_in = fe.Constant((0.01, 0.))
            u_out = u_in
            no_slip = fe.Constant((0., 0.))

            inflow = "near(x[0], 0)"
            walls = "near(x[1], 0) || near(x[1], 1.85)"

            # outflow = "near(x[0], 6)"
            # bcu_outflow = fe.DirichletBC(self.W.sub(0), u_out, outflow)

            bcu_inflow = fe.DirichletBC(self.W.sub(0), u_in, inflow)
            bcu_walls = fe.DirichletBC(self.W.sub(0), no_slip, walls)
            self.bcs = [bcu_inflow, bcu_walls]

            # need to include surface integrals if we integrate by parts
            # only left and right boundaries matter, as the rest are zero (no-slip condition)
            if self.integrate_continuity_by_parts:
                class LeftBoundary(fe.SubDomain):
                    def inside(self, x, on_boundary):
                        tol = 1E-14  # tolerance for coordinate comparisons
                        return on_boundary and abs(x[0]) < tol

                Gamma_1 = LeftBoundary()
                Gamma_1.mark(self.boundaries, 1)  # mark with tag 1 for LHS

                class RightBoundary(fe.SubDomain):
                    def inside(self, x, on_boundary):
                        tol = 1E-14  # tolerance for coordinate comparisons
                        return on_boundary and abs(x[0] - 6) < tol

                Gamma_3 = RightBoundary()
                Gamma_3.mark(self.boundaries, 2)  # mark with tag 2 for RHS
                ds = fe.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
                self.F += (
                    v_h * (self.H + h_mid) * (-u_in) * ds(1)  # LHS bounds
                    + v_h * (self.H + h_mid) * (u_out) * ds(2)  # RHS bounds
                )

            # TODO: take in cylinder mesh parameterisations as an argument/option
            # 0.925 is the centre of the domain
            if self.simulation == "cylinder":
                cylinder = "on_boundary && x[0] >= 2.55 && x[0] <= 2.65 && x[1] >= 0.875 && x[1] <= 0.975"
                self.bcs.append(
                    fe.DirichletBC(self.W.sub(0), no_slip, cylinder))

        problem = fe.NonlinearVariationalProblem(
            self.F, self.du, bcs=self.bcs, J=self.J)
        self.solver = fe.NonlinearVariationalSolver(problem)

        # solver options
        prm = self.solver.parameters
        prm["nonlinear_solver"] = "snes"
        prm["snes_solver"]["line_search"] = "bt"
        prm["snes_solver"]["linear_solver"] = "gmres"
        prm["snes_solver"]["preconditioner"] = "ilu"

        # don't print outputs from the Newton solver
        prm["snes_solver"]["report"] = False

        # JIC we want to tweak tolerances
        # prm["snes_solver"]["absolute_tolerance"] = 1e-6
        # prm["snes_solver"]["relative_tolerance"] = 1e-3

    def solve(self):
        self.solver.solve()
        fe.assign(self.du_prev, self.du)

    @staticmethod
    def steady_state(u, u_prev, tol=1e-6):
        diff = fe.errornorm(u, u_prev)
        if diff <= tol:
            return True
        else:
            return False

    def set_curr_vector(self, du_vec):
        self.du.vector().set_local(du_vec)

    def set_prev_vector(self, du_vec):
        self.du_prev.vector().set_local(du_vec)

    def get_vertex_values(self):
        n_vertices = len(self.mesh.coordinates())
        self.du_vertices[:] = self.du.compute_vertex_values()
        return (self.du_vertices[:n_vertices],
                self.du_vertices[n_vertices:(2 * n_vertices)],
                self.du_vertices[(2 * n_vertices):])

    def setup_checkpoint(self, checkpoint_file):
        """ Set up the checkpoint file, writing the appropriate things etc. """
        self.checkpoint = h5py.File(checkpoint_file, "w")
        self.checkpoint.create_dataset("x_vertices",
                                       data=self.mesh.coordinates())
        self.checkpoint.create_dataset("t", data=0.)
        self.checkpoint.create_dataset(
            "du", data=np.zeros_like(self.du.vector().get_local()))

    def checkpoint_load(self, checkpoint_file):
        """ Load from the checkpoint file. Returns the current time of the simulation. """
        self.checkpoint = h5py.File(checkpoint_file, "r+")

        # check that things are okay
        assert_allclose(self.checkpoint["x_vertices"][:].shape,
                        self.mesh.coordinates().shape)
        assert self.checkpoint["du"][:].shape == self.du.vector().get_local().shape
        assert self.checkpoint["t"][()] >= 0.

        # set both current and prev for SNES initialisation
        self.set_curr_vector(self.checkpoint["du"][:])
        self.set_prev_vector(self.checkpoint["du"][:])
        return self.checkpoint["t"][()]

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint["t"][()] = t
        self.checkpoint["du"][:] = self.du.vector().get_local()

    def checkpoint_close(self):
        self.checkpoint.close()
