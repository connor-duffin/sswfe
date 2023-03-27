""" Solve 2D Navier-Stokes for FEATflow Benchmark 2D-3. """
import logging

import numpy as np
import fenics as fe

# initialise the logger
logger = logging.getLogger(__name__)

# use default Fenics MPI comm (itself uses mpi4py)
comm = fe.MPI.comm_world
rank = comm.Get_rank()


class NSTwo:
    def __init__(self, mesh, control):
        """ Base class for Navier-Stokes solvers in Fenics. Uses P2-P1
        elements and is set up for the FEAT-DFG 2D-3 test case. """
        self.dt = control["dt"]
        self.t = 0.

        if type(mesh) == str:
            # read mesh from file
            logger.info("reading mesh from file")
            self.mesh = fe.Mesh()
            f = fe.XDMFFile(mesh)
            f.read(self.mesh)
        else:
            logger.info("setting mesh from object")
            self.mesh = mesh

        logger.info(f"mesh has {self.mesh.num_cells()} elements")
        self.dx = self.mesh.hmax()
        self.x = fe.SpatialCoordinate(self.mesh)
        self.x_coords = self.mesh.coordinates()
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        # use P2-P1 elements only
        self.U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        self.P = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)

        # set physical settings for all
        self.nu = 1e-3
        self.rho = 1.

        self.u_in = fe.Expression(
            ("(4 * 1.5 * sin(pi * t / 8) * x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."),
            pi=np.pi, t=0, degree=4)

        self.inflow = "near(x[0], 0)"
        self.outflow = "near(x[0], 2.2)"
        self.walls = "near(x[1], 0) || near(x[1], 0.41)"
        self.cylinder = ("on_boundary && x[0] >= 0.15 && x[0] <= 0.25 && "
                         + "x[1] >= 0.15 && x[1] <= 0.25")

    def setup_form(self):
        raise NotImplementedError

    def solve(self, t):
        raise NotImplementedError

    def setup_checkpoint(self, checkpoint_file):
        """ Set up the checkpoint file, writing the appropriate things etc. """
        logger.info(f"storing outputs in {checkpoint_file}")
        self.checkpoint = fe.HDF5File(
            self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        raise NotImplementedError

    def checkpoint_close(self):
        self.checkpoint.close()


class NSSemiImplicit(NSTwo):
    def __init__(self, mesh, control):
        """ Navier-Stokes solver which uses the Simo-Armero semi-implicit scheme;
        a linear 2nd-order timestepping scheme. """
        super().__init__(mesh=mesh, control=control)
        TH = fe.MixedElement([self.U, self.P])
        W = self.W = fe.FunctionSpace(self.mesh, TH)
        self.U_base, self.P_base = W.split()

        self.U_space = self.U_base.collapse()
        self.P_space = self.P_base.collapse()

        self.du = fe.Function(W)
        self.du_prev = fe.Function(W)
        self.du_prev_prev = fe.Function(W)

    def setup_form(self):
        no_slip = fe.Constant((0., 0.))
        bcu_inflow = fe.DirichletBC(self.W.sub(0), self.u_in, self.inflow)
        bcu_walls = fe.DirichletBC(self.W.sub(0), no_slip, self.walls)
        bcu_cyl = fe.DirichletBC(self.W.sub(0), no_slip, self.cylinder)

        # set the BC's
        self.bcu = [bcu_inflow, bcu_walls, bcu_cyl]

        nu = fe.Constant(self.nu)
        rho = fe.Constant(self.rho)
        dt = fe.Constant(self.dt)

        u, p = fe.TrialFunctions(self.W)
        v, q = fe.TestFunctions(self.W)

        u_prev, p_prev = fe.split(self.du_prev)
        u_prev_prev, p_prev_prev = fe.split(self.du_prev_prev)

        # use Simo-Armero scheme (unconditionally stable, 2nd-order accurate)
        # link: https://doi.org/10.1016/0045-7825(94)90042-6
        u_mid = 0.5 * (u + u_prev)
        p_mid = 0.5 * (p + p_prev)
        F = (fe.inner(u - u_prev, v) / dt * fe.dx  # mass term u
             + fe.inner(fe.dot(1.5 * u_prev - 0.5 * u_prev_prev,
                               fe.nabla_grad(u_mid)), v) * fe.dx  # advection
             + nu * fe.inner(fe.grad(u_mid), fe.grad(v)) * fe.dx  # dissipation
             - (1 / rho) * fe.inner(p_mid, fe.div(v)) * fe.dx   # pressure
             + fe.inner(fe.div(u_mid), q) * fe.dx)  # velocity field
        self.a, self.L = fe.lhs(F), fe.rhs(F)
        self.A = fe.assemble(self.a)
        for bc in self.bcu:
            bc.apply(self.A)

        # fe.PETScOptions.set("pc_fieldsplit_type", "additive")
        # fe.PETScOptions.set("fieldsplit_0_ksp_type", "gmres")
        # fe.PETScOptions.set("fieldsplit_1_ksp_type", "gmres")
        # fe.PETScOptions.set("fieldsplit_1_pc_type", "hypre")
        # fe.PETScOptions.set("fieldsplit_p_pc_hypre_type", "boomeramg")

        # setup krylov solver
        fe.PETScOptions.set("ksp_view")
        fe.PETScOptions.set("ksp_monitor_true_residual")
        fe.PETScOptions.set("pc_type", "fieldsplit")
        fe.PETScOptions.set("pc_fieldsplit_type", "schur")
        fe.PETScOptions.set("pc_fieldsplit_detect_saddle_point")
        fe.PETScOptions.set("pc_fieldsplit_schur_precondition", "selfp")
        fe.PETScOptions.set("fieldsplit_0_ksp_type", "preonly")
        fe.PETScOptions.set("fieldsplit_0_pc_type", "hypre")
        fe.PETScOptions.set("fieldsplit_0_pc_hypre_type", "boomeramg")
        fe.PETScOptions.set("fieldsplit_1_mat_schur_complement_ainv_type", "lump")

        self.krylov_solver = fe.PETScKrylovSolver("gmres")
        self.krylov_solver.set_operator(self.A)
        self.krylov_solver.set_from_options()

        self.krylov_solver.parameters["absolute_tolerance"] = 1e-10
        self.krylov_solver.parameters["relative_tolerance"] = 1e-7
        self.krylov_solver.parameters["maximum_iterations"] = 1000
        self.krylov_solver.parameters["report"] = False

        # now solve for du_prev to initialise (via Crank-Nicolson)
        # use direct solver for this step
        self.t += self.dt
        self.u_in.t = self.t
        u_prev_mid = 0.5 * (u_prev + u_prev_prev)
        p_prev_mid = 0.5 * (p_prev + p_prev_prev)
        F_init = (fe.inner(u_prev - u_prev_prev, v) / dt * fe.dx
                  + fe.inner(fe.dot(u_prev_mid,
                                    fe.nabla_grad(u_prev_mid)), v) * fe.dx
                  + nu * fe.inner(fe.grad(u_prev_mid), fe.grad(v)) * fe.dx
                  - (1 / rho) * fe.inner(p_prev_mid, fe.div(v)) * fe.dx
                  + fe.inner(fe.div(u_prev_mid), q) * fe.dx)
        J = fe.derivative(F_init, self.du_prev)
        fe.solve(F_init == 0, self.du_prev, bcs=self.bcu, J=J)

    def solve(self, krylov=False):
        """ Solve via classical direct method. """
        # push along time
        self.t += self.dt
        self.u_in.t = self.t

        # assemble RHS and solve via direct method (MUMPS)
        b = fe.assemble(self.L)
        for bc in self.bcu:
            bc.apply(b)

        # solve via krylov methods
        if krylov:
            self.krylov_solver.solve(self.du.vector(), b)
        else:
            fe.solve(self.A, self.du.vector(), b, "mumps")

        # set previous timestep
        self.du_prev_prev.assign(self.du_prev)
        self.du_prev.assign(self.du)

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)


class NSSplit(NSTwo):
    def __init__(self, mesh, control):
        super().__init__(mesh=mesh, control=control)
        self.U_space = fe.FunctionSpace(self.mesh, self.U)
        self.P_space = fe.FunctionSpace(self.mesh, self.P)

        # functions on u-space
        self.u = fe.Function(self.U_space)
        self.u_prev = fe.Function(self.U_space)
        self.u_prev_prev = fe.Function(self.U_space)
        self.u_star = fe.Function(self.U_space)

        # functions on p-space
        self.p = fe.Function(self.P_space)
        self.p_prev = fe.Function(self.P_space)

        # set constants
        self.t = 0.

    def setup_form(self):
        # zero velocity on bounds
        no_slip = fe.Constant((0., 0.))
        bcu_inflow = fe.DirichletBC(self.U_space, self.u_in, self.inflow)
        bcu_walls = fe.DirichletBC(self.U_space, no_slip, self.walls)
        bcu_cyl = fe.DirichletBC(self.U_space, no_slip, self.cylinder)

        # set the BC's
        self.bcu = [bcu_inflow, bcu_walls, bcu_cyl]
        self.bcp = fe.DirichletBC(self.P_space, fe.Constant(0.), self.outflow)

        nu = fe.Constant(self.nu)
        rho = fe.Constant(self.rho)
        dt = fe.Constant(self.dt)

        u = fe.TrialFunction(self.U_space)
        v = fe.TestFunction(self.U_space)
        p = fe.TrialFunction(self.P_space)
        q = fe.TestFunction(self.P_space)

        # step one: solve for u_star
        u_mid = 0.5 * (u + self.u_prev)
        F1 = (fe.inner(u - self.u_prev, v) / dt * fe.dx
              + fe.inner(fe.dot(1.5 * self.u_prev - 0.5 * self.u_prev_prev,
                                fe.nabla_grad(u_mid)), v) * fe.dx
              + nu * fe.inner(fe.grad(u_mid), fe.grad(v)) * fe.dx  # dissipation
              - (1 / rho) * fe.dot(self.p_prev, fe.div(v)) * fe.dx)   # pressure gradient
        self.a1, self.l1 = fe.lhs(F1), fe.rhs(F1)
        self.A1 = fe.assemble(self.a1)

        # step two: solve for pressure
        F2 = (0.5 * fe.inner(fe.grad(p - self.p_prev), fe.grad(q)) * fe.dx
              + rho / dt * fe.inner(fe.div(self.u_star), q) * fe.dx)
        self.a2, self.l2 = fe.lhs(F2), fe.rhs(F2)
        self.A2 = fe.assemble(self.a2)

        # step three: solve for pressure-corrected velocity
        F3 = (fe.inner(u - self.u_star, v) * fe.dx
              + (dt / rho) * 0.5 * fe.inner(fe.grad(self.p - self.p_prev), v) * fe.dx)
        self.a3, self.l3 = fe.lhs(F3), fe.rhs(F3)
        self.A3 = fe.assemble(self.a3)

    def solve(self, krylov=False):
        self.t += self.dt
        self.u_in.t = self.t

        # first solve for tentative velocity
        b1 = fe.assemble(self.l1)
        for bc in self.bcu:
            bc.apply(self.A1, b1)

        if krylov:
            fe.solve(self.A1, self.u_star.vector(), b1, "gmres", "sor")
        else:
            fe.solve(self.A1, self.u_star.vector(), b1, "mumps")

        # second solve for updated pressure
        b2 = fe.assemble(self.l2)
        self.bcp.apply(self.A2, b2)

        if krylov:
            fe.solve(self.A2, self.p.vector(), b2, "gmres", "amg")
        else:
            fe.solve(self.A2, self.p.vector(), b2, "mumps")

        # third solve for corrected velocity
        b3 = fe.assemble(self.l3)
        if krylov:
            fe.solve(self.A3, self.u.vector(), b3, "gmres", "sor")
        else:
            fe.solve(self.A3, self.u.vector(), b3, "mumps")

        # and set the previous values
        fe.assign(self.u_prev_prev, self.u_prev)
        fe.assign(self.u_prev, self.u)
        fe.assign(self.p_prev, self.p)

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.u, "/u", t)
        self.checkpoint.write(self.p, "/p", t)
