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
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.setup = control["setup"]
        logger.info("Using %s config", self.setup)
        assert self.setup in ("branson", "feat")

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
        U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        P = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, P])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        # split up function spaces for later interpolation
        self.U, self.P = W.split()
        self.U_space = self.U.collapse()
        self.P_space = self.P.collapse()

        # initialise the fcns
        self.du = fe.Function(W)
        self.du_prev = fe.Function(W)

        # storage for later
        self.du_vertices = np.copy(self.du.compute_vertex_values())

        # initialise solution objects
        self.F = None
        self.J = None
        self.bcs = None

    def setup_form(self, du, du_prev, du_prev_prev=None):
        if self.setup == "branson":
            self.nu = 1e-6
            self.rho = 1000.
        elif self.setup == "feat":
            self.nu = 1e-3
            self.rho = 1.

        nu = fe.Constant(self.nu)
        rho = fe.Constant(self.rho)
        dt = fe.Constant(self.dt)

        # define fcns
        u, p = fe.split(du)
        u_prev, p_prev = fe.split(du_prev)
        v, q = fe.TestFunctions(self.W)

        # use theta-method for timesteps
        u_mid = self.theta * u + (1 - self.theta) * u_prev
        p_mid = self.theta * p + (1 - self.theta) * p_prev
        F = (fe.inner(u - u_prev, v) / dt * fe.dx  # mass term u
             + fe.inner(fe.dot(u_mid, fe.nabla_grad(u_mid)), v) * fe.dx  # advection
             + nu * fe.inner(fe.grad(u_mid), fe.grad(v)) * fe.dx  # dissipation
             - (1 / rho) * fe.inner(p_mid, fe.div(v)) * fe.dx   # pressure gradient
             + fe.inner(fe.div(u_mid), q) * fe.dx)  # velocity field

        J = fe.derivative(F, du)
        return F, J

    def setup_bcs(self, F):
        # zero velocity on bounds
        no_slip = fe.Constant((0., 0.))

        # inflow velocity is horizontal
        if self.setup == "branson":
            self.u_in = fe.Constant((0.01, 0.))

            # inflow = "near(x[0], 0)"
            # walls = "near(x[1], 0) || near(x[1], 1.85)"
            # cylinder = ("on_boundary && x[0] >= 0.95 && x[0] <= 1.05 && "
            #             + "x[1] >= 0.875 && x[1] <= 0.975")

            # try FEAT mesh for this flow
            inflow = "near(x[0], 0)"
            walls = "near(x[1], 0) || near(x[1], 1.85)"
            cylinder = ("on_boundary && x[0] >= 2.55 && x[0] <= 2.65 && "
                        + "x[1] >= 0.875 && x[1] <= 0.975")
        elif self.setup == "feat":
            self.u_in = fe.Expression(
                ("(4 * 1.5 * sin(pi * t / 8) * x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."),
                pi=np.pi, t=0, degree=4)

            inflow = "near(x[0], 0)"
            walls = "near(x[1], 0) || near(x[1], 0.41)"
            cylinder = ("on_boundary && x[0] >= 0.15 && x[0] <= 0.25 && "
                        + "x[1] >= 0.15 && x[1] <= 0.25")

        bcu_inflow = fe.DirichletBC(self.W.sub(0), self.u_in, inflow)
        bcu_walls = fe.DirichletBC(self.W.sub(0), no_slip, walls)
        bcu_cyl = fe.DirichletBC(self.W.sub(0), no_slip, cylinder)
        bcs = [bcu_inflow, bcu_walls, bcu_cyl]

        return bcs, F

    def setup_solver(self, F, du, bcs, J):
        problem = fe.NonlinearVariationalProblem(F, du, bcs=bcs, J=J)
        solver = fe.NonlinearVariationalSolver(problem)

        # vanilla fenics solver options
        prm = solver.parameters
        # prm['newton_solver']['absolute_tolerance'] = 1E-8
        # prm['newton_solver']['relative_tolerance'] = 1E-6
        # prm['newton_solver']['convergence_criterion'] = "incremental"
        # prm['newton_solver']['maximum_iterations'] = 50
        # prm['newton_solver']['relaxation_parameter'] = 0.5

        # PETSc SNES config
        prm["nonlinear_solver"] = "snes"

        # MUMPs => direct linear solver, uncomment for direct solve:
        prm["snes_solver"]["linear_solver"] = "mumps"

        # otherwise uncomment for Krylov solve:
        # prm["snes_solver"]["line_search"] = "bt"
        # prm["snes_solver"]["linear_solver"] = "gmres"
        # prm["snes_solver"]["preconditioner"] = "jacobi"
        # logger.info(f"using {prm['snes_solver']['linear_solver']} solver with {prm['snes_solver']['preconditioner']} PC")

        # solver convergence
        prm["snes_solver"]["relative_tolerance"] = 1e-5
        prm["snes_solver"]['absolute_tolerance'] = 1e-5
        prm["snes_solver"]["maximum_iterations"] = 50
        prm["snes_solver"]['error_on_nonconvergence'] = True
        prm["snes_solver"]['krylov_solver']['nonzero_initial_guess'] = True

        # solver reporting
        prm["snes_solver"]['krylov_solver']['report'] = False
        prm["snes_solver"]['krylov_solver']['monitor_convergence'] = False

        # don't print outputs from the Newton solver
        prm["snes_solver"]["report"] = True
        return solver

    def assign_prev(self):
        # set previous timestep
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
        logger.info(f"storing outputs in {checkpoint_file}")
        self.checkpoint = fe.HDF5File(
            self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)

    def checkpoint_close(self):
        self.checkpoint.close()


class NSTwoSplit:
    def __init__(self, mesh, control):
        self.dt = control["dt"]
        self.setup = control["setup"]
        logger.info("Using %s config", self.setup)
        assert self.setup in ("branson", "feat")

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
        U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        P = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        self.U_space = fe.FunctionSpace(self.mesh, U)
        self.P_space = fe.FunctionSpace(self.mesh, P)

    def setup_form(self):
        # zero velocity on bounds
        self.u_in = fe.Constant((0.01, 0.))
        no_slip = fe.Constant((0., 0.))

        if self.setup == "branson":
            self.nu = 1e-6
            self.rho = 1000.
            self.u_in = fe.Constant((0.01, 0.))

            inflow = "near(x[0], 0)"
            walls = "near(x[1], 0) || near(x[1], 1.85)"
            cylinder = ("on_boundary && x[0] >= 2.55 && x[0] <= 2.65 && "
                        + "x[1] >= 0.875 && x[1] <= 0.975")
        elif self.setup == "feat":
            self.nu = 1e-3
            self.rho = 1.
            self.u_in = fe.Expression(
                ("(4 * 1.5 * sin(pi * t / 8) * x[1] * (0.41 - x[1])) / (0.41 * 0.41)", "0."),
                pi=np.pi, t=0, degree=4)

            inflow = "near(x[0], 0)"
            walls = "near(x[1], 0) || near(x[1], 0.41)"
            cylinder = ("on_boundary && x[0] >= 0.15 && x[0] <= 0.25 && "
                        + "x[1] >= 0.15 && x[1] <= 0.25")

        bcu_inflow = fe.DirichletBC(self.U_space, self.u_in, inflow)
        bcu_walls = fe.DirichletBC(self.U_space, no_slip, walls)
        bcu_cyl = fe.DirichletBC(self.U_space, no_slip, cylinder)
        bcu = [bcu_inflow, bcu_walls, bcu_cyl]

        nu = fe.Constant(self.nu)
        rho = fe.Constant(self.rho)
        dt = fe.Constant(self.dt)

        u = fe.TrialFunction(self.U_space)
        v = fe.TestFunction(self.U_space)
        p = fe.TrialFunction(self.P_space)
        q = fe.TestFunction(self.P_space)

        self.u_star = fe.Function(self.U_space)
        self.u_prev = fe.Function(self.U_space)
        self.p_prev = fe.Function(self.P_space)
        self.phi = fe.Function(self.P_space)

        # step one: solve for u_star
        u_mid = (u + self.u_prev) / 2
        F1 = (fe.inner(u - self.u_prev, v) / dt * fe.dx
              + fe.inner(fe.dot(self.u_prev, fe.nabla_grad(self.u_prev)), v) * fe.dx  # advection
              + nu * fe.inner(fe.grad(u_mid), fe.grad(v)) * fe.dx  # dissipation
              - (1 / rho) * fe.inner(self.p_prev, fe.div(v)) * fe.dx)   # pressure gradient
        self.a1, self.l1 = fe.lhs(F1), fe.rhs(F1)
        self.A1 = fe.assemble(self.a1)

        # step two: solve for pressure
        F2 = (fe.inner(fe.grad(p), fe.grad(q)) * fe.dx
              - rho / dt * fe.inner(fe.div(self.u_star), q) * fe.dx)
        self.a2, self.l2 = fe.lhs(F2), fe.rhs(F2)
        self.A2 = fe.assemble(self.a2)

        # step three: solve for pressure-corrected velocity
        F3 = (fe.inner(u - self.u_star, v) * fe.dx
              + rho * dt * fe.inner(fe.grad(self.phi), v) * fe.dx)
        self.a3, self.l3 = fe.lhs(F3), fe.rhs(F3)
        self.A3 = fe.assemble(self.a3)
