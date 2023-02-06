""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe
from swe_les import LES

# initialise the logger
logger = logging.getLogger(__name__)

# use default Fenics MPI comm (itself uses mpi4py)
comm = fe.MPI.comm_world
rank = comm.Get_rank()


H_INIT_BUMP = fe.Expression("exp(- pow(2 * (x[0] - 10), 2)) / 40", degree=2)


def stress_term(u, v, nu, nu_t, laplacian=False):
    if laplacian:
        return nu * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    else:
        return ((nu + nu_t) * fe.inner(fe.grad(u) + fe.grad(u).T, fe.grad(v)) * fe.dx
                - (nu + nu_t) * (2. / 3.) * fe.inner(fe.div(u) * fe.Identity(2), fe.grad(v)) * fe.dx)


def continuity_term(H, h, u, v, ibp=False):
    if ibp:
        return -fe.inner((H + h) * u, fe.grad(v)) * fe.dx
    else:
        return fe.inner(fe.div((H + h) * u), v) * fe.dx


class PiecewiseIC(fe.UserExpression):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def eval(self, values, x):
        if x[0] < self.L + fe.DOLFIN_EPS:
            values[0] = 5.
        else:
            values[0] = 0.


class BumpTopo(fe.UserExpression):
    def __init__(self, c, L):
        super().__init__()
        self.c = c
        self.L = L

        self.lower = c - 2
        self.upper = c + 2

    def eval(self, values, x):
        if x[0] <= self.upper and x[0] >= self.lower:
            values[0] = 0.5 * (0.6 + 0.1 * (x[0] - self.c)**2)
        else:
            values[0] = 0.5


# subdomain for periodic boundary condition
class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < fe.DOLFIN_EPS and x[0] > -fe.DOLFIN_EPS
                    and on_boundary)

    # map right boundary to left boundary
    def map(self, x, y):
        y[0] = x[0] - 10


class ShallowOneLinear:
    def __init__(self, control, params):
        # read in parameters
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.nu = params["nu"]
        self.bump_centre = params["bump_centre"]

        # HACK: trying out setting default length for now
        self.L = 25.

        # setup mesh and function spaces
        self.mesh = fe.IntervalMesh(self.nx, 0., self.L)
        self.x = fe.SpatialCoordinate(self.mesh)

        U = fe.FiniteElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        self.W = fe.FunctionSpace(self.mesh, TH,
                                  constrained_domain=PeriodicBoundary())
        self.U, self.H = self.W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.x_coords = self.mesh.coordinates()
        self.n_vertices = len(self.x_coords)
        self.x_dofs = self.W.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        self.u_dofs = self.W.sub(0).dofmap().dofs()
        self.h_dofs = self.W.sub(1).dofmap().dofs()

        self.x_dofs_u = self.x_dofs[self.u_dofs, :]
        self.x_dofs_h = self.x_dofs[self.h_dofs, :]

        # HACK: introduce new function space to construct interpolant
        g = fe.Constant(9.8)

        def bounds(x, on_boundary):
            return on_boundary

        self.bcs = fe.DirichletBC(self.W.sub(1), fe.Constant(0.), bounds)

        H = BumpTopo(self.bump_centre, self.L)
        self.H = fe.interpolate(H, self.H_space)
        u, h = fe.TrialFunctions(self.W)
        v_u, v_h = fe.TestFunctions(self.W)

        self.du_prev = fe.Function(self.W)
        fe.assign(self.du_prev.sub(1),
                  fe.interpolate(H_INIT_BUMP, self.H_space))
        u_prev, h_prev = fe.split(self.du_prev)

        dt = fe.Constant(self.dt)
        self.theta = control["theta"]
        u_theta = self.theta * u + (1 - self.theta) * u_prev
        h_theta = self.theta * h + (1 - self.theta) * h_prev

        # (inviscid flow)
        self.F = (fe.inner(h - h_prev, v_h) / dt * fe.dx
                  + (self.H * u_theta).dx(0) * v_h * fe.dx
                  + fe.inner(u - u_prev, v_u) / dt * fe.dx
                  + g * h_theta.dx(0) * v_u * fe.dx)

        self.du = fe.Function(self.W)
        self.du_vertices = np.copy(self.du.compute_vertex_values())

        self.a, self.L = fe.system(self.F)
        self.problem = fe.LinearVariationalProblem(self.a, self.L, self.du, bcs=self.bcs)
        self.solver = fe.LinearVariationalSolver(self.problem)

    def solve(self):
        self.solver.solve()
        fe.assign(self.du_prev, self.du)

    def compute_energy(self):
        u, h = fe.split(self.du)
        return fe.assemble(u**2 * fe.dx)

    def set_curr_vector(self, du_vec):
        self.du.vector().set_local(du_vec)

    def set_prev_vector(self, du_vec):
        self.du_prev.vector().set_local(du_vec)

    def get_vertex_values(self):
        self.du_vertices[:] = self.du.compute_vertex_values()
        return (self.du_vertices[:self.n_vertices],
                self.du_vertices[self.n_vertices:])

    def get_vertex_values_prev(self):
        self.du_vertices[:] = self.du_prev.compute_vertex_values()
        return (self.du_vertices[:self.n_vertices],
                self.du_vertices[self.n_vertices:])

    def setup_checkpoint(self, checkpoint_file):
        """ Set up the checkpoint file, writing the appropriate things etc. """
        logger.info(f"storing outputs in {checkpoint_file}")
        self.checkpoint = fe.HDF5File(self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)

    def checkpoint_close(self):
        self.checkpoint.close()


class ShallowOne:
    def __init__(self, control, params):
        self.nx = control["nx"]
        self.dt = control["dt"]
        self.simulation = control["simulation"]

        if self.simulation == "dam_break":
            self.L = 2000
        elif self.simulation == "tidal_flow":
            self.L = 14_000
        elif self.simulation == "immersed_bump":
            self.bump_centre = params["bump_centre"]
            self.L = 25.
        else:
            raise ValueError("Simulation setup not recognised")

        # read in parameter values
        self.nu = params["nu"]

        # setup mesh and function spaces
        self.mesh = fe.IntervalMesh(self.nx, 0., self.L)
        self.x = fe.SpatialCoordinate(self.mesh)
        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        U = fe.FiniteElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        self.W = fe.FunctionSpace(self.mesh, TH)
        self.U, self.H = self.W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.x_coords = self.mesh.coordinates()
        self.n_vertices = len(self.x_coords)
        self.x_dofs = self.W.tabulate_dof_coordinates()
        self.n_dofs = self.x_dofs.shape[0]

        self.u_dofs = self.W.sub(0).dofmap().dofs()
        self.h_dofs = self.W.sub(1).dofmap().dofs()

        self.x_dofs_u = self.x_dofs[self.u_dofs, :]
        self.x_dofs_h = self.x_dofs[self.h_dofs, :]

        if self.simulation == "dam_break":
            self.H = fe.Constant(5.0)

            p = PiecewiseIC(self.L // 2)
            ic = fe.interpolate(p, self.H_space)
        elif self.simulation == "tidal_flow":
            H = fe.Expression(
                "50.5 - 40 * x[0] / L - 10 * sin(pi * (4 * x[0] / L - 0.5))",
                L=self.L,
                degree=2)
            self.H = fe.interpolate(H, self.H_space)
        elif self.simulation == "immersed_bump":
            # set H
            H = BumpTopo(self.bump_centre, self.L)
            self.H = fe.interpolate(H, self.H_space)

        self.du = fe.Function(self.W)
        self.du_vertices = np.copy(self.du.compute_vertex_values())
        u, h = fe.split(self.du)

        self.du_prev = fe.Function(self.W)
        u_prev, h_prev = fe.split(self.du_prev)

        v_u, v_h = fe.TestFunctions(self.W)

        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        dt = fe.Constant(self.dt)

        self.theta = control["theta"]
        u_theta = self.theta * u + (1 - self.theta) * u_prev
        h_theta = self.theta * h + (1 - self.theta) * h_prev
        self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx
                  + u_prev * u_theta.dx(0) * v_u * fe.dx
                  + nu * 2 * fe.inner(fe.grad(u_theta), fe.grad(v_u)) * fe.dx
                  + g * h_theta.dx(0) * v_u * fe.dx
                  + fe.inner(h - h_prev, v_h) / dt * fe.dx
                  + ((self.H + h_theta) * u_theta).dx(0) * v_h * fe.dx)

        # assemble RHS
        self.rhs = (- u * u.dx(0) * v_u * fe.dx
                    - 2 * nu * fe.inner(fe.grad(u), fe.grad(v_u)) * fe.dx
                    - g * h.dx(0) * v_u * fe.dx
                    - ((self.H + h) * u).dx(0) * v_h * fe.dx)
        self.rhs_derivative = fe.derivative(self.rhs, self.du)
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

            bc_h_left = fe.DirichletBC(self.W.sub(1), h_left, self._left)
            bc_h_right = fe.DirichletBC(self.W.sub(1), h_right, self._right)
            self.bcs = [bc_h_left, bc_h_right]

            # add in boundary terms to the weak form
            self.F += v_h * u_prev * (
                self.H + h_right) * fe.ds - v_h * u_prev * (self.H +
                                                            h_left) * fe.ds
        elif self.simulation == "tidal_flow":
            u_right = fe.Constant(0.0)

            bc_h_left = fe.DirichletBC(self.W.sub(1), self.tidal_bc(0), self._left)
            bc_u_right = fe.DirichletBC(self.W.sub(0), u_right, self._right)
            self.bcs = [bc_h_left, bc_u_right]
        elif self.simulation == "immersed_bump":
            for init in [self.du, self.du_prev]:
                ic = fe.interpolate(H_INIT_BUMP, self.H_space)
                fe.assign(init.sub(1), ic)

            def bounds(x, on_boundary):
                return on_boundary

            self.bcs = fe.DirichletBC(self.W.sub(1), fe.Constant(0.), bounds)

    @staticmethod
    def tidal_bc(t):
        return 4 - 4 * np.sin(np.pi * ((4 * t) / 86_400 + 0.5))

    def compute_energy(self):
        u, h = fe.split(self.du)
        return fe.assemble(u**2 * fe.dx)

    def solve(self, t, set_prev=True):
        if self.simulation == "tidal_flow":
            self.bcs[0] = fe.DirichletBC(self.W.sub(1), self.tidal_bc(t),
                                         self._left)

        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)

        if set_prev:
            fe.assign(self.du_prev, self.du)

    def set_curr_vector(self, du_vec):
        self.du.vector().set_local(du_vec)

    def set_prev_vector(self, du_vec):
        self.du_prev.vector().set_local(du_vec)

    def get_vertex_values(self):
        self.du_vertices[:] = self.du.compute_vertex_values()
        return (self.du_vertices[:self.n_vertices],
                self.du_vertices[self.n_vertices:])

    def get_vertex_values_prev(self):
        self.du_vertices[:] = self.du_prev.compute_vertex_values()
        return (self.du_vertices[:self.n_vertices],
                self.du_vertices[self.n_vertices:])

    def setup_checkpoint(self, checkpoint_file):
        """ Set up the checkpoint file, writing the appropriate things etc. """
        logger.info(f"storing outputs in {checkpoint_file}")
        self.checkpoint = fe.HDF5File(self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)

    def checkpoint_close(self):
        self.checkpoint.close()


class ShallowTwo:
    def __init__(self, mesh, control):
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.simulation = control["simulation"]
        self.integrate_continuity_by_parts = control["integrate_continuity_by_parts"]
        self.use_laplacian = control["laplacian"]
        self.use_les = control["les"]

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
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        # split up function spaces for later interpolation
        self.U, self.H = W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.du = fe.Function(W)
        self.du_prev = fe.Function(W)
        self.du_prev_prev = fe.Function(W)

        # storage for later
        self.du_vertices = np.copy(self.du.compute_vertex_values())

        if self.simulation == "mms":
            self.nu = 0.6
            self.C = 0.0025
            self.H = 50.
        elif self.simulation in ["cylinder", "laminar"]:
            self.nu = 1e-6
            self.C = 0.
            self.H = 0.16

        # initialise solution objects
        self.F = None
        self.J = None
        self.bcs = None
        self.solver = None

    def setup_form(self, du, du_prev, du_prev_prev=None, bdf2=False):
        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        C = fe.Constant(self.C)
        dt = fe.Constant(self.dt)

        u, h = fe.split(du)
        u_prev, h_prev = fe.split(du_prev)

        if du_prev_prev is not None:
            u_prev_prev, h_prev_prev = fe.split(du_prev_prev)

        u_mag = fe.sqrt(fe.dot(u_prev, u_prev))

        v_u, v_h = fe.TestFunctions(self.W)

        self.f_u = fe.Function(self.U_space)
        self.f_h = fe.Function(self.H_space)

        if bdf2:
            u_mid = u
            h_mid = h
            F = (3/2 * fe.inner(u - 4/3 * u_prev + 1/3 * u_prev_prev, v_u) / dt * fe.dx
                 + 3/2 * fe.inner(h - 4/3 * h_prev + 1/3 * h_prev_prev, v_h) / dt * fe.dx
                 + fe.inner(fe.dot(u, fe.nabla_grad(u)), v_u) * fe.dx  # advection
                 + C * u_mag * fe.inner(u, v_u) / (self.H + h_prev) * fe.dx  # friction term
                 + g * fe.inner(fe.grad(h), v_u) * fe.dx  # surface term
                 - fe.inner(self.f_u, v_u) * fe.dx
                 - fe.inner(self.f_h, v_h) * fe.dx)
        else:
            u_mid = self.theta * u + (1 - self.theta) * u_prev
            h_mid = self.theta * h + (1 - self.theta) * h_prev
            F = (fe.inner(u - u_prev, v_u) / dt * fe.dx  # mass term u
                 + fe.inner(h - h_prev, v_h) / dt * fe.dx  # mass term h
                 + fe.inner(fe.dot(u_mid, fe.nabla_grad(u_mid)), v_u) * fe.dx  # advection
                 + C * u_mag * fe.inner(u_mid, v_u) / (self.H + h_prev) * fe.dx  # friction term
                 + g * fe.inner(fe.grad(h_mid), v_u) * fe.dx  # surface term
                 - fe.inner(self.f_u, v_u) * fe.dx
                 - fe.inner(self.f_h, v_h) * fe.dx)

        # add in (parameterised) dissipation effects
        if self.use_les:
            self.les = LES(mesh=self.mesh, fs=self.H_space, u=u_mid, density=1.0, smagorinsky_coefficient=0.164)
            nu_t = self.les.eddy_viscosity
        else:
            nu_t = 0.

        dissipation = stress_term(u_mid, v_u, nu, nu_t=nu_t, laplacian=self.use_laplacian)
        F += dissipation

        # add in continuity term
        F += continuity_term(self.H, h_mid, u_mid, v_h,
                             self.integrate_continuity_by_parts)

        J = fe.derivative(F, du)
        return F, J

    def setup_bcs(self, F):
        if self.simulation == "mms":
            self.u_exact = fe.Expression(
                ("cos(x[0]) * sin(x[1])", "sin(pow(x[0], 2)) + cos(x[1])"),
                degree=4)
            self.h_exact = fe.Expression("sin(x[0]) * sin(x[1])", degree=4)

            def boundary(x, on_boundary):
                return on_boundary

            bc_u = fe.DirichletBC(self.W.sub(0), self.u_exact, boundary)
            bc_h = fe.DirichletBC(self.W.sub(1), self.h_exact, boundary)
            bcs = [bc_u, bc_h]
        elif self.simulation in ["cylinder", "laminar"]:
            # basic BC's
            # set inflow, outflow, walls all via Dirichlet BC's
            u_in = fe.Constant((0.535, 0.))
            u_out = u_in
            no_slip = fe.Constant((0., 0.))

            inflow = "near(x[0], 0)"
            outflow = "near(x[0], 1)"
            walls = "near(x[1], 0) || near(x[1], 0.56)"

            bcu_inflow = fe.DirichletBC(self.W.sub(0), u_in, inflow)
            bcu_outflow = fe.DirichletBC(self.W.sub(0), u_out, outflow)
            bcu_walls = fe.DirichletBC(self.W.sub(0), no_slip, walls)
            bcs = [bcu_inflow, bcu_outflow, bcu_walls]

            # need to include surface integrals if we integrate by parts
            # only left and right boundaries matter, as the rest are zero (no-slip condition)
            if self.integrate_continuity_by_parts:
                class LeftBoundary(fe.SubDomain):
                    def inside(self, x, on_boundary):
                        tol = 1E-14  # tolerance for coordinate comparisons
                        return on_boundary and abs(x[0] - 0.) < tol

                class RightBoundary(fe.SubDomain):
                    def inside(self, x, on_boundary):
                        tol = 1E-14  # tolerance for coordinate comparisons
                        return on_boundary and abs(x[0] - 1.) < tol

                Gamma_left = LeftBoundary()
                Gamma_left.mark(self.boundaries, 1)  # mark with tag 1 for LHS
                Gamma_right = RightBoundary()
                Gamma_right.mark(self.boundaries, 2)  # mark with tag 2 for RHS

                v_u, v_h = fe.TestFunctions(self.W)
                u_prev, h_prev = fe.split(self.du_prev)

                n = fe.FacetNormal(self.mesh)
                ds = fe.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
                F += (
                    (self.H + h_prev) * fe.inner(u_prev, n) * v_h * ds(1)  # LHS set via Dirichlet's
                    + (self.H + h_prev) * fe.inner(u_prev, n) * v_h * ds(2)  # RHS also set via Dirichlet's
                    + 0.  # all other conditions have no-normal/zero flow
                )

            # TODO: take in cylinder mesh parameterisations as an argument/option
            # 0.925 is the centre of the domain
            if self.simulation == "cylinder":
                cylinder = "on_boundary && x[0] >= 0.18 && x[0] <= 0.22 && x[1] >= 0.26 && x[1] <= 0.3"
                bcs.append(
                    fe.DirichletBC(self.W.sub(0), no_slip, cylinder))

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
        prm["snes_solver"]["line_search"] = "bt"
        prm["snes_solver"]["linear_solver"] = "gmres"
        # prm["snes_solver"]["linear_solver"] = "mumps"
        prm["snes_solver"]["preconditioner"] = "jacobi"
        logger.info(f"using {prm['snes_solver']['linear_solver']} solver with {prm['snes_solver']['preconditioner']} PC")

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

    def solve(self, bdf2):
        # set the LES parameter
        if self.use_les:
            self.les.solve()

        # solve at current time
        self.solver.solve()

        # set previous timesteps appropriately
        if bdf2:
            fe.assign(self.du_prev_prev, self.du_prev)

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
        self.checkpoint = fe.HDF5File(self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)

    def checkpoint_close(self):
        self.checkpoint.close()
