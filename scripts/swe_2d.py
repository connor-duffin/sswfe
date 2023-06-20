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


class ShallowTwo:
    def __init__(self, mesh, control):
        self.dt = control["dt"]
        self.theta = control["theta"]
        self.simulation = control["simulation"]

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
            self.nu = 1e-4
            self.C = 0.
            self.H = 0.16

        # initialise solution objects
        self.F = None
        self.J = None
        self.bcs = None
        self.solver = None

    # TODO(connor): delete the BDF2 option + clean up bottom friction
    def setup_form(self, du, du_prev, du_prev_prev=None, bdf2=False):
        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        dt = fe.Constant(self.dt)

        u, h = fe.split(du)
        u_prev, h_prev = fe.split(du_prev)

        # C = fe.Constant(self.C)
        # u_mag = fe.sqrt(fe.inner(u_prev, u_prev))

        if du_prev_prev is not None or bdf2:
            logger.warning("BDF2 option is DEPRECATED and will be removed soon")
            raise ValueError

        v_u, v_h = fe.TestFunctions(self.W)
        self.f_u = fe.Function(self.U_space)
        self.f_h = fe.Function(self.H_space)

        # weak form for the system:
        # continuity is integrated by parts;
        # laplacian is used for dissipative effects (no LES)
        u_mid = self.theta * u + (1 - self.theta) * u_prev
        h_mid = self.theta * h + (1 - self.theta) * h_prev
        F = (fe.inner(u - u_prev, v_u) / dt * fe.dx  # mass term u
             + fe.inner(fe.dot(u_mid, fe.nabla_grad(u_mid)), v_u) * fe.dx  # advection
             + nu * fe.inner(fe.grad(u_mid), fe.grad(v_u)) * fe.dx  # dissipation
             # + C * u_mag * fe.inner(u_mid, v_u) / (self.H + h_prev) * fe.dx  # bottom friction term
             + fe.inner(h - h_prev, v_h) / dt * fe.dx  # mass term h
             + g * fe.inner(fe.grad(h_mid), v_u) * fe.dx  # surface term
             - fe.inner((self.H + h_mid) * u_mid, fe.grad(v_h)) * fe.dx
             - fe.inner(self.f_u, v_u) * fe.dx
             - fe.inner(self.f_h, v_h) * fe.dx)
        J = fe.derivative(F, du)

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
            # inflow: fixed, u:= (0.535, 0)
            # outflow: flather, u := u_in + sqrt(g / H) * h
            # walls: no-normal flow, dot(u, n) = 0
            # obstacle: no-slip, u:= (0, 0)
            # set inflow via Dirichlet BC's
            u_in = fe.Constant((0.535, 0.))
            no_slip = fe.Constant((0., 0.))

            inflow = "near(x[0], 0)"
            bcu_inflow = fe.DirichletBC(self.W.sub(0), u_in, inflow)
            bcs = [bcu_inflow]

            # TODO: option for cylinder mesh
            if self.simulation == "cylinder":
                cylinder = "on_boundary && x[0] >= 0.18 && x[0] <= 0.22 && x[1] >= 0.26 && x[1] <= 0.3"
                bcs.append(fe.DirichletBC(self.W.sub(0), no_slip, cylinder))

            # need to include surface integrals as we integrate by parts
            # only left and right boundaries matter;
            # the rest are zero (no-slip OR no-normal flow)
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

            n = fe.FacetNormal(self.mesh)
            ds = fe.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
            F += ((self.H + h_prev) * fe.inner(u_in, n) * v_h * ds(1)  # inflow
                  + (self.H + h_prev) * fe.inner(u_in, n) * v_h * ds(2)  # flather
                  + (self.H + h_prev) * fe.sqrt(9.8 / self.H) * h_mid * v_h * ds(2)  # flather
                  + 0.)  # all other conditions have no-normal/zero flow

        return F, J, bcs

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
        prm["snes_solver"]["preconditioner"] = "jacobi"
        # prm["snes_solver"]["preconditioner"] = "fieldsplit"

        # prm["snes_solver"]["linear_solver"] = "mumps"
        # prm["snes_solver"]["krylov_solver"]["pc_fieldsplit_type"] = "schur"
        # prm["snes_solver"]["krylov_solver"]["pc_fieldsplit_schur_fact_type"] = "FULL"
        # prm["snes_solver"]["krylov_solver"]["fieldsplit_0_ksp_type"] = "preonly"
        # prm["snes_solver"]["krylov_solver"]["fieldsplit_1_ksp_type"] = "preonly"
        # prm["snes_solver"]["krylov_solver"]["fieldsplit_0_pc_type"] = "ilu"
        # prm["snes_solver"]["krylov_solver"]["fieldsplit_1_pc_type"] = "ilu"

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

    def solve(self, bdf2=False):
        # solve at current time
        self.solver.solve()

        # set previous timesteps appropriately
        if bdf2:
            logger.warning("BDF2 option is DEPRECATED and will be removed soon")
            raise ValueError

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
