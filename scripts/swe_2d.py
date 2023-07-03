""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe

from scipy.linalg import cholesky, cho_factor, cho_solve
from statfenics.covariance import (sq_exp_covariance,
                                   sq_exp_evd_hilbert,
                                   sq_exp_evd)
from statfenics.utils import dolfin_to_csr
from swe_les import LES

# initialise the logger
logger = logging.getLogger(__name__)

# use default Fenics MPI comm (itself uses mpi4py)
comm = fe.MPI.comm_world
rank = comm.Get_rank()


def stress_term(u, v, nu, nu_t, laplacian=False):
    if laplacian:
        return nu * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    else:
        return ((nu + nu_t) * fe.inner(fe.grad(u) + fe.grad(u).T, fe.grad(v)) * fe.dx
                - (nu + nu_t) * (2. / 3.) * fe.inner(fe.div(u) * fe.Identity(2), fe.grad(v)) * fe.dx)


class ShallowTwo:
    def __init__(self, mesh, params, control):
        self.dt = control["dt"]
        self.simulation = control["simulation"]
        self.use_imex = control["use_imex"]
        self.use_les = control["use_les"]

        # use theta-method or IMEX (CNLF)
        if not self.use_imex:
            self.theta = control["theta"]

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
        self.dx = self.mesh.hmin()
        self.dx_max = self.mesh.hmax()
        self.x = fe.SpatialCoordinate(self.mesh)
        self.x_coords = self.mesh.coordinates()
        logger.info("dx(max) = %.5f, dx(min): %.5f", self.dx_max, self.dx)

        assert np.all(self.x_coords >= 0.)
        self.L = np.amax(self.x_coords[:, 0])
        self.B = np.amax(self.x_coords[:, 1])

        self.boundaries = fe.MeshFunction("size_t", self.mesh,
                                          self.mesh.topology().dim() - 1, 0)

        # use P2-P1 elements only
        U = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        H = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        TH = fe.MixedElement([U, H])
        W = self.W = fe.FunctionSpace(self.mesh, TH)

        # get dof labels for each
        self.u_dofs = self.W.sub(0).dofmap().dofs()
        self.h_dofs = self.W.sub(1).dofmap().dofs()

        # split up function spaces for later interpolation
        self.U, self.H = W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        self.du = fe.Function(W)
        self.du_prev = fe.Function(W)
        self.du_prev_prev = fe.Function(W)

        # storage for later
        self.du_vertices = np.copy(self.du.compute_vertex_values())

        # set parameters etc
        self.nu = params["nu"]
        self.C = params["C"]
        self.H = params["H"]

        # set inflow conditions
        self.u_inflow = params["u_inflow"]
        self.inflow_period = params["inflow_period"]
        logger.info("CFL number is %f", self.u_inflow * self.dt / self.dx)

        self.inlet_velocity = fe.Expression(
            (f"{self.u_inflow} * sin(2 * pi * t / {self.inflow_period})", "0.0"),
            pi=np.pi, t=0., degree=4)
        self.inlet_velocity.t = 0.

        # initialise solution objects
        self.F = None
        self.J = None
        self.bcs = []
        self.solver = None

    def setup_form(self):
        g = fe.Constant(9.8)
        nu = fe.Constant(self.nu)
        dt = fe.Constant(self.dt)
        C = fe.Constant(self.C)

        u_prev, h_prev = fe.split(self.du_prev)
        u_prev_prev, h_prev_prev = fe.split(self.du_prev_prev)
        v_u, v_h = fe.TestFunctions(self.W)  # test fcn's

        # weak form: continuity is integrated by parts
        if self.use_imex:
            u, h = fe.TrialFunctions(self.W)
            u_theta = 9/16 * u + 3/8 * u_prev + 1/16 * u_prev_prev
            h_theta = 9/16 * h + 3/8 * h_prev + 1/16 * h_prev_prev
            # u_theta = 1/2 * u + 1/2 * u_prev
            # h_theta = 1/2 * h + 1/2 * h_prev
            u_mag = fe.sqrt(fe.inner(u_prev, u_prev))
            self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx  # mass term u
                      + g * fe.inner(fe.grad(h_theta), v_u) * fe.dx  # surface term
                      + (3/2 * fe.inner(fe.dot(u_prev, fe.nabla_grad(u_prev)), v_u) * fe.dx
                         - 1/2 * fe.inner(fe.dot(u_prev_prev, fe.nabla_grad(u_prev_prev)), v_u) * fe.dx)
                      + (3/2 * C * (u_mag / (self.H + h_prev + 1e-14)) * fe.inner(u_prev, v_u) * fe.dx
                         - 1/2 * C * (u_mag / (self.H + h_prev_prev + 1e-14)) * fe.inner(u_prev_prev, v_u) * fe.dx)
                      + fe.inner(h - h_prev, v_h) / dt * fe.dx  # mass term h
                      - (3/2 * fe.inner((self.H + h_prev) * u_prev, fe.grad(v_h)) * fe.dx
                         - 1/2 * fe.inner((self.H + h_prev_prev) * u_prev_prev, fe.grad(v_h)) * fe.dx))  # bottom friction
        else:
            u, h = fe.split(self.du)
            u_theta = self.theta * u + (1 - self.theta) * u_prev
            h_theta = self.theta * h + (1 - self.theta) * h_prev
            u_mag = fe.sqrt(fe.inner(u_prev, u_prev))
            self.F = (fe.inner(u - u_prev, v_u) / dt * fe.dx  # mass term u
                      + g * fe.inner(fe.grad(h_theta), v_u) * fe.dx  # surface term
                      + fe.inner(fe.dot(u_theta, fe.nabla_grad(u_theta)), v_u) * fe.dx
                      + C * (u_mag / (self.H + h_theta)) * fe.inner(u_theta, v_u) * fe.dx
                      + fe.inner(h - h_prev, v_h) / dt * fe.dx  # mass term h
                      - fe.inner((self.H + h_theta) * u_theta, fe.grad(v_h)) * fe.dx)  # bottom friction

        # laplacian/LES is used for dissipative effects
        # add in (parameterised) dissipation effects
        if self.use_les:
            self.les = LES(
                mesh=self.mesh,
                fs=self.H_space,
                u=u_prev,
                density=1.0,
                smagorinsky_coefficient=0.164)
            nu_t = self.les.eddy_viscosity
            dissipation = stress_term(
                u_theta, v_u, nu, nu_t=nu_t, laplacian=False)
        else:
            nu_t = 0.
            dissipation = stress_term(
                u_theta, v_u, nu, nu_t=nu_t, laplacian=True)

        self.F += dissipation

        if self.simulation == "mms":
            self.u_exact = fe.Expression(
                ("cos(x[0]) * sin(x[1])", "sin(pow(x[0], 2)) + cos(x[1])"),
                degree=4)
            self.h_exact = fe.Expression("sin(x[0]) * sin(x[1])", degree=4)

            def boundary(x, on_boundary):
                return on_boundary

            self.bcs.append(fe.DirichletBC(self.W.sub(0), self.u_exact, boundary))
            self.bcs.append(fe.DirichletBC(self.W.sub(1), self.h_exact, boundary))
        elif self.simulation in ["cylinder", "laminar"]:
            # basic BC's
            # inflow: fixed, u:= u_in
            # outflow: flather, u := u_in + sqrt(g / H) * h
            # walls: no-normal flow, dot(u, n) = 0
            # obstacle: no-slip, u:= (0, 0)
            # set inflow via Dirichlet BC's
            no_slip = fe.Constant((0., 0.))

            inflow = "near(x[0], 0)"
            bcu_inflow = fe.DirichletBC(self.W.sub(0), self.inlet_velocity, inflow)
            self.bcs.append(bcu_inflow)

            # TODO: option for cylinder mesh
            if self.simulation == "cylinder":
                cylinder = ("on_boundary && x[0] >= 0.95 && x[0] <= 1.05 "
                            + "&& x[1] >= 0.45 && x[1] <= 0.55")
                self.bcs.append(fe.DirichletBC(self.W.sub(0), no_slip, cylinder))

            # need to include surface integrals as we integrate by parts
            # only left and right boundaries matter;
            # the rest are zero (no-slip OR no-normal flow)
            class LeftBoundary(fe.SubDomain):
                def inside(self, x, on_boundary):
                    tol = 1e-14
                    return on_boundary and abs(x[0] - 0.) < tol

            class RightBoundary(fe.SubDomain):
                def inside(self, x, on_boundary):
                    tol = 1e-14
                    return on_boundary and abs(x[0] - 2.) < tol

            gamma_left = LeftBoundary()
            gamma_left.mark(self.boundaries, 1)  # mark with tag 1 for LHS
            gamma_right = RightBoundary()
            gamma_right.mark(self.boundaries, 2)  # mark with tag 2 for RHS

            # all other conditions are just no-normal/no-slip
            n = fe.FacetNormal(self.mesh)
            ds = fe.Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
            if self.use_imex:
                self.F += ((3/2 * (self.H + h_prev) * fe.inner(u_prev, n) * v_h * ds(1)
                            - 1/2 * (self.H + h_prev_prev) * fe.inner(u_prev_prev, n) * v_h * ds(1))  # inflow
                           + (self.H + h_theta) * fe.inner(self.inlet_velocity, n) * v_h * ds(2)  # flather
                           + (3/2 * (self.H + h_prev) * fe.sqrt(g / self.H) * h_prev * v_h * ds(2)
                              - 1/2 * (self.H + h_prev_prev) * fe.sqrt(g / self.H) * h_prev_prev * v_h * ds(2)))  # outflow
                self.J = None
            else:
                self.F += ((self.H + h_theta) * fe.inner(u_theta, n) * v_h * ds(1)  # inflow
                           + (self.H + h_theta) * fe.inner(self.inlet_velocity, n) * v_h * ds(2)  # flather
                           + (self.H + h_theta) * fe.sqrt(g / self.H) * h_theta * v_h * ds(2))
                self.J = fe.derivative(self.F, self.du, fe.TrialFunction(self.W))

    def setup_solver(self, use_ksp=False):
        if self.use_imex:
            self.fem_lhs, self.fem_rhs = fe.lhs(self.F), fe.rhs(self.F)
            self.A = fe.assemble(self.fem_lhs)
            for bc in self.bcs:
                bc.apply(self.A)

            # setup solver and set operators
            self.solver = fe.PETScKrylovSolver()
            self.solver.set_operator(self.A)
            self.solver.set_reuse_preconditioner(True)

            if use_ksp:
                fe.PETScOptions.set("ksp_type", "gmres")
                fe.PETScOptions.set("pc_type", "jacobi")
            else:
                fe.PETScOptions.set("ksp_type", "preonly")
                fe.PETScOptions.set("pc_type", "lu")

            fe.PETScOptions.set("ksp_gmres_restart", 100)
            fe.PETScOptions.set("ksp_atol", 1e-6)
            fe.PETScOptions.set("ksp_rtol", 1e-6)
            fe.PETScOptions.set("ksp_view")
        else:
            problem = fe.NonlinearVariationalProblem(
                self.F, self.du, bcs=self.bcs, J=self.J)
            self.solver = fe.NonlinearVariationalSolver(problem)

            # use PETSc SNES for all solvers
            prm = self.solver.parameters
            prm["nonlinear_solver"] = "snes"
            prm["snes_solver"]["line_search"] = "bt"

            if use_ksp:
                # PETSc SNES linear solver: gmres + jacobi
                prm["snes_solver"]["linear_solver"] = "gmres"
                prm["snes_solver"]["preconditioner"] = "jacobi"

                # set PETSc options
                fe.PETScOptions.set("ksp_gmres_restart", 100)
            else:
                # PETSc SNES linear solver: MUMPS
                prm["snes_solver"]["linear_solver"] = "mumps"

            logger.info(f"using {prm['snes_solver']['linear_solver']} solver "
                        + f"with {prm['snes_solver']['preconditioner']} PC")

            # SNES configs
            prm["snes_solver"]["report"] = True
            prm["snes_solver"]["relative_tolerance"] = 1e-5
            prm["snes_solver"]['absolute_tolerance'] = 1e-5
            prm["snes_solver"]["maximum_iterations"] = 10_000
            prm["snes_solver"]['error_on_nonconvergence'] = True
            prm["snes_solver"]['krylov_solver']['nonzero_initial_guess'] = True

            # solver reporting
            prm["snes_solver"]['krylov_solver']['relative_tolerance'] = 1e-6
            prm["snes_solver"]['krylov_solver']['absolute_tolerance'] = 1e-6
            prm["snes_solver"]['krylov_solver']['report'] = True
            prm["snes_solver"]['krylov_solver']['monitor_convergence'] = False

    def solve(self):
        if self.use_les:
            self.les.solve()

        if self.use_imex:
            b = fe.assemble(self.fem_rhs)
            for bc in self.bcs:
                bc.apply(b)

            self.solver.solve(self.du.vector(), b)
            fe.assign(self.du_prev_prev, self.du_prev)
        else:
            # solve at current time
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
        logger.info(f"storing outputs in {checkpoint_file}")
        self.checkpoint = fe.HDF5File(self.mesh.mpi_comm(), checkpoint_file, "w")

    def checkpoint_save(self, t):
        """ Save the simulation at the current time. """
        self.checkpoint.write(self.du, "/du", t)

    def checkpoint_close(self):
        self.checkpoint.close()


class ShallowTwoFilter(ShallowTwo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_filter(self, stat_params):
        u, v = fe.TrialFunction(self.W), fe.TestFunction(self.W)
        M = fe.assemble(fe.inner(u, v) * fe.dx)
        M_scipy = dolfin_to_csr(M)

        self.mean = np.copy(self.du.vector().get_local())
        self.k_init_u = stat_params["k_init_u"]
        self.k_init_h = stat_params["k_init_h"]
        self.k = stat_params["k"]

        # matrix inits
        self.cov_sqrt = np.zeros((self.mean.shape[0], self.k))
        self.cov_sqrt_prev = np.zeros((self.mean.shape[0], self.k))
        self.cov_sqrt_pred = np.zeros((self.mean.shape[0],
                                       self.k + self.k_init_u + self.k_init_h))
        self.G_sqrt = np.zeros((self.mean.shape[0],
                                self.k_init_u + self.k_init_h))

        if stat_params["rho_u"] > 0.:
            self.Ku_vals, Ku_vecs = sq_exp_evd_hilbert(
                self.U_space, self.k_init_u,
                stat_params["rho_u"],
                stat_params["ell_u"])

            self.G_sqrt[self.u_dofs, 0:len(self.Ku_vals)] = (
                Ku_vecs @ np.diag(np.sqrt(self.Ku_vals)))
            print(f"Spectral diff (u): {self.Ku_vals[-1]:.4e}, {self.Ku_vals[0]:.4e}")

        if stat_params["rho_h"] > 0.:
            self.Kh_vals, Kh_vecs = sq_exp_evd_hilbert(
                self.H_space, self.k_init_h,
                stat_params["rho_h"],
                stat_params["ell_h"])
            self.G_sqrt[self.h_dofs, self.k_init_u:(self.k_init_u + len(self.Kh_vals))] = (
                Kh_vecs @ np.diag(np.sqrt(self.Kh_vals)))
            print(f"Spectral diff (h): {self.Kh_vals[-1]:.4e}, {self.Kh_vals[0]:.4e}")

        # multiplication *after* the initial construction
        self.G_sqrt[:] = M_scipy @ self.G_sqrt

    def prediction_step(self, t):
        raise NotImplementedError

    def compute_lml(self, y, H, sigma_y):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean
        n_obs = len(mean_obs)

        HL = H @ self.cov_sqrt
        cov_obs = HL @ HL.T
        cov_obs[np.diag_indices_from(cov_obs)] += sigma_y**2 + 1e-10
        S_chol = cho_factor(cov_obs, lower=True)
        S_inv_y = cho_solve(S_chol, y - mean_obs)
        log_det = 2 * np.sum(np.log(np.diag(S_chol[0])))

        return (- S_inv_y @ S_inv_y / 2
                - log_det / 2
                - n_obs * np.log(2 * np.pi) / 2)

    def update_step(self, y, H, sigma_y, return_correction=False):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean

        HL = H @ self.cov_sqrt
        cov_obs = HL @ HL.T
        cov_obs[np.diag_indices_from(cov_obs)] += sigma_y**2 + 1e-10
        S_chol = cho_factor(cov_obs, lower=True)
        S_inv_y = cho_solve(S_chol, y - mean_obs)

        # kalman updates: for high-dimensions this is the bottleneck
        # TODO(connor): avoid re-allocation and poor memory management
        HL = H @ self.cov_sqrt
        S_inv_HL = cho_solve(S_chol, HL)

        correction = self.cov_sqrt @ HL.T @ S_inv_y
        self.mean += correction
        R = cholesky(np.eye(HL.shape[1]) - HL.T @ S_inv_HL, lower=True)
        self.cov_sqrt[:] = self.cov_sqrt @ R

        # update fenics state vector
        self.du.vector().set_local(self.mean.copy())

        if return_correction:
            return correction

    def set_prev(self):
        fe.assign(self.du_prev, self.du)
        if self.lr:
            self.cov_sqrt_prev[:] = self.cov_sqrt
        else:
            self.cov_prev[:] = self.cov
