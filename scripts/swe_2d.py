""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe

from scipy.linalg import cholesky, cho_factor, cho_solve, eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from statfenics.utils import dolfin_to_csr
from statfenics.covariance import sq_exp_spectral_density
from swe_les import LES

# for parallelisation
from petsc4py import PETSc
from slepc4py import SLEPc

# initialise the logger
logger = logging.getLogger(__name__)

# use default Fenics MPI comm (itself uses mpi4py)
comm = fe.MPI.comm_world
rank = comm.Get_rank()


def boundary(x, on_boundary):
    return on_boundary


def laplacian_evd(V, k=64, bc="Dirichlet"):
    """
    Aproximate the smallest k eigenvalues/eigenfunctions of the Laplacian.

    I.e. solve u_xx + u_yy + u_zz + ... = - lambda u,
    using shift-invert mode in SLEPc for scalable computations.

    Parameters
    ----------
    V : fenics.FunctionSpace
        FunctionSpace on which to compute the approximation.
    k : int, optional
        Number of modes to take in the approximation.
    bc : str, optional
        Boundary conditions to use in the approximation. Either 'Dirichlet' or
        'Neumann'.
    """
    e = V.element()
    dim = e.num_sub_elements()

    def boundary(x, on_boundary):
        return on_boundary

    bc_types = ["Dirichlet", "Neumann"]
    if bc not in bc_types:
        raise ValueError("Invalid bc, expected one of {bc_types}")
    elif bc == "Dirichlet":
        if dim == 0:
            bc = fe.DirichletBC(V, fe.Constant(0), boundary)
        elif dim == 2:
            bc = fe.DirichletBC(V, fe.Constant((0, 0, 0)), boundary)
        else:
            raise NotImplementedError
    else:
        bc = None

    # define variational problem
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    a = fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    A = fe.PETScMatrix()
    fe.assemble(a, tensor=A)

    M = fe.PETScMatrix()
    M_no_bc = fe.PETScMatrix()
    fe.assemble(fe.inner(u, v) * fe.dx, tensor=M)
    fe.assemble(fe.inner(u, v) * fe.dx, tensor=M_no_bc)

    if bc is not None:
        # sets BC rows of A to identity
        bc.apply(A)

        # sets rows of M to zeros
        bc.apply(M)
        bc.zero(M)

    M = M.mat()
    M_no_bc = M_no_bc.mat()
    A = A.mat()

    # solver inspired by: cmaurini
    # https://gist.github.com/cmaurini/6dea21fc01c6a07caeb96ff9c86dc81e
    E = SLEPc.EPS()
    E.create()
    E.setOperators(A, M)
    E.setDimensions(nev=k)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(0)
    E.setTolerances(1e-12, 100_000)
    S = E.getST()
    S.setType("sinvert")
    E.setFromOptions()
    E.solve()

    # check that things have converged
    logger.info("Eigenvalues converged: %d", E.getConverged())

    # and set up objects for storage
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    laplace_eigenvals = np.zeros((k, ))
    eigenvecs = np.zeros((vr.array_r.shape[0], k))
    errors = np.zeros((k, ))

    for i in range(k):
        laplace_eigenvals[i] = np.real(E.getEigenpair(i, vr, vi))
        eigenvecs[:, i] = vr.array_r
        errors[i] = E.computeError(i)

    M_scipy = csr_matrix(M_no_bc.getValuesCSR()[::-1], shape=M.size)
    for i in range(k):
        eigenvecs_scale = np.sqrt(eigenvecs[:, i].T @ M_scipy @ eigenvecs[:, i])
        eigenvecs[:, i] /= eigenvecs_scale

    return (laplace_eigenvals, eigenvecs)


def sq_exp_evd_hilbert(V, k=64, scale=1., ell=1., bc="Dirichlet"):
    """
    Approximate the SqExp covariance using Hilbert-GP.

    For full details:
    Solin, A., Särkkä, S., 2020. Hilbert space methods for reduced-rank
    Gaussian process regression. Stat Comput 30, 419–446.
    https://doi.org/10.1007/s11222-019-09886-w

    Parameters
    ----------
    V : fenics.FunctionSpace
        FunctionSpace on which to compute the approximation.
    k : int, optional
        Number of modes to take in the approximation.
    scale : float
        Variance hyperparameter.
    ell : float
        Length-scale hyperparameter.
    bc : str, optional
        Boundary conditions to use in the approximation. Either 'Dirichlet' or
        'Neumann'.
    """
    laplace_eigenvals, eigenvecs = laplacian_evd(V, k=k, bc=bc)

    # enforce positivity --- picks up eigenvalues that are negative
    laplace_eigenvals = laplace_eigenvals
    logger.info("Laplacian eigenvalues: %s", laplace_eigenvals)
    eigenvals = sq_exp_spectral_density(
        np.sqrt(laplace_eigenvals),
        scale=scale,
        ell=ell,
        D=V.mesh().geometric_dimension())
    return (eigenvals, eigenvecs)


def stress_term(u, v, nu, nu_t, laplacian=False):
    if laplacian:
        return nu * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
    else:
        return ((nu + nu_t) * fe.inner(fe.grad(u) + fe.grad(u).T, fe.grad(v)) * fe.dx
                - (nu + nu_t) * (2. / 3.) * fe.inner(fe.div(u) * fe.Identity(2), fe.grad(v)) * fe.dx)


class ShallowTwo:
    def __init__(self, mesh, params, control, comm=None):
        self.dt = control["dt"]
        self.simulation = control["simulation"]
        self.use_imex = control["use_imex"]
        self.use_les = control["use_les"]

        # register MPI communicator
        if comm is not None:
            self.comm = comm

        # use theta-method or IMEX (CNLF)
        if not self.use_imex:
            self.theta = control["theta"]

        if type(mesh) == str:
            # read mesh from file
            logger.info("reading mesh from file")
            if self.comm is None:
                self.mesh = fe.Mesh()
            else:
                self.mesh = fe.Mesh(self.comm)

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
        self.u_dofs = self.W.sub(0).sub(0).dofmap().dofs()
        self.v_dofs = self.W.sub(0).sub(1).dofmap().dofs()
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
            u_mag = fe.sqrt(fe.inner(u_prev, u_prev) + 1e-12)  # for numerical stability
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
            self.J_prev = fe.derivative(self.F, self.du_prev, fe.TrialFunction(self.W))

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
            prm["snes_solver"]["report"] = False
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
        else:
            # solve at current time
            self.solver.solve()

    def set_prev(self):
        if self.use_imex:
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


class ShallowTwoFilter(ShallowTwo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_filter(self, stat_params):
        if self.use_imex:
            logger.error("Not setup for IMEX yet, bailing out")
            raise NotImplementedError

        u, v = fe.TrialFunction(self.W), fe.TestFunction(self.W)
        M = fe.assemble(fe.inner(u, v) * fe.dx)
        self.M_scipy = dolfin_to_csr(M)

        self.mean = np.copy(self.du.vector().get_local())
        self.k_init_u = stat_params["k_init_u"]
        self.k_init_v = stat_params["k_init_v"]
        self.k_init_h = stat_params["k_init_h"]
        self.k = stat_params["k"]

        # matrix inits
        self.cov_sqrt = np.zeros((self.mean.shape[0], self.k))
        self.cov_sqrt_prev = np.zeros((self.mean.shape[0], self.k))
        self.cov_sqrt_pred = np.zeros((self.mean.shape[0],
                                       self.k + self.k_init_u + self.k_init_v + self.k_init_h))
        self.K_sqrt = np.zeros((self.mean.shape[0],
                                self.k_init_u + self.k_init_v + self.k_init_h))
        self.G_sqrt = np.zeros((self.mean.shape[0],
                                self.k_init_u + self.k_init_v + self.k_init_h))

        # setup spaces in which we posit things accordingly
        U, V = self.U.split()
        U_space, V_space = U.collapse(), V.collapse()

        if stat_params["rho_u"] > 0.:
            self.Ku_vals, Ku_vecs = sq_exp_evd_hilbert(
                U_space,
                self.k_init_u,
                stat_params["rho_u"],
                stat_params["ell_u"])

            for i in range(self.k_init_u):
                self.K_sqrt[self.u_dofs, i] = Ku_vecs[:, i] * np.sqrt(self.Ku_vals[i])

            logger.info(f"Spectral diff (u): {self.Ku_vals[-1]:.4e}, {self.Ku_vals[0]:.4e}")

        if stat_params["rho_v"] > 0.:
            self.Kv_vals, Kv_vecs = sq_exp_evd_hilbert(
                V_space,
                self.k_init_v,
                stat_params["rho_v"],
                stat_params["ell_v"])

            self.K_sqrt[self.v_dofs,
                        self.k_init_u:(self.k_init_u + len(self.Kv_vals))] = (
                Kv_vecs @ np.diag(np.sqrt(self.Kv_vals)))
            logger.info(f"Spectral diff (v): {self.Kv_vals[-1]:.4e}, {self.Kv_vals[0]:.4e}")

        if stat_params["rho_h"] > 0.:
            self.Kh_vals, Kh_vecs = sq_exp_evd_hilbert(
                self.H_space,
                self.k_init_h,
                stat_params["rho_h"],
                stat_params["ell_h"])
            self.K_sqrt[self.h_dofs,
                        (self.k_init_u + self.k_init_v):(self.k_init_u + self.k_init_v + len(self.Kh_vals))] = (
                Kh_vecs @ np.diag(np.sqrt(self.Kh_vals)))
            logger.info(f"Spectral diff (h): {self.Kh_vals[-1]:.4e}, {self.Kh_vals[0]:.4e}")

        # multiplication *after* the initial construction
        self.G_sqrt[:] = self.M_scipy @ self.G_sqrt

        try:
            self.H = stat_params["H"]
            self.sigma_y = stat_params["sigma_y"]

            self.n_obs = self.H.shape[0]

            self.mean_obs = np.zeros((self.n_obs, ))
            self.HL = np.zeros((self.n_obs, self.k))

            self.S_inv_HL = np.zeros((self.n_obs, self.k))
            self.S_inv_y = np.zeros((self.n_obs, ))

            self.cov_obs = np.zeros((self.n_obs, self.n_obs))
            self.R = np.zeros((self.k, self.k))
        except KeyError:
            logger.warn(
                "Obs. operator and noise not parsed: setup for prior run ONLY")

        # tangent linear models
        self.J_mat = fe.assemble(self.J)
        self.J_prev_mat = fe.assemble(self.J_prev)

        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

        # initialise LU factorisation (from scratch) of tangent linear
        self.J_scipy_lu = splu(self.J_scipy.tocsc())

    def assemble_derivatives(self):
        self.J_mat = fe.assemble(self.J)
        self.J_prev_mat = fe.assemble(self.J_prev)

        # set things up appropriately
        for J in [self.J_mat, self.J_prev_mat]:
            for bc in self.bcs: bc.apply(J)

        # TODO(connor): re-use sparsity pattern and speed-up
        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

    def prediction_step(self, t):
        self.solve()
        self.mean[:] = self.du.vector().get_local()

        # TODO(connor): reuse sparsity patterns?
        self.assemble_derivatives()
        self.J_scipy_lu = splu(self.J_scipy.tocsc())  # options=dict(Fact="SamePattern")

        self.cov_sqrt_pred[:, :self.k] = self.J_prev_scipy @ self.cov_sqrt_prev
        self.cov_sqrt_pred[:, self.k:] = np.sqrt(self.dt) * self.G_sqrt
        self.cov_sqrt_pred[:] = self.J_scipy_lu.solve(self.cov_sqrt_pred)

        # perform reduction
        # TODO(connor) avoid reallocation
        D, V = eigh(self.cov_sqrt_pred.T @ self.cov_sqrt_pred)
        D, V = D[::-1], V[:, ::-1]
        logger.debug("Prop. variance kept in the reduction: %f",
                     np.sum(D[0:self.k]) / np.sum(D))
        np.dot(self.cov_sqrt_pred, V[:, 0:self.k], out=self.cov_sqrt)

    def update_step(self, y, compute_lml=False):
        self.mean_obs[:] = self.H @ self.mean
        self.HL[:] = self.H @ self.cov_sqrt
        self.cov_obs[:] = self.HL @ self.HL.T
        self.cov_obs[np.diag_indices_from(self.cov_obs)] += self.sigma_y**2 + 1e-10
        S_chol = cho_factor(
            self.cov_obs, lower=True, overwrite_a=True, check_finite=False)

        self.S_inv_y[:] = cho_solve(S_chol, y - self.mean_obs)
        self.S_inv_HL[:] = cho_solve(S_chol, self.HL)

        self.mean += self.cov_sqrt @ self.HL.T @ self.S_inv_y
        self.R[:] = cholesky(
            np.eye(self.HL.shape[1]) - self.HL.T @ self.S_inv_HL, lower=True)
        self.cov_sqrt[:] = self.cov_sqrt @ self.R

        # update fenics state vector
        self.du.vector().set_local(self.mean)

        if compute_lml:
            log_det = 2 * np.sum(np.log(np.diag(S_chol[0])))
            lml = (- self.S_inv_y @ self.S_inv_y / 2
                   - log_det / 2
                   - self.n_obs * np.log(2 * np.pi) / 2)
            return lml

    def set_prev(self):
        fe.assign(self.du_prev, self.du)
        self.cov_sqrt_prev[:] = self.cov_sqrt


class ShallowTwoFilterPETSc(ShallowTwo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_filter(self, stat_params):
        if self.use_imex:
            logger.error("Not setup for IMEX yet, bailing out")
            raise NotImplementedError

        self.mean = self.du.vector().vec()
        self.k_init_u = stat_params["k_init_u"]
        self.k_init_v = stat_params["k_init_v"]
        self.k_init_h = stat_params["k_init_h"]
        self.k = stat_params["k"]

        # setup stuff for ordering
        self.dofmap = self.W.dofmap()
        self.local_dofs = self.dofmap.dofs()
        lgmap = [int(x) for x in self.dofmap.tabulate_local_to_global_dofs()]
        lgmap = PETSc.LGMap().create(lgmap, comm=self.comm)
        n_dofs = self.dofmap.index_map().size(fe.IndexMap.MapSize.OWNED)
        n_dofs_global = self.dofmap.index_map().size(fe.IndexMap.MapSize.GLOBAL)

        # matrix inits
        self.cov_sqrt = PETSc.Mat()
        self.cov_sqrt.create(comm=self.comm)
        self.cov_sqrt.setSizes([[n_dofs, n_dofs_global], self.k])
        self.cov_sqrt.setType("dense")
        self.cov_sqrt.setUp()
        self.cov_sqrt.setLGMap(
            lgmap,
            PETSc.LGMap().create([int(i) for i in range(self.k)], comm=self.comm))

        self.cov_sqrt_prev = PETSc.Mat()
        self.cov_sqrt_prev.create(comm=self.comm)
        self.cov_sqrt_prev.setSizes([[n_dofs, n_dofs_global], self.k])
        self.cov_sqrt_prev.setType("dense")
        self.cov_sqrt_prev.setUp()
        self.cov_sqrt_prev.setLGMap(
            lgmap,
            PETSc.LGMap().create([int(i) for i in range(self.k)], comm=self.comm))

        self.cov_sqrt_pred = PETSc.Mat()
        self.cov_sqrt_pred.create(comm=self.comm)
        self.cov_sqrt_pred.setSizes(
            [[n_dofs, n_dofs_global],
             self.k + self.k_init_u + self.k_init_v + self.k_init_h])
        self.cov_sqrt_pred.setType("dense")
        self.cov_sqrt_pred.setUp()
        self.cov_sqrt_pred.setLGMap(
            lgmap,
            PETSc.LGMap().create([int(i) for i in range(self.k + self.k_init_u + self.k_init_v + self.k_init_h)], comm=self.comm))

        self.K_sqrt = PETSc.Mat()
        self.K_sqrt.create(comm=self.comm)
        self.K_sqrt.setSizes(
            [[n_dofs, n_dofs_global],
             self.k_init_u + self.k_init_v + self.k_init_h])
        self.K_sqrt.setType("dense")
        self.K_sqrt.setUp()
        self.K_sqrt.setLGMap(
            lgmap,
            PETSc.LGMap().create(
                [int(i) for i in range(self.k_init_u + self.k_init_v + self.k_init_h)],
                comm=self.comm))

        # NOW start eigenvalue computations: do it on the joint space
        du = fe.TrialFunction(self.W)
        vel, h = fe.split(du)
        u1, u2 = fe.split(vel)
        # test functions
        qv = fe.TestFunction(self.W)
        q, v = fe.split(qv)
        q1, q2 = fe.split(q)

        # setup everything
        a = (fe.inner(fe.grad(u1), fe.grad(q1)) * fe.dx
             + fe.inner(fe.grad(u2), fe.grad(q2)) * fe.dx
             + fe.inner(fe.grad(h), fe.grad(v)) * fe.dx)
        A = fe.PETScMatrix()
        fe.assemble(a, tensor=A)

        M = fe.PETScMatrix()
        fe.assemble(fe.inner(du, qv) * fe.dx, tensor=M)

        M_no_bc = fe.PETScMatrix()
        fe.assemble(fe.inner(du, qv) * fe.dx, tensor=M_no_bc)

        # sets rows of M to zeros
        bc = fe.DirichletBC(self.W, fe.Constant((0, 0, 0)), boundary)
        bc.apply(A)
        bc.apply(M)
        bc.zero(M)

        E = SLEPc.EPS(comm=self.comm)
        E.create()
        E.setOperators(A.mat(), M.mat())

        E.setDimensions(nev=3*self.k, ncv=2*3*self.k)
        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
        E.setTarget(0.)
        E.setTolerances(1e-12, 100_000)
        S = E.getST()
        S.setType("sinvert")
        E.setFromOptions()
        E.solve()

        print(f"{E.getConverged()} eigenpairs converged, requested {3 * self.k}")

        # set up vectors for storage
        vr, vi = A.mat().getVecs()
        wr, wi = A.mat().getVecs()
        laplace_eigenvals = np.zeros((self.k, ))
        errors = np.zeros((self.k, ))

        # u-eigenpairs
        for i in range(self.k_init_u):
            vr.zeroEntries()
            wr.zeroEntries()
            eigenvals = E.getEigenpair(3 * i, vr, vi)

            # zero the v-values (due to identifiability, then re-normalize)
            vr.setValues(self.v_dofs, np.zeros_like(self.v_dofs))
            vr.setValues(self.h_dofs, np.zeros_like(self.h_dofs))
            vr.assemble()

            print(i, f"{vr.norm():.5e}")
            if not np.isclose(vr.norm(), 1.):
                vr.normalize()
                vr.assemble()

            # should be positive for laplacian
            assert np.isclose(np.imag(eigenvals), 0.)
            laplace_eigenvals[i] = np.real(eigenvals)

            M_no_bc.mat().mult(vr, wr)
            weighted_norm = np.sqrt(vr.tDot(wr))

            spec_dens = sq_exp_spectral_density(
                np.sqrt(laplace_eigenvals[i]),
                scale=stat_params["rho_u"],
                ell=stat_params["ell_u"],
                D=2
            )

            errors[i] = E.computeError(i)
            local_vals = vr.getValues(self.u_dofs)
            print(i, f"{np.linalg.norm(vr.array_r):.5e}")
            self.K_sqrt.setValues(
                self.u_dofs, i,
                local_vals * np.sqrt(spec_dens) / weighted_norm)

        # v-eigenpairs
        for i in range(self.k_init_v):
            vr.zeroEntries(); wr.zeroEntries()
            laplace_eigenvals[i] = np.abs(np.real(E.getEigenpair(1 + 3 * i, vr, vi)))

            # zero the u-values due to identifiability
            vr.setValues(self.u_dofs, np.zeros_like(self.u_dofs))
            vr.setValues(self.h_dofs, np.zeros_like(self.h_dofs))
            vr.assemble()

            vr.normalize()
            vr.assemble()

            M_no_bc.mat().mult(vr, wr)
            weighted_norm = np.sqrt(vr.tDot(wr))

            errors[i] = E.computeError(i)
            spec_dens = sq_exp_spectral_density(
                np.sqrt(laplace_eigenvals[i]),
                scale=stat_params["rho_v"],
                ell=stat_params["ell_v"],
                D=2
            )
            # local_vals = np.sqrt(spec_dens) * vr.getValues(self.v_dofs)
            local_vals = vr.getValues(self.v_dofs)
            self.K_sqrt.setValues(
                self.v_dofs,
                self.k_init_u + i,
                local_vals * np.sqrt(spec_dens) / weighted_norm)

        # h-eigenpairs
        for i in range(self.k_init_h):
            vr.zeroEntries(); wr.zeroEntries()
            laplace_eigenvals[i] = np.abs(np.real(E.getEigenpair(2 + 3 * i, vr, vi)))

            # zero relevant entries
            vr.setValues(self.u_dofs, np.zeros_like(self.u_dofs))
            vr.setValues(self.v_dofs, np.zeros_like(self.v_dofs))
            vr.assemble()

            M_no_bc.mat().mult(vr, wr)
            weighted_norm = np.sqrt(vr.tDot(wr))

            spec_dens = sq_exp_spectral_density(
                np.sqrt(laplace_eigenvals[i]),
                scale=stat_params["rho_h"],
                ell=stat_params["ell_h"],
                D=2
            )

            local_vals = vr.getValues(self.h_dofs)
            errors[i] = E.computeError(i)
            self.K_sqrt.setValues(
                self.h_dofs, self.k_init_u + self.k_init_v + i,
                local_vals * np.sqrt(spec_dens) / weighted_norm)

        print(f"Spectral diff (laplacian): {laplace_eigenvals[-1]:.4e}, {laplace_eigenvals[0]:.4e}")

        # finally assemble everything
        self.cov_sqrt.assemble()
        self.cov_sqrt_prev.assemble()
        self.cov_sqrt_pred.assemble()
        self.K_sqrt.assemble()

        # multiplication *after* the initial construction
        self.G_sqrt = self.K_sqrt.copy()
        M.mat().matMult(self.K_sqrt, self.G_sqrt)

        # create matrix for values
        self.V = PETSc.Mat().create(comm=self.comm)
        self.V.setSizes(
            [self.k + self.k_init_u + self.k_init_v + self.k_init_h, self.k])
        self.V.setType("mpidense")
        self.V.setUp()
        self.V.assemble()

        # tangent linear models
        self.J_mat = fe.PETScMatrix()
        fe.assemble(self.J, tensor=self.J_mat)

        self.J_prev_mat = fe.PETScMatrix()
        fe.assemble(self.J_prev, tensor=self.J_prev_mat)

    def assemble_derivatives(self):
        fe.assemble(self.J, tensor=self.J_mat)
        fe.assemble(self.J_prev, tensor=self.J_prev_mat)

        # set things up appropriately
        for J in [self.J_mat, self.J_prev_mat]:
            for bc in self.bcs: bc.apply(J)

    def prediction_step(self, t):
        self.solve()
        self.mean.setArray(self.du.vector().get_local())

        # TODO(connor): reuse sparsity patterns?
        self.assemble_derivatives()

        # TODO(connor): this should NOT be created every iteration
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.getPC().setFactorSolverType("mumps")

        # allow command-line customization
        ksp.setFromOptions()
        ksp.setOperators(self.J_mat.mat())

        # and here we create the requisite objects and the like
        vec_extract, vec_pred = self.J_mat.mat().getVecs()
        vec_solve, _ = self.J_mat.mat().getVecs()

        for i in range(self.k + self.k_init_u + self.k_init_v + self.k_init_h):
            vec_extract.zeroEntries()
            vec_pred.zeroEntries()
            vec_solve.zeroEntries()

            if i < self.k:
                self.cov_sqrt_prev.getColumnVector(i, vec_extract)
                self.J_prev_mat.mat().mult(vec_extract, vec_pred)
                ksp.solve(vec_pred, vec_solve)
            else:
                self.G_sqrt.getColumnVector(i - self.k, vec_extract)
                vec_extract.scale(np.sqrt(self.dt))
                ksp.solve(vec_extract, vec_solve)

            self.cov_sqrt_pred.setValues(rows=self.local_dofs, cols=i,
                                         values=vec_solve.getArray())

        self.cov_sqrt_pred.assemble()

        S = SLEPc.SVD(comm=self.comm)
        S.create()
        S.setOperator(self.cov_sqrt_pred)
        S.setDimensions(nsv=2 * self.k)
        S.setType(S.Type.CROSS)
        S.setCrossExplicitMatrix(True)
        S.setFromOptions()
        S.setUp()
        S.solve()

        rows = self.V.getOwnershipRange()
        V_rows = list(range(rows[0], rows[1]))
        v, u = self.cov_sqrt_pred.getVecs()

        sigmas = np.zeros((self.k, ))
        for i in range(self.k):
            sigmas[i] = S.getSingularTriplet(i, u, v)
            self.V.setValues(rows=V_rows, cols=i, values=v.getArray())

        print(sigmas[0], sigmas[-1])
        self.V.assemble()
        self.cov_sqrt_pred.matMult(self.V, result=self.cov_sqrt)

    def set_prev(self):
        """ Copy values into previous matrix. """
        fe.assign(self.du_prev, self.du)
        self.cov_sqrt.copy(result=self.cov_sqrt_prev,
                           structure=PETSc.Mat.Structure.SAME)
