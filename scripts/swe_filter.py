""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe

from scipy.sparse.linalg import splu, eigs
from scipy.linalg import cholesky, cho_factor, cho_solve, eigh
from statfenics.covariance import sq_exp_covariance
from statfenics.utils import dolfin_to_csr

from swe import ShallowOne, ShallowOneLinear

# initialise the logger
logger = logging.getLogger(__name__)


class ShallowOneKalman(ShallowOneLinear):
    def __init__(self, control, params, stat_params, lr=False):
        super().__init__(control=control, params=params)
        u, v = fe.TrialFunction(self.U_space), fe.TestFunction(self.U_space)
        M_u = fe.assemble(fe.inner(u, v) * fe.dx)
        M_u_scipy = dolfin_to_csr(M_u)

        u, v = fe.TrialFunction(self.H_space), fe.TestFunction(self.H_space)
        M_h = fe.assemble(fe.inner(u, v) * fe.dx)
        M_h_scipy = dolfin_to_csr(M_h)

        self.lr = lr
        self.mean = self.du.vector().get_local()

        # TODO: extend to allow forcing on multiple components
        # assume that forcing is 'h'-only, for now
        # rho = stat_params["rho"]
        # ell = stat_params["ell"]
        K_u = sq_exp_covariance(self.x_dofs_u,
                                stat_params["u_cov"]["rho"],
                                stat_params["u_cov"]["ell"])
        K_h = sq_exp_covariance(self.x_dofs_h,
                                stat_params["h_cov"]["rho"],
                                stat_params["h_cov"]["ell"])

        self.G = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        self.G[np.ix_(self.u_dofs, self.u_dofs)] = M_u_scipy @ K_u @ M_u_scipy.T
        self.G[np.ix_(self.h_dofs, self.h_dofs)] = M_h_scipy @ K_h @ M_h_scipy.T
        self.G[np.diag_indices_from(self.G)] += 1e-10

        if self.lr:
            self.k_init = stat_params["k_init"]
            self.k = stat_params["k"]

            # initialisation from above
            self.G_vals, self.G_vecs = eigs(self.G, self.k_init)
            self.G_vals, self.G_vecs = np.real(self.G_vals), np.real(self.G_vecs)
            self.G_sqrt = self.G_vecs @ np.diag(np.sqrt(self.G_vals))

            # covariance square-root
            self.cov_sqrt = np.zeros((self.mean.shape[0], self.k))
            self.cov_sqrt_prev = np.zeros((self.mean.shape[0], self.k))
            self.cov_sqrt_pred = np.zeros((self.mean.shape[0], self.k + self.k_init))
        else:
            # normal covariance structure
            self.cov = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_prev = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_pred = np.zeros((self.mean.shape[0], self.mean.shape[0]))

        self.A_mat = fe.assemble(self.a)
        self.A_prev = fe.derivative(self.L, self.du_prev)
        self.A_prev_mat = fe.assemble(self.A_prev)

        for A in [self.A_mat, self.A_prev_mat]: self.bcs.apply(A)

        self.A_scipy = dolfin_to_csr(self.A_mat)
        self.A_prev_scipy = dolfin_to_csr(self.A_prev_mat)


class ShallowOneEx(ShallowOne):
    def __init__(self, control, params, stat_params, lr=False):
        super().__init__(control=control, params=params)
        u, v = fe.TrialFunction(self.U_space), fe.TestFunction(self.U_space)
        M_u = fe.assemble(fe.inner(u, v) * fe.dx)
        M_u_scipy = dolfin_to_csr(M_u)

        u, v = fe.TrialFunction(self.H_space), fe.TestFunction(self.H_space)
        M_h = fe.assemble(fe.inner(u, v) * fe.dx)
        M_h_scipy = dolfin_to_csr(M_h)

        self.lr = lr
        self.mean = self.du.vector().get_local()

        # TODO: extend to allow forcing on multiple components
        # assume that forcing is 'h'-only, for now
        # rho = stat_params["rho"]
        # ell = stat_params["ell"]
        K_u = sq_exp_covariance(self.x_dofs_u,
                                stat_params["u_cov"]["rho"],
                                stat_params["u_cov"]["ell"])
        K_h = sq_exp_covariance(self.x_dofs_h,
                                stat_params["h_cov"]["rho"],
                                stat_params["h_cov"]["ell"])

        self.G = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        self.G[np.ix_(self.u_dofs, self.u_dofs)] = M_u_scipy @ K_u @ M_u_scipy.T
        self.G[np.ix_(self.h_dofs, self.h_dofs)] = M_h_scipy @ K_h @ M_h_scipy.T
        self.G[np.diag_indices_from(self.G)] += 1e-10

        if self.lr:
            self.k_init = stat_params["k_init"]
            self.k = stat_params["k"]

            # initialisation from above
            self.G_vals, self.G_vecs = eigs(self.G, self.k_init)
            self.G_vals, self.G_vecs = np.real(self.G_vals), np.real(self.G_vecs)
            self.G_sqrt = self.G_vecs @ np.diag(np.sqrt(self.G_vals))

            # covariance square-root
            self.cov_sqrt = np.zeros((self.mean.shape[0], self.k))
            self.cov_sqrt_prev = np.zeros((self.mean.shape[0], self.k))
            self.cov_sqrt_pred = np.zeros((self.mean.shape[0], self.k + self.k_init))
        else:
            # normal covariance structure
            self.cov = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_prev = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_pred = np.zeros((self.mean.shape[0], self.mean.shape[0]))

        self.J_mat = fe.assemble(self.J)
        self.J_prev = fe.derivative(self.F, self.du)
        self.J_prev_mat = fe.assemble(self.J_prev)

        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

    def prediction_step(self, t):
        if self.simulation == "tidal_flow":
            self.bcs[0] = fe.DirichletBC(self.W.sub(1), self.tidal_bc(t),
                                         self._left)

        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)
        self.assemble_derivatives()
        self.J_scipy_lu = splu(self.J_scipy.tocsc())

        if self.lr:
            # push cov. forward
            self.cov_sqrt_pred[:, :self.k] = self.J_prev_scipy @ self.cov_sqrt_prev
            self.cov_sqrt_pred[:, self.k:] = self.dt * self.G_sqrt
            self.cov_sqrt_pred[:] = self.J_scipy_lu.solve(self.cov_sqrt_pred)

            # perform reduction
            # TODO avoid reallocation
            D, V = eigh(self.cov_sqrt_pred.T @ self.cov_sqrt_pred)
            D, V = D[::-1], V[:, ::-1]
            logger.info("Prop. variance kept in the reduction: %f",
                        np.sum(D[0:self.k]) / np.sum(D))
            np.dot(self.cov_sqrt_pred, V[:, 0:self.k], out=self.cov_sqrt)
        else:
            self.cov_pred[:] = (self.J_prev_scipy @ self.cov_prev @ self.J_prev_scipy.T
                                + self.dt * self.G)

            self.cov_pred[:] = self.J_scipy_lu.solve(self.cov_pred.T)
            self.cov[:] = self.J_scipy_lu.solve(self.cov_pred.T)

    def update_step(self, y, H, sigma_y):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean
        cov_obs = (H @ self.cov_sqrt) @ (H @ self.cov_sqrt).T

        cov_obs[np.diag_indices_from(cov_obs)] += sigma_y**2 + 1e-10
        S_chol = cho_factor(cov_obs, lower=True)
        S_inv_y = cho_solve(S_chol, y - mean_obs)

        # kalman updates: for high-dimensions this is the bottleneck
        # TODO: avoid re-allocation and poor memory management
        if self.lr:
            HL = H @ self.cov_sqrt
            S_inv_HL = cho_solve(S_chol, HL)

            self.mean += self.cov_sqrt @ HL.T @ S_inv_y
            R = cholesky(np.eye(HL.shape[1]) - HL.T @ S_inv_HL, lower=True)
            self.cov_sqrt[:] = self.cov_sqrt @ R
        else:
            HC = H @ self.cov

            S_inv_HC = cho_solve(S_chol, HC)
            self.mean += HC.T @ S_inv_y
            self.cov -= HC.T @ S_inv_HC

        # update fenics state vector
        self.du.vector().set_local(self.mean.copy())

    def assemble_derivatives(self):
        fe.assemble(self.J, self.J_mat)
        fe.assemble(self.J_prev, self.J_prev_mat)

        # TODO: check application of temporally-varying BC
        for J in [self.J_mat, self.J_prev_mat]:
            for bc in self.bcs: bc.apply(J)

        # TODO: make use of constant sparsity pattern
        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

    def set_prev(self):
        """ Assign the current to the previous solution vector. """
        fe.assign(self.du_prev, self.du)
        if self.lr:
            self.cov_sqrt_prev[:] = self.cov_sqrt
        else:
            self.cov_prev[:] = self.cov
