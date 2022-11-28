""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe

from scipy.sparse.linalg import splu, eigs
from scipy.linalg import cholesky, cho_factor, cho_solve, eigh

from statfenics.covariance import (sq_exp_covariance,
                                   sq_exp_evd_hilbert,
                                   sq_exp_evd)
from statfenics.utils import dolfin_to_csr
from swe import ShallowOne, ShallowOneLinear

# initialise the logger
logger = logging.getLogger(__name__)


class ShallowOneFilter:
    def __init__(self, stat_params, lr=False):
        u, v = fe.TrialFunction(self.U_space), fe.TestFunction(self.U_space)
        M_u = fe.assemble(fe.inner(u, v) * fe.dx)
        M_u_scipy = dolfin_to_csr(M_u)

        u, v = fe.TrialFunction(self.H_space), fe.TestFunction(self.H_space)
        M_h = fe.assemble(fe.inner(u, v) * fe.dx)
        M_h_scipy = dolfin_to_csr(M_h)

        u, v = fe.TrialFunction(self.W), fe.TestFunction(self.W)
        M = fe.assemble(fe.inner(u, v) * fe.dx)
        M_scipy = dolfin_to_csr(M)

        self.lr = lr
        self.mean = self.du.vector().get_local()

        if self.lr:
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
                if stat_params["hilbert_gp"]:
                    Ku_vals, Ku_vecs = sq_exp_evd_hilbert(
                        self.U_space, self.k_init_u,
                        stat_params["rho_u"],
                        stat_params["ell_u"])
                else:
                    Ku_vals, Ku_vecs = sq_exp_evd(self.x_dofs_u,
                                                  stat_params["rho_u"],
                                                  stat_params["ell_u"],
                                                  k=self.k_init_u)

                self.G_sqrt[self.u_dofs, 0:len(Ku_vals)] = (
                    Ku_vecs @ np.diag(np.sqrt(Ku_vals)))

            if stat_params["rho_h"] > 0.:
                if stat_params["hilbert_gp"]:
                    Kh_vals, Kh_vecs = sq_exp_evd_hilbert(
                        self.H_space, self.k_init_h,
                        stat_params["rho_h"],
                        stat_params["ell_h"])
                else:
                    Kh_vals, Kh_vecs = sq_exp_evd(self.x_dofs_h,
                                                  stat_params["rho_h"],
                                                  stat_params["ell_h"],
                                                  k=self.k_init_h)
                self.G_sqrt[self.h_dofs, self.k_init_u:(self.k_init_u + len(Kh_vals))] = (
                    Kh_vecs @ np.diag(np.sqrt(Kh_vals)))

            # multiplication *after* the initial construction
            self.G_sqrt[:] = M_scipy @ self.G_sqrt
        else:
            K_u = sq_exp_covariance(self.x_dofs_u,
                                    stat_params["rho_u"],
                                    stat_params["ell_u"])
            K_h = sq_exp_covariance(self.x_dofs_h,
                                    stat_params["rho_h"],
                                    stat_params["ell_h"])

            self.G = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.G[np.ix_(self.u_dofs, self.u_dofs)] = M_u_scipy @ K_u @ M_u_scipy.T
            self.G[np.ix_(self.h_dofs, self.h_dofs)] = M_h_scipy @ K_h @ M_h_scipy.T
            self.G[np.diag_indices_from(self.G)] += 1e-10

            # normal covariance structure
            self.cov = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_prev = np.zeros((self.mean.shape[0], self.mean.shape[0]))
            self.cov_pred = np.zeros((self.mean.shape[0], self.mean.shape[0]))

    def prediction_step(self, t):
        raise NotImplementedError

    def compute_lml(self, y, H, sigma_y):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean
        n_obs = len(mean_obs)

        if self.lr:
            HL = H @ self.cov_sqrt
            cov_obs = HL @ HL.T
        else:
            HC = H @ self.cov
            cov_obs = H @ self.cov @ H.T

        cov_obs[np.diag_indices_from(cov_obs)] += sigma_y**2 + 1e-10
        S_chol = cho_factor(cov_obs, lower=True)
        S_inv_y = cho_solve(S_chol, y - mean_obs)
        log_det = 2 * np.sum(np.log(np.diag(S_chol[0])))

        return (- S_inv_y @ S_inv_y / 2
                - log_det / 2
                - n_obs * np.log(2 * np.pi) / 2)

    def update_step(self, y, H, sigma_y):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean

        if self.lr:
            HL = H @ self.cov_sqrt
            cov_obs = HL @ HL.T
        else:
            HC = H @ self.cov
            cov_obs = H @ self.cov @ H.T

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

    def set_prev(self):
        """ Assign the current to the previous solution vector. """
        fe.assign(self.du_prev, self.du)
        if self.lr:
            self.cov_sqrt_prev[:] = self.cov_sqrt
        else:
            self.cov_prev[:] = self.cov


class ShallowOneEx(ShallowOne, ShallowOneFilter):
    def __init__(self, control, params, stat_params, lr=False):
        ShallowOne.__init__(self, control=control, params=params)
        ShallowOneFilter.__init__(self, stat_params=stat_params, lr=lr)

        self.J = fe.derivative(self.F, self.du)
        self.J_prev = fe.derivative(self.F, self.du_prev)

        self.J_mat = fe.assemble(self.J)
        self.J_prev_mat = fe.assemble(self.J_prev)

        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

    def prediction_step(self, t):
        if self.simulation == "tidal_flow":
            self.bcs[0] = fe.DirichletBC(self.W.sub(1), self.tidal_bc(t),
                                         self._left)

        # solve for the mean
        fe.solve(self.F == 0, self.du, bcs=self.bcs, J=self.J)
        self.mean[:] = self.du.vector().get_local()

        self.assemble_derivatives()
        self.J_scipy_lu = splu(self.J_scipy.tocsc())

        if self.lr:
            # push cov. forward
            # print(self.J_prev_scipy.todense())
            # print(self.J_prev_scipy @ self.cov_sqrt_prev)
            self.cov_sqrt_pred[:, :self.k] = self.J_prev_scipy @ self.cov_sqrt_prev
            self.cov_sqrt_pred[:, self.k:] = self.dt * self.G_sqrt
            self.cov_sqrt_pred[:] = self.J_scipy_lu.solve(self.cov_sqrt_pred)

            # perform reduction
            # TODO avoid reallocation
            D, V = eigh(self.cov_sqrt_pred.T @ self.cov_sqrt_pred)
            D, V = D[::-1], V[:, ::-1]
            logger.debug("Prop. variance kept in the reduction: %f",
                         np.sum(D[0:self.k]) / np.sum(D))
            np.dot(self.cov_sqrt_pred, V[:, 0:self.k], out=self.cov_sqrt)
        else:
            self.cov_pred[:] = (self.J_prev_scipy @ self.cov_prev @ self.J_prev_scipy.T
                                + self.dt * self.G)

            self.cov_pred[:] = self.J_scipy_lu.solve(self.cov_pred)
            self.cov[:] = self.J_scipy_lu.solve(self.cov_pred.T)

    def assemble_derivatives(self):
        self.J_mat = fe.assemble(self.J)
        self.J_prev_mat = fe.assemble(self.J_prev)

        # TODO: check application of temporally-varying BC
        for J in [self.J_mat, self.J_prev_mat]:
            if self.simulation == "immersed_bump":
                self.bcs.apply(J)
            else:
                for bc in self.bcs: bc.apply(J)

        # TODO: make use of constant sparsity pattern
        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)


class ShallowOneKalman(ShallowOneLinear, ShallowOneFilter):
    def __init__(self, control, params, stat_params, lr=False):
        ShallowOneLinear.__init__(self, control=control, params=params)
        ShallowOneFilter.__init__(self, stat_params=stat_params, lr=lr)

        self.A_mat = fe.assemble(self.a)
        self.A_prev = fe.derivative(self.L, self.du_prev)
        self.A_prev_mat = fe.assemble(self.A_prev)

        for A in [self.A_mat, self.A_prev_mat]: self.bcs.apply(A)

        self.A_scipy = dolfin_to_csr(self.A_mat)
        self.A_scipy_lu = splu(self.A_scipy.tocsc())
        self.A_prev_scipy = dolfin_to_csr(self.A_prev_mat)

    def prediction_step(self, t):
        self.solver.solve()
        self.mean[:] = self.du.vector().get_local()

        if self.lr:
            # push cov. forward
            self.cov_sqrt_pred[:, :self.k] = self.A_prev_scipy @ self.cov_sqrt_prev
            self.cov_sqrt_pred[:, self.k:] = self.dt * self.G_sqrt
            self.cov_sqrt_pred[:] = self.A_scipy_lu.solve(self.cov_sqrt_pred)

            # perform reduction
            # TODO avoid reallocation
            D, V = eigh(self.cov_sqrt_pred.T @ self.cov_sqrt_pred)
            D, V = D[::-1], V[:, ::-1]
            logger.debug("Prop. variance kept in the reduction: %f",
                         np.sum(D[0:self.k]) / np.sum(D))
            np.dot(self.cov_sqrt_pred, V[:, 0:self.k], out=self.cov_sqrt)
        else:
            self.cov_pred[:] = (
                self.A_prev_scipy @ self.cov_prev @ self.A_prev_scipy.T
                + self.dt * self.G)

            self.cov_pred[:] = self.A_scipy_lu.solve(self.cov_pred)
            self.cov[:] = self.A_scipy_lu.solve(self.cov_pred.T)
