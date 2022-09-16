""" Solve the Shallow-water equations in non-conservative form. """
import logging

import numpy as np
import fenics as fe

import matplotlib.pyplot as plt

from scipy.sparse.linalg import splu
from scipy.linalg import cho_factor, cho_solve
from statfenics.covariance import sq_exp_covariance
from statfenics.utils import dolfin_to_csr

from swe import ShallowOne

# initialise the logger
logger = logging.getLogger(__name__)


class ShallowOneEx(ShallowOne):
    def __init__(self, control, stat_params, lr=False):
        super().__init__(control=control)

        self.U, self.H = self.W.split()
        self.U_space = self.U.collapse()
        self.H_space = self.H.collapse()

        u, v = fe.TrialFunction(self.H_space), fe.TestFunction(self.H_space)
        M_h = fe.assemble(fe.inner(u, v) * fe.dx)
        M_h_scipy = dolfin_to_csr(M_h)

        self.lr = lr
        self.mean = self.du.vector().get_local()

        self.G = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        self.cov = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        self.cov_prev = np.zeros((self.mean.shape[0], self.mean.shape[0]))
        self.cov_pred = np.zeros((self.mean.shape[0], self.mean.shape[0]))

        # assume that forcing is 'h'-only, for now
        rho = stat_params["rho"]
        ell = stat_params["ell"]
        K = sq_exp_covariance(self.x_dofs_h, rho, ell)

        self.G[np.ix_(self.h_dofs, self.h_dofs)] = M_h_scipy @ K @ M_h_scipy.T

        if lr:
            pass

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
        self.cov_pred[:] = (self.J_prev_scipy @ self.cov_prev @ self.J_prev_scipy
                            + self.dt * self.G)
        self.cov_pred[:] = self.J_scipy_lu.solve(self.cov_pred.T)
        self.cov[:] = self.J_scipy_lu.solve(self.cov_pred.T)

    def update_step(self, y, H, sigma_y):
        self.mean[:] = self.du.vector().get_local()
        mean_obs = H @ self.mean
        cov_obs = H @ self.cov @ H.T

        cov_obs[np.diag_indices_from(cov_obs)] += sigma_y**2 + 1e-8

        # kalman updates: for high-dimensions this is the bottleneck
        HC = H @ self.cov
        S_chol = cho_factor(cov_obs, lower=True)
        self.mean += HC.T @ (cho_solve(S_chol, y - mean_obs))
        self.cov -= HC.T @ (cho_solve(S_chol, HC))

        # update fenics state vector
        self.du.vector().set_local(self.mean)

    def assemble_derivatives(self):
        fe.assemble(self.J, self.J_mat)
        fe.assemble(self.J_prev, self.J_prev_mat)
        for J in [self.J_mat, self.J_prev_mat]:
            for bc in self.bcs: bc.apply(J)

        # TODO: check to see if sparsity pattern changes
        self.J_scipy = dolfin_to_csr(self.J_mat)
        self.J_prev_scipy = dolfin_to_csr(self.J_prev_mat)

    def set_prev(self, cov=False):
        """ Assign the current to the previous solution vector. """
        fe.assign(self.du_prev, self.du)
        if cov:
            self.cov_prev[:] = self.cov
