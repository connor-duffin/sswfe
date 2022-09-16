import os
import pytest
import numpy as np
import fenics as fe

from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from statfenics.covariance import sq_exp_covariance
from statfenics.utils import dolfin_to_csr

from swe_filter import ShallowOneEx


def test_1d_filter():
    control = {"nx": 32, "dt": 1., "theta": 1.0, "simulation": "tidal_flow"}
    stat_params = {"rho": 1e-2, "ell": 100}
    swe = ShallowOneEx(control, stat_params, lr=False)

    # check covariance initialisation
    u, v = fe.TrialFunction(swe.H_space), fe.TestFunction(swe.H_space)
    M = fe.assemble(fe.inner(u, v) * fe.dx)
    M = dolfin_to_csr(M)
    K = sq_exp_covariance(swe.x_dofs_h, stat_params["rho"], stat_params["ell"])
    G = M @ K @ M.T

    for row, idx in enumerate(swe.h_dofs):
        assert_allclose(G[row, :], swe.G[idx, swe.h_dofs])

    swe.assemble_derivatives()
    assert type(swe.J_scipy) == csr_matrix
    assert type(swe.J_prev_scipy) == csr_matrix

    # check previous allocation
    u_test = np.sin(swe.x_dofs[:, 0])
    cov_test = np.random.normal() * np.eye(swe.n_dofs)
    assert u_test.shape[0] == swe.n_dofs
    swe.du.vector().set_local(u_test)
    swe.cov[:] = cov_test
    swe.set_prev(cov=True)
    assert_allclose(swe.du_prev.vector().get_local(), u_test)
    assert_allclose(swe.cov_prev, cov_test)

    # check sparsity pattern (experimental)
    swe.assemble_derivatives()
    jacobian_alt = swe.J_scipy.copy()
    for i in range(100):
        scale, shift = np.random.normal(size=(2, ))
        u_curr = np.sin(scale * swe.x_dofs[:, 0] + shift)**2
        u_prev = np.cos(scale * swe.x_dofs[:, 0] + shift)**2

        # verification
        swe.du.vector().set_local(u_curr)
        swe.du_prev.vector().set_local(u_prev)
        swe.assemble_derivatives()

        assert swe.J_scipy.nnz == jacobian_alt.nnz
        assert_allclose(swe.J_scipy.indices, jacobian_alt.indices)
        assert_allclose(swe.J_scipy.indptr, jacobian_alt.indptr)
