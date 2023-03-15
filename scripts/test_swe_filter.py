import os
import pytest
import numpy as np
import fenics as fe

from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from statfenics.covariance import sq_exp_covariance, sq_exp_evd, sq_exp_evd_hilbert
from statfenics.utils import dolfin_to_csr

from swe_filter import ShallowOneEx, ShallowOneKalman


def test_1d_linear_filter():
    control = {"nx": 32, "dt": 0.5, "theta": 1.0, "simulation": "immersed_bump"}
    params = {"nu": 0., "bump_centre": 10}
    stat_params = dict(rho_u=1e-2, ell_u=5000.,
                       rho_h=0., ell_h=5000.)
    # k=k, k_init_u=k, k_init_h=k

    # init according to system
    swe = ShallowOneKalman(control, params, stat_params, lr=False)

    # verify our construction/filtering is OK
    du = fe.Function(swe.W)
    du_prev = fe.Function(swe.W)
    v_u, v_h = fe.TestFunctions(swe.W)

    u, h = fe.split(du)
    u_prev, h_prev = fe.split(du_prev)

    dt = swe.dt
    u_theta = swe.theta * u + (1 - swe.theta) * u_prev
    h_theta = swe.theta * h + (1 - swe.theta) * h_prev

    F = (fe.inner(h - h_prev, v_h) / dt * fe.dx
         + (swe.H * u_theta).dx(0) * v_h * fe.dx
         + fe.inner(u - u_prev, v_u) / dt * fe.dx
         + 9.8 * h_theta.dx(0) * v_u * fe.dx)

    J = fe.assemble(fe.derivative(F, du))
    swe.bcs.apply(J)
    J_scipy = dolfin_to_csr(J)
    np.testing.assert_allclose(swe.A_scipy.todense(), J_scipy.todense())


def test_1d_nonlinear_filter():
    control = {"nx": 32, "dt": 1., "theta": 1.0, "simulation": "tidal_flow"}
    params = {"nu": 1.}
    stat_params = dict(rho_u=1., ell_u=5000.,
                       rho_h=0., ell_h=5000.)
    swe = ShallowOneEx(control, params, stat_params, lr=False)

    # check covariance initialisation
    u, v = fe.TrialFunction(swe.H_space), fe.TestFunction(swe.H_space)
    M = fe.assemble(fe.inner(u, v) * fe.dx)
    M = dolfin_to_csr(M)
    K = sq_exp_covariance(swe.x_dofs_h,
                          stat_params["rho_h"],
                          stat_params["ell_h"])
    G = M @ K @ M.T

    for row, idx in enumerate(swe.h_dofs):
        assert_allclose(swe.G[idx, swe.h_dofs], G[row, :], atol=1e-8)

    swe.assemble_derivatives()
    assert type(swe.J_scipy) == csr_matrix
    assert type(swe.J_prev_scipy) == csr_matrix

    # check previous allocation
    u_test = np.sin(swe.x_dofs[:, 0])
    cov_test = np.random.normal() * np.eye(swe.n_dofs)
    assert u_test.shape[0] == swe.n_dofs
    swe.du.vector().set_local(u_test)
    swe.cov[:] = cov_test
    swe.set_prev()
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


def test_1d_filter_lr():
    k = 16
    control = {"nx": 32, "dt": 1., "theta": 1.0, "simulation": "tidal_flow"}
    params = {"nu": 1.}
    stat_params = dict(rho_u=1, ell_u=5000.,
                       rho_h=1, ell_h=5000.,
                       k=k, k_init_u=k, k_init_h=k, hilbert_gp=False)

    swe = ShallowOneEx(control, params, stat_params, lr=True)
    u, v = fe.TrialFunction(swe.W), fe.TestFunction(swe.W)
    M = fe.assemble(fe.inner(u, v) * fe.dx)
    M_scipy = dolfin_to_csr(M)

    # check dimensions
    assert swe.cov_sqrt_pred.shape == (98, 48)
    assert swe.cov_sqrt_prev.shape == (98, 16)
    assert swe.cov_sqrt.shape == (98, 16)

    # TODO: unit test update steps
    swe.prediction_step(0.)

    # first, check that full construction is correct
    G_full = np.zeros((swe.mean.shape[0], swe.mean.shape[0]))
    Ku_vals, Ku_vecs = sq_exp_evd(swe.x_dofs_u,
                                  stat_params["rho_u"],
                                  stat_params["ell_u"],
                                  k=swe.k_init_u)
    Kh_vals, Kh_vecs = sq_exp_evd(swe.x_dofs_h,
                                  stat_params["rho_h"],
                                  stat_params["ell_h"],
                                  k=swe.k_init_h)

    # compute initial EVD approximation
    G_full[np.ix_(swe.h_dofs, swe.h_dofs)] = (
        Kh_vecs @ np.diag(Kh_vals) @ Kh_vecs.T)
    G_full[np.ix_(swe.u_dofs, swe.u_dofs)] = (
        Ku_vecs @ np.diag(Ku_vals) @ Ku_vecs.T)

    # check all ok
    G_sqrt = swe.G_sqrt.copy()
    G_full[:] = M_scipy @ G_full @ M_scipy.T
    np.testing.assert_allclose(swe.G_sqrt[swe.u_dofs, swe.k_init_u:], 0.)
    np.testing.assert_allclose(swe.G_sqrt[swe.h_dofs, :swe.k_init_h], 0.)
    np.testing.assert_allclose(swe.G_sqrt @ swe.G_sqrt.T, G_full)

    # now test hilbert-GP approach
    stat_params.update(hilbert_gp=True)
    swe = ShallowOneEx(control, params, stat_params, lr=True)
    Ku_vals, Ku_vecs = sq_exp_evd_hilbert(swe.U_space, swe.k_init_u,
                                          stat_params["rho_u"],
                                          stat_params["ell_u"])
    Kh_vals, Kh_vecs = sq_exp_evd_hilbert(swe.H_space, swe.k_init_h,
                                          stat_params["rho_h"],
                                          stat_params["ell_h"])

    G_sqrt_hilbert = swe.G_sqrt.copy()
    G_full[:] = 0.
    G_full[np.ix_(swe.u_dofs, swe.u_dofs)] = (
        Ku_vecs @ np.diag(Ku_vals) @ Ku_vecs.T)
    G_full[np.ix_(swe.h_dofs, swe.h_dofs)] = (
        Kh_vecs @ np.diag(Kh_vals) @ Kh_vecs.T)
    G_full[:] = M_scipy @ G_full @ M_scipy.T

    np.testing.assert_allclose(swe.G_sqrt[swe.u_dofs, swe.k_init_u:], 0.)
    np.testing.assert_allclose(swe.G_sqrt[swe.h_dofs, :swe.k_init_h], 0.)
    np.testing.assert_allclose(swe.G_sqrt @ swe.G_sqrt.T, G_full)

    # regression test to see that computations are the same
    np.testing.assert_allclose(np.linalg.norm(G_sqrt - G_sqrt_hilbert), 3827.659732)
