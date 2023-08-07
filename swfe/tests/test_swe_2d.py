import os
import pytest
import numpy as np
import fenics as fe

from numpy.testing import assert_allclose
from swfe.swe_2d import (ShallowTwo,
                         ShallowTwoFilter,
                         ShallowTwoFilterPETSc)
from scipy.sparse.linalg import spsolve
from statfenics.utils import dolfin_to_csr, build_observation_operator

from petsc4py import PETSc


def rmse(y_est, y):
    n_obs = len(y)
    return np.linalg.norm(y_est - y, ord=2) / np.sqrt(n_obs)


@pytest.fixture
def swe_2d():
    mesh = fe.RectangleMesh(fe.Point(0., 0.), fe.Point(2., 1.), 32, 16)
    params = dict(nu=1e-2, C=0., H=0.05, g=9.8,
                  u_inflow=0.1, inflow_period=120.,
                  length=2., width=1.)
    control = dict(dt=0.01,
                   theta=0.5,
                   simulation="laminar",
                   use_imex=False,
                   use_les=False)
    # params = {"nu": 1e-2, "C": 0., "H": 0.05, "g": 9.8, "u_inflow": 0.1, "inflow_period": 120}
    # control = {"dt": 0.01, "theta": 0.5, "simulation": "laminar", "use_imex": False, "use_les": False}
    return ShallowTwo(mesh, params, control)


def test_shallowtwo_init(swe_2d):
    assert swe_2d.length == 2.
    assert swe_2d.width == 1.

    assert swe_2d.nu == 0.01
    assert swe_2d.C == 0.
    assert swe_2d.H == 0.05

    assert len(swe_2d.du_vertices) == 3 * len(swe_2d.x_coords)

    swe_2d.setup_form()
    swe_2d.setup_solver(use_ksp=False)
    assert type(swe_2d.solver) == fe.NonlinearVariationalSolver


def test_shallowtwo_solve(swe_2d):
    swe_2d.inlet_velocity.t = 30.  # max of period

    def setup_and_step(swe_2d, use_imex=False, atol=1e-6):
        swe_2d.use_imex = use_imex
        # verify one step with CN
        swe_2d.setup_form()
        swe_2d.setup_solver()

        u_p = np.copy(swe_2d.du_prev.vector().get_local())
        swe_2d.solve()
        u = np.copy(swe_2d.du.vector().get_local())

        assert np.linalg.norm(u - u_p) >= atol

    setup_and_step(swe_2d)
    setup_and_step(swe_2d, use_imex=True)


def test_shallowtwo_ss(swe_2d):
    assert swe_2d.steady_state(swe_2d.du, swe_2d.du_prev)
    swe_2d.set_prev_vector(np.ones_like(swe_2d.du_prev.vector().get_local()))
    assert not swe_2d.steady_state(swe_2d.du, swe_2d.du_prev)


def test_shallowtwo_jac_bc(swe_2d):
    assert swe_2d.length == 2.
    assert swe_2d.width == 1.

    swe_2d.setup_form()
    swe_2d.setup_solver()

    for i in range(10):
        swe_2d.solve()

        J_approx = fe.assemble(fe.derivative(swe_2d.F, swe_2d.du))
        J_model = fe.assemble(swe_2d.J)
        np.testing.assert_allclose(J_approx.array(), J_model.array())

        fe.assign(swe_2d.du_prev, swe_2d.du)

    # check integration along the boundaries is sensible
    n = fe.FacetNormal(swe_2d.mesh)
    ds = fe.Measure('ds', domain=swe_2d.mesh, subdomain_data=swe_2d.boundaries)

    h_test = fe.Function(swe_2d.H_space)
    h_test.interpolate(fe.Expression("cos(x[0]) * sin(x[1])", degree=4))
    assert np.abs(fe.assemble(h_test * ds(1)) - 0.4596976941318603) / np.abs(0.4596976941318603) <= 1e-3
    assert np.abs(fe.assemble(h_test * ds(2)) - (-0.1913017411809895)) / np.abs(-0.1913017411809895) <= 1e-3

    u_test = fe.Function(swe_2d.U_space)
    u_test.interpolate(fe.Expression(("1.0", "0.0"), degree=4))
    assert np.abs(fe.assemble(fe.inner(u_test, n) * ds(1)) + 1.) <= 1e-8
    assert np.abs(fe.assemble(fe.inner(u_test, n) * ds(2)) - 1.) <= 1e-8


def test_shallowtwo_save(swe_2d):
    f = "temp.h5"
    swe_2d.setup_checkpoint(f)
    du_true = fe.Expression(("t * x[0]", "t * x[1]", "t * x[0] * x[1]"),
                            t=0.,
                            degree=2)

    t = np.linspace(0, 1, 10)
    for i, t_curr in enumerate(t):
        du_true.t = t_curr
        swe_2d.du.interpolate(du_true)
        swe_2d.checkpoint_save(t_curr)

    swe_2d.checkpoint_close()

    checkpoint = fe.HDF5File(swe_2d.mesh.mpi_comm(), f, "r")
    du_true_function = fe.Function(swe_2d.W)
    for i, t_curr in enumerate(t):
        du_true.t = t_curr
        du_true_function.interpolate(du_true)

        vec_name = f"/du/vector_{i}"
        checkpoint.read(swe_2d.du, vec_name)  # read into du
        timestamp = checkpoint.attributes(vec_name)["timestamp"]
        result = swe_2d.du.vector() - du_true_function.vector()
        assert timestamp == t_curr
        np.testing.assert_allclose(result.get_local(), 0.)

    checkpoint.close()
    os.remove(f)


def test_shallowtwo_filter():
    mesh = fe.RectangleMesh(fe.Point(0., 0.),
                            fe.Point(2., 1.), 32, 16)
    params = {"nu": 1e-2, "C": 0., "H": 0.05, "g": 9.8,
              "u_inflow": 0.004, "inflow_period": 120,
              "length": 2., "width": 1.}
    control = {"dt": 0.01,
               "theta": 0.5,
               "simulation": "laminar",
               "use_imex": False,
               "use_les": False}
    swe = ShallowTwoFilter(mesh, params, control)
    swe.setup_form()
    swe.setup_solver()

    assert swe.length == 2.
    assert swe.width == 1.

    # check that all the dofs line up
    assert_allclose(np.unique(swe.W.dofmap().dofs()),
                    np.unique(np.concatenate((swe.u_dofs,
                                              swe.v_dofs,
                                              swe.h_dofs))))

    # setup filter (basically compute prior additive noise covariance)
    stat_params = dict(rho_u=1., rho_v=1., rho_h=1.,
                       ell_u=0.5, ell_v=0.5, ell_h=0.5,
                       k_init_u=16, k_init_v=16, k_init_h=16, k=16)
    swe.setup_filter(stat_params)

    # as on the same fcn space
    assert_allclose(swe.Ku_vals, swe.Kv_vals)
    # not too fazed on these: as long as they are ~ok
    assert_allclose(swe.Ku_vals, swe.Kh_vals, atol=1e-2)

    # check that cross-correlations are 0, accordingly
    u, v = fe.TrialFunction(swe.W), fe.TestFunction(swe.W)
    M = fe.assemble(fe.inner(u, v) * fe.dx)
    M_scipy = dolfin_to_csr(M)

    K_sqrt = spsolve(M_scipy, swe.G_sqrt)
    K = K_sqrt @ K_sqrt.T
    assert_allclose(K[np.ix_(swe.u_dofs, swe.v_dofs)], 0.)
    assert_allclose(K[np.ix_(swe.u_dofs, swe.h_dofs)], 0.)
    assert_allclose(K[np.ix_(swe.v_dofs, swe.u_dofs)], 0.)
    assert_allclose(K[np.ix_(swe.v_dofs, swe.h_dofs)], 0.)
    assert_allclose(K[np.ix_(swe.h_dofs, swe.u_dofs)], 0.)
    assert_allclose(K[np.ix_(swe.h_dofs, swe.v_dofs)], 0.)

    # check BC's are zero
    U, V = swe.U.split()
    U_space, V_space = U.collapse(), V.collapse()

    for space, global_space in zip([U_space, V_space, swe.H_space],
                                   [swe.W.sub(0).sub(0), swe.W.sub(0).sub(1), swe.W.sub(1)]):
        g = fe.Function(space)
        g.interpolate(fe.Expression("1.", degree=4), )

        def boundary(x, on_boundary):
            return on_boundary

        bc = fe.DirichletBC(space, fe.Constant((0)), boundary)
        bc.apply(g.vector())
        dofs = np.isclose(g.vector().get_local(), 0.)
        bc_dofs = np.array(global_space.dofmap().dofs())[dofs]

        assert_allclose(K[bc_dofs, bc_dofs], 0., atol=1e-12)


def test_shallowtwo_filter_conditioning():
    np.random.seed(27)
    mesh = fe.RectangleMesh(fe.Point(0., 0.),
                            fe.Point(2., 1.), 32, 16)
    params = {"nu": 1e-2, "C": 0., "H": 0.05, "g": 9.8,
              "u_inflow": 0.004, "inflow_period": 120,
              "length": 2., "width": 1.}
    control = {"dt": 0.01,
               "theta": 0.5,
               "simulation": "laminar",
               "use_imex": False,
               "use_les": False}
    swe = ShallowTwoFilter(mesh, params, control)
    swe.setup_form()
    swe.setup_solver()

    # check setup
    assert swe.length == 2.
    assert swe.width == 1.

    # observe every 10th point + sanity check
    x_obs = swe.x_coords[::10]
    H = build_observation_operator(x_obs, swe.W, sub=(1,), out="scipy")
    assert x_obs.shape[1] == 2
    assert H.shape[0] == x_obs.shape[0]
    assert H.shape[1] == len(swe.W.tabulate_dof_coordinates())

    # setup filter: noise is approx 5% of state magnitude
    rho = 1e-4
    ell = 0.25
    stat_params = dict(rho_u=rho, rho_v=rho, rho_h=rho,
                       ell_u=ell, ell_v=ell, ell_h=ell,
                       k_init_u=16, k_init_v=16, k_init_h=16, k=64,
                       H=H, sigma_y=0.01 * params["u_inflow"])
    swe.setup_filter(stat_params)

    # increment forward SLIGHTLY! otherwise useless
    t = swe.dt
    swe.inlet_velocity.t = t
    swe.prediction_step(t)

    u, v, h = swe.get_vertex_values()
    y_denoised = h[::10]
    y = (y_denoised
         + stat_params["sigma_y"] * np.random.normal(size=y_denoised.shape))
    np.testing.assert_allclose(H @ swe.mean, y_denoised)

    # check that normal updating is OK
    mean_pred = swe.mean.copy()
    cov_sqrt_pred = swe.cov_sqrt.copy()
    swe.update_step(y)

    mean_chol = swe.mean.copy()
    cov_sqrt_chol = swe.cov_sqrt.copy()

    swe.mean[:] = mean_pred
    swe.cov_sqrt[:] = cov_sqrt_pred
    swe.update_step_svd(y)

    # expect that the corrections are similar but not identical:
    # they use a different covariance updating method
    np.testing.assert_allclose(
        mean_chol, swe.mean, atol=1e-8)

    # RMSEs should be  similar
    np.testing.assert_allclose(
        rmse(H @ swe.mean, y), rmse(H @ mean_chol, y),
        atol=1e-8
    )

    # as should the marginal variances
    var_chol = np.sum(cov_sqrt_chol**2, axis=1)
    var_svd = np.sum(swe.cov_sqrt**2, axis=1)
    np.testing.assert_allclose(var_svd, var_chol, atol=1e-8)


def test_shallowtwo_filter_petsc():
    np.random.seed(27)
    mesh = fe.RectangleMesh(fe.Point(0., 0.),
                            fe.Point(2., 1.), 32, 16)
    params = {"nu": 1e-2, "C": 0., "H": 0.05, "g": 9.8,
              "u_inflow": 0.004, "inflow_period": 120,
              "length": 2., "width": 1.}
    control = {"dt": 0.01,
               "theta": 0.5,
               "simulation": "laminar",
               "use_imex": False,
               "use_les": False}
    swe = ShallowTwoFilterPETSc(mesh, params, control, comm=fe.MPI.comm_world)
    swe.setup_form()
    swe.setup_solver()

    # check setup
    assert swe.length == 2.
    assert swe.width == 1.

    # observe every 10th point + sanity check running
    x_obs = swe.x_coords[::5]
    H = build_observation_operator(x_obs, swe.W, sub=(1,), out="petsc")

    # setup filter: noise is approx 5% of state magnitude
    rho = 1e-4
    ell = 0.25
    stat_params = dict(rho_u=rho, rho_v=rho, rho_h=rho,
                       ell_u=ell, ell_v=ell, ell_h=ell,
                       k_init_u=16, k_init_v=16, k_init_h=16, k=64,
                       H=H, sigma_y=0.01 * params["u_inflow"])
    swe.setup_filter(stat_params)
    assert swe.n_obs == len(x_obs)
    assert swe.mean_obs.getSize() == swe.n_obs
    assert swe.HL.getSize() == (swe.n_obs, swe.k)

    # check that we can just do these operations
    swe.setup_prior_covariance()
    swe.H.matMult(swe.cov_sqrt, result=swe.HL)

    # create and setup as needed
    y = PETSc.Vec().create(comm=swe.comm)
    y.setSizes(swe.n_obs)
    y.setUp()
    y.assemble()

    # increment forward SLIGHTLY! otherwise useless
    t = swe.dt
    swe.inlet_velocity.t = t
    swe.prediction_step(t)

    # and get the obs. operator to work
    H.mult(swe.mean, y)

    # generate random Gaussian noise
    noise = np.random.normal(0, swe.sigma_y, swe.n_obs)
    noise_vec = PETSc.Vec().createWithArray(noise)
    y.axpy(1.0, noise_vec)

    # now perform the update:
    # sanity check that i don't break anything
    swe.update_step(y)
    assert swe.mean.norm() >= 1e-10
    assert swe.cov_sqrt.norm() >= 1e-10
