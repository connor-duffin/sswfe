import pytest
import fenics as fe
import numpy as np
from ns_2d import NSTwo, NSSemiImplicit, NSSplit


def test_ns_two():
    ns = NSTwo(fe.RectangleMesh(fe.Point(0, 0), fe.Point(2.2, 0.41),
                                20, 10),
               dict(dt=1/1600))
    with pytest.raises(NotImplementedError):
        ns.setup_form()

    with pytest.raises(NotImplementedError):
        ns.solve(t=ns.dt)

    # some regression testing for the setup
    assert ns.t == 0.
    assert ns.rho is not None
    assert ns.nu is not None

    assert len(ns.x_coords) == 231
    assert ns.dx == 0.11739250401963523


def test_ns_semi_implicit():
    ns = NSSemiImplicit(
        fe.RectangleMesh(fe.Point(0, 0), fe.Point(2.2, 0.41), 20, 10),
        dict(dt=1/1600))

    # check UFL shapes
    u, p = fe.split(ns.du)
    assert u.ufl_shape == (2, )
    assert p.ufl_shape == ()

    # check all intialised to 0
    np.testing.assert_allclose(ns.du.vector().get_local(), 0.)
    np.testing.assert_allclose(ns.du_prev.vector().get_local(), 0.)
    np.testing.assert_allclose(ns.du_prev_prev.vector().get_local(), 0.)

    ns.setup_form()
    np.testing.assert_allclose(ns.du.vector().get_local(), 0.)
    assert np.linalg.norm(ns.du_prev.vector().get_local()) > 0.
    assert ns.t == ns.dt

    # check both solvers work
    ns.solve(krylov=False)
    ns.solve(krylov=True)
    assert ns.t == 3 * ns.dt

    # check that we set the previous timepoint OK
    np.testing.assert_allclose(ns.du_prev.vector().get_local(),
                               ns.du.vector().get_local())


def test_ns_operator_splitting():
    ns = NSSplit(
        fe.RectangleMesh(fe.Point(0, 0), fe.Point(2.2, 0.41), 20, 10),
        dict(dt=1/1600))

    # check UFL shapes
    assert ns.u.ufl_shape == (2, )
    assert ns.p.ufl_shape == ()

    # check both solvers
    ns.setup_form()
    ns.solve(krylov=False)
    ns.solve(krylov=True)
    assert ns.t == 2 * ns.dt

    # check that setting previous is OK
    np.testing.assert_allclose(ns.u.vector().get_local(),
                               ns.u_prev.vector().get_local())
