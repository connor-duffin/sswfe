import pytest
import numpy as np
import fenics as fe

from numpy.testing import assert_allclose
from swe import ShallowOneLinear, ShallowOne, BumpTopo


def test_shallowone_linear_init():
    control = {"nx": 32, "dt": 0.02, "theta": 1.0}
    params = {"nu": 1.0, "bump_centre": 10.}

    swe = ShallowOneLinear(control, params)
    assert len(swe.x_dofs_u) == 65
    assert len(swe.x_dofs_h) == 33
    assert len(swe.x_coords) == 33

    # check bilinear form stuff
    # a, L = fe.lhs(swe.F), fe.rhs(swe.F)
    # assert swe.a == a
    # assert L == swe.L

    # check that topo is set OK
    H = swe.H.compute_vertex_values()
    H_true = np.zeros_like(swe.x_coords)
    for i in range(len(H_true)):
        BumpTopo(swe.bump_centre, swe.L).eval(H_true[i], swe.x_coords[i])

    np.testing.assert_allclose(H, H_true.flatten())

    # check that initial DOFs are set OK
    u_prev, h_prev = swe.get_vertex_values_prev()
    np.testing.assert_allclose(u_prev, 0.)
    np.testing.assert_allclose(
        h_prev,
        np.exp(-(2 * (swe.x_coords[:, 0] - 10))**2) / 40)

    # and run solve just to check that things work OK
    swe.solve()


def test_shallowone_init():
    # re-instantiate with diff setup BCs
    control = {"nx": 32, "dt": 0.02, "theta": 1.0, "simulation": "tidal_flow"}
    params = {"nu": 1.0}
    swe = ShallowOne(control, params)

    assert len(swe.x_dofs_u) == 65
    assert len(swe.x_dofs_h) == 33
    assert len(swe.x_coords) == 33

    # regression test for BC
    assert swe.tidal_bc(1.) == 4.23079749012345e-8

    # check that topography is all g
    def topo(x):
        return 50.5 - 40 * x / swe.L - 10 * np.sin(np.pi *
                                                   (4 * x / swe.L - 0.5))

    assert_allclose(
        swe.H.vector().get_local(),
        topo(swe.H.function_space().tabulate_dof_coordinates())[:, 0])

    # now verify dam break scenario
    control["simulation"] = "dam_break"
    swe = ShallowOne(control, params)

    assert len(swe.x_dofs_u) == 65
    assert len(swe.x_dofs_h) == 33
    assert len(swe.x_coords) == 33

    # check that solving actually does something
    # (measured in the L2 norm)
    u_prev = np.copy(swe.du_prev.vector().get_local())
    swe.solve(0. + swe.dt)
    u_next = np.copy(swe.du.vector().get_local())
    assert np.linalg.norm(u_prev - u_next) >= 1e-6


def test_shallowone_vertices():
    # check that function computations look reasonable
    control = {"nx": 32, "dt": 0.02, "theta": 1.0, "simulation": "tidal_flow"}
    params = {"nu": 1.0}
    swe = ShallowOne(control, params)

    du_true = fe.Expression(("t * sin(x[0])", "t * cos(x[0])"), t=2., degree=4)
    swe.du.interpolate(du_true)
    u, h = swe.get_vertex_values()

    assert_allclose(u, 2 * np.sin(swe.x_coords.flatten()))
    assert_allclose(h, 2 * np.cos(swe.x_coords.flatten()))
