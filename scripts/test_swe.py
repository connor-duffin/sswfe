import os
import pytest
import numpy as np
import fenics as fe

from numpy.testing import assert_allclose
from swe import ShallowOne, ShallowTwo


# @pytest.fixture
# def swe_1d():
#     control = {"nx": 32, "dt": 0.02}
#     return ShallowOne(control)


@pytest.fixture
def swe_2d():
    mesh = fe.UnitSquareMesh(32, 32)
    control = {"dt": 0.01,
               "theta": 1.,
               "simulation": "mms",
               "integrate_continuity_by_parts": False}
    return ShallowTwo(mesh, control)


def test_shallowone_init():
    # re-instantiate with diff setup BCs
    control = {"nx": 32, "dt": 0.02, "simulation": "tidal_flow"}
    swe = ShallowOne(control)

    assert len(swe.x_dofs_u) == 65
    assert len(swe.x_dofs_h) == 33
    assert len(swe.x_coords) == 33

    # regression test for BC
    assert swe.tidal_bc(1.) == 4.23079749012345e-8

    # check that topography is all g
    def topo(x):
        return 50.5 - 40 * x / swe.L - 10 * np.sin(np.pi * (4 * x / swe.L - 0.5))

    assert_allclose(swe.H.vector().get_local(),
                    topo(swe.H.function_space().tabulate_dof_coordinates())[:, 0])

    # now verify dam break scenario
    control["simulation"] = "dam_break"
    swe = ShallowOne(control)

    assert len(swe.x_dofs_u) == 65
    assert len(swe.x_dofs_h) == 33
    assert len(swe.x_coords) == 33

    # check that solving actually does something
    # (measured in the L2 norm)
    u_prev = np.copy(swe.du_prev.vector().get_local())
    swe.solve(0. + swe.dt)
    u_next = np.copy(swe.du.vector().get_local())
    assert np.linalg.norm(u_prev - u_next) >= 1e-6


def test_shallowtwo_init():
    mesh = fe.UnitSquareMesh(32, 32)
    control = {"dt": 0.01,
               "theta": 1.,
               "simulation": "mms",
               "integrate_continuity_by_parts": False}
    swe = ShallowTwo(mesh, control)

    assert swe.nu == 0.6
    assert swe.C == 0.0025
    assert swe.H == 50.

    assert_allclose(swe.dx, np.sqrt(2 * (1/32)**2))
    assert len(swe.du_vertices) == 3 * len(swe.x_coords)
    assert len(swe.bcs) == 2

    assert len(swe.F.integrals()) == 10

    assert swe.solver.parameters["snes_solver"]["linear_solver"] == "gmres"
    assert swe.solver.parameters["snes_solver"]["preconditioner"] == "fieldsplit"

    swe = ShallowTwo(mesh, control)
    control = {"dt": 0.01,
               "theta": 1.,
               "simulation": "laminar",
               "integrate_continuity_by_parts": True}
    swe = ShallowTwo(mesh, control)
    assert len(swe.bcs) == 2
    assert len(swe.F.integrals()) == 11

    for forcing in [swe.f_u, swe.f_h]:
        f = np.copy(forcing.vector().get_local())
        assert_allclose(f, np.zeros_like(f))


def test_shallowtwo_solve(swe_2d):
    u_prev = np.copy(swe_2d.du_prev.vector().get_local())
    swe_2d.solve()
    u_next = np.copy(swe_2d.du.vector().get_local())
    assert np.linalg.norm(u_prev - u_next) >= 1e-6


def test_shallowtwo_ss(swe_2d):
    assert swe_2d.steady_state(swe_2d.du, swe_2d.du_prev)
    swe_2d.set_prev_vector(np.ones_like(swe_2d.du_prev.vector().get_local()))
    assert not swe_2d.steady_state(swe_2d.du, swe_2d.du_prev)


def test_shallowtwo_save(swe_2d):
    f = "temp.h5"
    swe_2d.setup_checkpoint(f)
    du_true = fe.Expression(("t*x[0]", "t * x[1]", "t * x[0] * x[1]"), t=0., degree=2)

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
