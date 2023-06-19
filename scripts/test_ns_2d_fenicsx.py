from ns_2d_fenicsx import NSSplit


def test_ns():
    mesh_file = "mesh/branson-refined.msh"
    ns = NSSplit(mesh_file, 1e-4, dict(mu=1e-3, rho=1))

    # check derived function spaces are OK
    assert ns.V_dof_coordinates.shape[0] > 0
    assert ns.Q_dof_coordinates.shape[0] > 0
    assert ns.fdim == 1


