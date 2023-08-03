import pytest

import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

from swfe.swe_les import LES


def les_smagorinsky_eddy_viscosity():
    """
    Smagorinsky Large Eddy Simulation, done in Fenics.

    Implementation uses the same code as in the LES module in
    firedrake-fluids: https://github.com/firedrakeproject/firedrake-fluids,
    and is inspired by similar in the LES module, given in Oasis:
    https://github.com/mikaem/Oasis
    """
    errors = []

    for n in [2, 4, 8, 16, 32]:
        mesh = fe.UnitSquareMesh(n, n)
        smagorinsky_coefficient = 2.0
        filter_width = fe.CellVolume(mesh)**(1.0 / 2.0
                                             )  # Square root of element area

        fs_exact = fe.FunctionSpace(mesh, "CG", 3)
        fs = fe.FunctionSpace(mesh, "CG", 1)
        vfs = fe.VectorFunctionSpace(mesh, "CG", 1)

        u = fe.interpolate(fe.Expression(('sin(x[0])', 'cos(x[0])'), degree=8),
                           vfs)

        exact_solution = fe.project(
            fe.Expression(
                'pow(%f, 2) * sqrt(2.0*cos(x[0])*cos(x[0]) + 0.5*sin(x[0])*sin(x[0]) + 0.5*sin(x[0])*sin(x[0]))'
                % smagorinsky_coefficient,
                degree=8), fs_exact)

        # les = LES(swe.mesh, V, u, 1.0, 0.164)
        les = LES(mesh, fs, u, 1.0, (smagorinsky_coefficient / filter_width))
        les.solve()

        eddy_viscosity = les.eddy_viscosity
        errors.append(
            fe.sqrt(
                fe.assemble(
                    fe.inner(eddy_viscosity - exact_solution,
                             eddy_viscosity - exact_solution) * fe.dx)))

    return errors


def test_les_smagorinsky_eddy_viscosity():
    """ Unit test for LES simulation."""
    errors = np.array(les_smagorinsky_eddy_viscosity())
    rate = np.array(
        [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)])
    print(rate)
    assert (rate > 1.45).all()
