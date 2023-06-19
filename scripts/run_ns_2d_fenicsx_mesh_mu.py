import os
import h5py
import tqdm

import matplotlib.pyplot as plt
import numpy as np

from dolfinx.fem import Function
from dolfinx.geometry import (BoundingBoxTree, compute_collisions,
                              compute_colliding_cells)

from itertools import product
from multiprocessing import Pool
from ns_2d_fenicsx import NSSplit

from mpi4py import MPI
from petsc4py import PETSc


mesh_comm = MPI.COMM_WORLD
model_rank = 0


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]),
                          dtype=PETSc.ScalarType)
        values[0] = 2.5e-2 * np.sin(2 * np.pi * self.t / 10)
        return values


def run_ns_2d(mesh_file, mu):
    t = 0
    T = 60.
    dt = 1e-2
    nt = int(T / dt)
    thin = 100
    nt_save = nt // thin

    if nt % thin >= 1:
        nt_save += 1

    params = dict(mu=mu, rho=1000)
    ns = NSSplit(mesh_file, dt, params)

    # set inlet velocity
    inlet_velocity = InletVelocity(0.)
    ns.u_inlet = Function(ns.V)
    ns.u_inlet.interpolate(inlet_velocity)

    # then set form
    ns.setup_form()

    x_eval = [2.88, 0.9, 0]
    tree = BoundingBoxTree(ns.msh, dim=2)
    cell_candidates = compute_collisions(tree, x_eval)
    colliding_cells = compute_colliding_cells(ns.msh, cell_candidates, x_eval)

    if mesh_comm.rank == 0:
        t_test = np.zeros(nt, dtype=PETSc.ScalarType)
        u_test = np.zeros(nt, dtype=PETSc.ScalarType)
        v_test = np.zeros(nt, dtype=PETSc.ScalarType)

    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=nt)
    for i in range(nt):
        progress.update(1)

        t += dt
        inlet_velocity.t = t
        ns.u_inlet.interpolate(inlet_velocity)
        ns.solve(t)

        # check for NaNs
        if (np.any(np.isnan(ns.u_.vector.array)) and np.any(np.isnan(ns.p_.vector.array))):
            print(f"Simulation failed at t = {t:.4e}")
            break

        # assemble the pressure difference
        u_flow = None
        if len(colliding_cells) > 0:
            u_flow = ns.u_.eval(x_eval, colliding_cells[:1])

        u_flow = mesh_comm.gather(u_flow, root=0)
        if mesh_comm.rank == 0:
            t_test[i] = t

            for u in u_flow:
                if u is not None:
                    u_test[i] = u[0]
                    v_test[i] = u[1]
                    break

    if mesh_comm.rank == 0:
        print(u_test, v_test)

        fig, axs = plt.subplots(
            1, 2, constrained_layout=True, figsize=(8, 3))
        axs[0].plot(t_test, u_test)
        axs[0].set_ylabel(r"$u$")
        axs[1].plot(t_test, v_test)
        axs[1].set_ylabel(r"$v$")
        for ax in axs:
            ax.set_xlabel(r"$t$")
        plt.savefig("figures/u-observed.png")
        plt.close()


run_ns_2d(mesh_file="mesh/branson-3.msh", mu=0.1)
