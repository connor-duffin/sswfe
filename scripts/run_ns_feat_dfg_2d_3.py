import logging

from argparse import ArgumentParser
import numpy as np
import fenics as fe
from ns_2d import NSTwo

logging.basicConfig(level=logging.INFO)

comm = fe.MPI.comm_world
rank = comm.rank


def run_ns_feat(output_file, theta=1.):
    mesh = "mesh/featflow-2d-3-benchmark.xdmf"
    ns = NSTwo(mesh, dict(dt=1/1600, theta=0.5))

    F, J = ns.setup_form(ns.du, ns.du_prev)
    bcs, F = ns.setup_bcs(F)
    solver = ns.setup_solver(F, ns.du, bcs, J)

    # timestepping setup
    t = 0.
    t_final = 8.
    nt = int((t_final - t) / ns.dt)
    logging.info("running for %d timesteps", nt)

    # output setup
    logging.info("storing solution at %s", output_file)
    ns.setup_checkpoint(output_file)
    attrs = ns.checkpoint.attributes("/")
    attrs["scheme"] = "theta"
    attrs["theta"] = theta

    for i in range(nt):
        if rank == 0:
            logging.info(
                "storing solution at time %.5f" +
                " iteration %d of %d complete", t, i + 1, nt)
        ns.checkpoint_save(t)
        try:
            t += ns.dt
            ns.u_in.t = t
            solver.solve()
        except RuntimeError:
            print(f"Solver failed at time {t:.5f}")
            break

    ns.checkpoint.close()


logging.info("Running FEAT 2D-3 benchmark")
run_ns_feat(output_file="outputs/ns-feat-2d-3-theta-0.60.h5", theta=0.6)
