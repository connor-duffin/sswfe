import logging

from argparse import ArgumentParser
import numpy as np
import fenics as fe

from tqdm import tqdm

from ns_2d import NSTwo, NSTwoSplit

comm = fe.MPI.comm_world
rank = comm.rank


def run_ns_feat(output_file, theta=1.):
    mesh = "mesh/featflow-2d-3-benchmark.xdmf"
    ns = NSTwo(mesh, dict(dt=1/1600, theta=theta, setup="feat"))
    ns.setup_form()

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
    attrs["nt"] = nt

    for i in tqdm(range(nt)):
        ns.checkpoint_save(t)
        if i % 1000 == 0:
            logging.info(f"Solver finished time {t:.5f}")
        try:
            t += ns.dt
            ns.solve(t)
        except RuntimeError:
            logging.info(f"Solver failed at time {t:.5f}")
            break

    ns.checkpoint.close()


def run_ns_split_feat(output_file):
    mesh = "mesh/featflow-2d-3-benchmark.xdmf"
    ns = NSTwoSplit(mesh, dict(dt=1/1000, setup="feat"))
    ns.setup_form()

    # timestepping setup
    t = 0.
    t_final = 8.
    nt = int((t_final - t) / ns.dt)
    logging.info("running for %d timesteps", nt)

    # output setup
    logging.info("storing solution at %s", output_file)
    ns.setup_checkpoint(output_file)
    attrs = ns.checkpoint.attributes("/")
    attrs["scheme"] = "split"
    attrs["nt"] = nt

    for i in tqdm(range(nt)):
        if i % 1000 == 0:
            logging.info(f"Solver finished time {t:.5f}")
        try:
            t += ns.dt
            ns.solve(t)

            if np.any(np.isnan(ns.u.vector().get_local())):
                raise RuntimeError
        except RuntimeError:
            logging.info(f"Solver failed at time {t:.5f}")
            break

        ns.checkpoint_save(t)

    ns.checkpoint.close()


logging.info("Running FEAT 2D-3 benchmark")
run_ns_feat(output_file="outputs/ns-feat-2d-3-theta-0.50.h5", theta=0.5)
# run_ns_split_feat(output_file="outputs/ns-feat-2d-3-split.h5")
