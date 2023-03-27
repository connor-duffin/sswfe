import logging

from argparse import ArgumentParser
import numpy as np
import fenics as fe

from tqdm import tqdm

from ns_2d import NSSplit, NSSemiImplicit

comm = fe.MPI.comm_world
rank = comm.rank


def run_ns_feat(output_file):
    mesh = "mesh/featflow-2d-3-benchmark.xdmf"
    ns = NSSemiImplicit(mesh, dict(dt=1/1600))
    ns.setup_form()

    # timestepping setup
    t = 0.
    t_final = 1.
    nt = int((t_final - t) / ns.dt)
    logging.info("running for %d timesteps", nt)

    # output setup
    logging.info("storing solution at %s", output_file)
    ns.setup_checkpoint(output_file)
    attrs = ns.checkpoint.attributes("/")
    attrs["scheme"] = "simo-armero"
    attrs["nt"] = nt

    for i in tqdm(range(nt)):
        ns.checkpoint_save(t)
        if i % 1000 == 0:
            logging.info(f"Solver finished time {t:.5f}")
        try:
            ns.solve(krylov=True)
        except RuntimeError:
            logging.info(f"Solver failed at time {t:.5f}")
            ns.checkpoint.close()
            raise

    ns.checkpoint.close()


logging.info("Running FEAT 2D-3 benchmark")
run_ns_feat(output_file="outputs/ns-testing.h5")
