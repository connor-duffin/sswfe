import logging

from argparse import ArgumentParser
import numpy as np
import fenics as fe
from swe import ShallowTwo

logging.basicConfig(level=logging.INFO)

comm = fe.MPI.comm_world
rank = comm.rank


def format_output_file(output_dir, BDF2=False, theta=1., nu_t=0.):
    output_file = output_dir + f"swe-channel-laplacian-nu_t-{nu_t:.4e}-"

    if BDF2:
        output_file += "bdf2.h5"
    else:
        output_file += f"theta-{theta:.2f}.h5"

    return output_file


def run_swe_cylinder(output_file, BDF2=False, theta=1., nu_t=0.):
    swe = ShallowTwo(mesh="mesh/channel-piggott.xdmf",
                     control={
                         "dt": 1e-3,
                         "theta": theta,
                         "simulation": "cylinder",
                         "integrate_continuity_by_parts": False,
                         "laplacian": True,
                         "les": False
                     })
    swe.nu += nu_t  # constant eddy viscosity
    nt = 5001  # run up to time t = 5 s  (failure occurs at ~1.2)
    nt_thin = 1  # store every iteration
    t = 0.

    logging.info("storing solution at %s", output_file)
    swe.setup_checkpoint(output_file)
    attrs = swe.checkpoint.attributes("/")
    attrs["nu_t"] = nu_t
    if BDF2:
        attrs["scheme"] = "BDF2"
    else:
        attrs["scheme"] = "theta"
        attrs["theta"] = theta

    # logging.info("loading solution from %s, NOT STORING", output_file)
    # checkpoint = fe.HDF5File(swe.mesh.mpi_comm(), output_file, "r")
    # vec_init = "du/vector_12"
    # checkpoint.read(swe.du_prev_prev, vec_init)
    # checkpoint.read(swe.du_prev, vec_init)
    # checkpoint.read(swe.du, vec_init)
    # t = checkpoint.attributes(vec_init)["timestamp"]
    # checkpoint.close()

    # solves for du_prev, from du_prev_prev
    if BDF2:
        F, J = swe.setup_form(swe.du_prev, swe.du_prev_prev, bdf2=False)
        bcs, F = swe.setup_bcs(F)
        solver = swe.setup_solver(F, swe.du_prev, bcs, J)
        solver.solve()
        t += swe.dt

    F, J = swe.setup_form(swe.du, swe.du_prev, swe.du_prev_prev, bdf2=BDF2)
    bcs, F = swe.setup_bcs(F)
    swe.solver = swe.setup_solver(F, swe.du, bcs, J)

    for i in range(nt):
        if i % nt_thin == 0:
            if rank == 0:
                logging.info(
                    "storing solution at time %.5f" +
                    " iteration %d of %d complete", t, i + 1, nt)
            swe.checkpoint_save(t)
        try:
            swe.solve(bdf2=BDF2)
            t += swe.dt
        except RuntimeError:
            print(f"SOLVER FAILED AT TIME {t:.5f}")
            break

    swe.checkpoint.close()


# logging.info("Running default LES simulation")
# run_swe_cylinder(output_file="outputs/swe-channel-les-theta-0.60.h5",
#                  BDF2=False, theta=0.6, nu_t=0.)

# parameters
# schemes = [{"BDF2": True, "theta": 1.0}]
# nus = [8e-5, 9e-5, 1e-4]
# BDF2 = [True, False]

schemes = []
thetas = [0.5, 0.6, 1.]
for theta in thetas:
    schemes.append({"BDF2": False, "theta": theta})

kwargs_all = []
nus = np.linspace(0, 1e-4, 11).tolist()
for scheme in schemes:
    for nu in nus:
        kwargs = {**scheme, "nu_t": nu}
        kwargs_all.append({
            "output_file": format_output_file("outputs/", **kwargs),
            **kwargs})

for kwargs in kwargs_all:
    print(kwargs)
    run_swe_cylinder(**kwargs)
