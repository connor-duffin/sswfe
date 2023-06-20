import h5py
import logging

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from swe_2d import ShallowTwo
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(relativeCreated)d ms - %(name)s - %(levelname)s - %(message)s',
                    level=logging.WARN)


def evaluate_function(u, x, mesh):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(fe.Point(*x))
    u_eval = u(*x) if distance < 1e-14 else None

    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array([y for y in computed_u if y is not None], dtype=np.double)
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)
        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u


# parser = ArgumentParser()
# parser.add_argument("--integrate_continuity_by_parts", action="store_true")
# parser.add_argument("--cylinder", action="store_true")
# parser.add_argument("mesh_file", type=str)
# parser.add_argument("output_file", type=str)
# args = parser.parse_args()

mesh_file = "mesh/swe-square-cylinder-test.xdmf"
simulation = "cylinder"

logging.info("using %s as the simulation settings", simulation)
control = {"dt": 5e-4,
           "theta": 1.,
           "simulation": simulation,
           "laplacian": True,
           "les": False,
           "integrate_continuity_by_parts": True}

swe = ShallowTwo(mesh=mesh_file, control=control)

# try vanilla integrator
F, J, bcs = swe.setup_form(swe.du, swe.du_prev)
swe.solver = swe.setup_solver(F, swe.du, bcs, J)

t_final = 15.
nt = np.int32(np.round(t_final / control["dt"]))

# HACK: because I am lazy at the moment
n_dofs = len(swe.du.vector().get_local())
logging.info("running simulation up to time %f.5f", nt)

thin = 10
n_eval = 100
nt_thin = len([i for i in range(nt) if i % thin == 0])
x_eval = [[x, 7 * 0.04] for x in np.linspace(6 * 0.04, 1, num=n_eval)]

t_out = np.zeros((nt_thin, ))
u_out = np.zeros((nt_thin, n_eval))
print(u_out.shape)

t = 0.
i_save = 0
for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.solve()
    except RuntimeError:
        print(f"failed at iteration {i}, time {t:.5f} exiting")
        break

    if i % thin == 0:
        u, h = swe.du.split()
        for j, x in enumerate(x_eval):
            u_curr = evaluate_function(u, x, swe.mesh)[0]
            u_out[i_save, j] = u_curr

        t_out[i_save] = t
        i_save += 1

if swe.mesh.mpi_comm().rank == 0:
    fig, ax = plt.subplots(1, figsize=(5, 3), constrained_layout=True)
    ax.plot(np.asarray(x_eval)[:, 0] / 0.04,
            np.mean(u_out / 0.535, axis=0))
    ax.set_xlabel(r"$x / d$")
    ax.set_ylabel(r"$u / U$")
    plt.savefig("figures/test-velocity.pdf")
