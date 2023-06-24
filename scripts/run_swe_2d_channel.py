import logging

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from swe_2d import ShallowTwo
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(relativeCreated)d ms - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)


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


mesh_file = "mesh/branson.xdmf"
output_file = "outputs/branson-swe-high-re.h5"
params = dict(nu=5e-5, H=0.053, C=0.)
control = {"dt": 1e-2,
           "theta": 0.51,
           "simulation": "cylinder",
           "laplacian": True,
           "les": False,
           "integrate_continuity_by_parts": True}

swe = ShallowTwo(mesh=mesh_file, params=params, control=control)
F, J, bcs = swe.setup_form(swe.du, swe.du_prev)
swe.solver = swe.setup_solver(F, swe.du, bcs, J)
n_dofs = len(swe.du.vector().get_local())

# streamline immediately past the obstacle
x_eval = [[1.15, 0.5]]

# load datasets from checkpoints
write_checkpoint = True
load_from_checkpoint = False

if load_from_checkpoint:
    with fe.HDF5File(swe.mesh.mpi_comm(), output_file, "r") as checkpoint_file:
        i = 0
        while checkpoint_file.has_dataset(f"/du/vector_{i}"):
            i += 1

        vec_name = f"/du/vector_{i - 1}"
        checkpoint_file.read(swe.du_prev, vec_name)
        checkpoint_file.read(swe.du, vec_name)
        t = checkpoint_file.attributes(vec_name)["timestamp"]
        logging.info(f"Loading checkpoint into current state at time t = {t:.4e} (index {i})")
else:
    t = 0.

t_final = 5 * 60.
nt = np.int32(np.round((t_final - t) / control["dt"]))
thin = 10
nt_thin = len([i for i in range(nt) if i % thin == 0])
logging.info(
    f"running simulation up to time {t_final:.5f}, from {t:.5f}")

# append to output_file
if write_checkpoint:
    logging.info(f"Writing checkpoint to disk at {output_file}")
    swe.setup_checkpoint(output_file)
    swe.checkpoint.write(swe.mesh, "mesh")

t_out = np.zeros((nt_thin, ))
u_streamline_out = np.zeros((nt_thin, ))
v_streamline_out = np.zeros((nt_thin, ))

i_save = 0
for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.inlet_velocity.t = t
        swe.u_in.interpolate(swe.inlet_velocity)
        swe.solve()
    except RuntimeError:
        print(f"failed at iteration {i}, time {t:.5f}, exiting")
        break

    if i % thin == 0 and write_checkpoint:
        swe.checkpoint_save(t)

if write_checkpoint:
    swe.checkpoint_close()
