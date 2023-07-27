import h5py
import logging

import fenics as fe
import numpy as np

from argparse import ArgumentParser
from swe_2d import ShallowTwo
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s - %(relativeCreated)d ms - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# register MPI communicator
comm = fe.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# set inputs etc
parser = ArgumentParser()
parser.add_argument("--mesh_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--write_checkpoint", action="store_true")
parser.add_argument("--load_from_checkpoint", action="store_true")
parser.add_argument("--checkpoint_file", type=str)
args = parser.parse_args()

# run 8 from Paul's JFM paper
# physical settings
period = 120.
nu = 1e-6
g = 9.8

# reference values
u_ref = 0.01  # cm/s
length_ref = 0.1  # cylinder
time_ref = length_ref / u_ref
H_ref = length_ref

# compute reynolds number
Re = u_ref * length_ref / nu

params = dict(
    nu=1 / Re,
    g=g * H_ref / u_ref**2,
    C=0.,
    H=0.053 / H_ref,
    u_inflow=0.004 / u_ref,
    inflow_period=period / time_ref)
control = dict(
    dt=5e-2,
    theta=0.5,
    simulation="cylinder",
    use_imex=False,
    use_les=False)

# mesh = fe.RectangleMesh(comm, fe.Point(0, 0,), fe.Point(10, 5), 32, 16)
swe = ShallowTwo(mesh=args.mesh_file,
                 params=params,
                 control=control,
                 comm=comm)
swe.setup_form()
swe.setup_solver(use_ksp=False)

# load datasets from checkpoints (if set)
if args.load_from_checkpoint:
    with fe.HDF5File(comm, args.checkpoint_file, "r") as checkpoint_file:
        i = 0
        while checkpoint_file.has_dataset(f"/du/vector_{i}"):
            i += 1

        vec_name = f"/du/vector_{i - 1}"
        checkpoint_file.read(swe.du_prev, vec_name)
        checkpoint_file.read(swe.du, vec_name)
        t = checkpoint_file.attributes(vec_name)["timestamp"]
        logging.info(
            f"Loading checkpoint into current state at time t = {t:.4e} (index {i})")
else:
    t = 0.

t_final = 5. * params["inflow_period"]
nt = np.int32(np.round((t_final - t) / control["dt"]))

t_thin = 0.1
thin = np.int32(np.round(t_thin / control["dt"]))
nt_thin = len([i for i in range(nt) if i % thin == 0])

logging.info(
    f"running simulation up to time {t_final:.5f}, from {t:.5f}")

# append to checkpoint_file
if args.write_checkpoint:
    logging.info(f"Writing checkpoint to disk at {args.checkpoint_file}")
    swe.setup_checkpoint(args.checkpoint_file)
    swe.checkpoint.write(swe.mesh, "mesh")

    # store attributes as needed (write attributes to root of file)
    attrs = swe.checkpoint.attributes("/")
    metadata = {**params, **control}
    for name, val in metadata.items():
        attrs[name] = val

# coordinates across all
mesh_coords = swe.mesh.coordinates().copy()
n_coords = np.array(comm.allgather(len(mesh_coords)))
coords_displacements = np.cumsum(n_coords) - n_coords[0]
n_coords_all = np.sum(n_coords)

# vertex values across all
du_vertices = swe.du.compute_vertex_values()
n_vertices = np.array(comm.allgather(len(du_vertices)))
n_vertices_all = np.sum(n_vertices)

# check that logic is correct across processors
np.testing.assert_allclose(n_vertices / 3, n_coords)
np.testing.assert_allclose(n_vertices_all / 3, n_coords_all)

gdim = 2
if rank == 0:
    coords_recvbuf = np.zeros((n_coords_all, gdim), dtype=np.float64)
    u_recvbuf = np.zeros((n_coords_all, ), dtype=np.float64)
    v_recvbuf = np.zeros((n_coords_all, ), dtype=np.float64)
    h_recvbuf = np.zeros((n_coords_all, ), dtype=np.float64)

    # for output storage
    t_out = np.zeros((nt_thin))
    u_out = np.zeros((nt_thin, n_coords_all))
    v_out = np.zeros((nt_thin, n_coords_all))
    h_out = np.zeros((nt_thin, n_coords_all))
else:
    coords_recvbuf = None
    u_recvbuf = None
    v_recvbuf = None
    h_recvbuf = None

    t_out = None
    u_out = None
    v_out = None

# gather coordinate array
comm.Gatherv(sendbuf=mesh_coords, recvbuf=(coords_recvbuf, gdim * n_coords), root=0)

i_save = 0
for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.inlet_velocity.t = t
        swe.solve()
    except RuntimeError as e:
        raise e
        print(f"failed at iteration {i}, time {t:.5f}, exiting")
        break

    # checkpoint (optional: flagged at runtime)
    if i % thin == 0 and args.write_checkpoint:
        swe.checkpoint_save(t)

    # output (always stored)
    if i % thin == 0:
        du_vertices = swe.du.compute_vertex_values()
        u_vertices = du_vertices[:n_coords[rank]]
        v_vertices = du_vertices[n_coords[rank]:(2 * n_coords[rank])]
        h_vertices = du_vertices[(2 * n_coords[rank]):]

        # gather appropriate data arrays
        comm.Gatherv(sendbuf=u_vertices, recvbuf=(u_recvbuf, n_coords), root=0)
        comm.Gatherv(sendbuf=v_vertices, recvbuf=(v_recvbuf, n_coords), root=0)
        comm.Gatherv(sendbuf=h_vertices, recvbuf=(h_recvbuf, n_coords), root=0)

        # store outputs
        if rank == 0:
            t_out[i_save] = t
            u_out[i_save, :] = u_recvbuf
            v_out[i_save, :] = v_recvbuf
            h_out[i_save, :] = h_recvbuf

        i_save += 1

    swe.set_prev()

if args.write_checkpoint:
    swe.checkpoint_close()

# store outputs after MPI has gathered each
if rank == 0:
    with h5py.File(args.output_file, "w") as f:
        # tag with appropriate metadata
        metadata = {**params, **control}
        for name, val in metadata.items():
            f.attrs.create(name, val)

        f.create_dataset("t", data=t_out)
        f.create_dataset("x", data=coords_recvbuf)

        for name, out in zip(["u", "v", "h"], [u_out, v_out, h_out]):
            f.create_dataset(name, data=out)
