import h5py
import logging

import fenics as fe
import numpy as np

from argparse import ArgumentParser
from swfe.swe_2d import ShallowTwo
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
parser.add_argument('--write_checkpoint', action='store_true')
parser.add_argument('--load_from_checkpoint', action='store_true')
parser.add_argument('--checkpoint_file', type=str)
args = parser.parse_args()

MESH_FILE = 'mesh/branson-mesh-nondim-coarse.xdmf'
OUTPUT_FILE = 'outputs/branson-coarse-dampened-testing.h5'

# physical settings
period = 120.
nu = 1e-5
g = 9.8

# reference values
u_ref = 0.01  # cm/s
length_ref = 0.1  # cylinder
time_ref = length_ref / u_ref
H_ref = length_ref
Re = u_ref * length_ref / nu

params = dict(nu=1 / Re, g=g * H_ref / u_ref**2, C=0., H=0.053 / H_ref,
              length=20., width=5., cylinder_centre=(10, 2.5), cylinder_radius=0.5,
              u_inflow=0.004 / u_ref, inflow_period=period / time_ref)
control = dict(dt=0.4, simulation='cylinder', use_imex=False, use_les=False, theta=0.5)

swe = ShallowTwo(mesh=MESH_FILE, params=params, control=control, comm=comm)
swe.setup_form()
swe.setup_solver(use_ksp=False)

# load datasets from checkpoints (if set)
if args.load_from_checkpoint:
    with fe.HDF5File(comm, args.checkpoint_file, 'r') as checkpoint_file:
        i = 0
        while checkpoint_file.has_dataset(f'/du/vector_{i}'):
            i += 1

        vec_name = f'/du/vector_{i - 1}'
        checkpoint_file.read(swe.du_prev, vec_name)
        checkpoint_file.read(swe.du, vec_name)
        t = checkpoint_file.attributes(vec_name)['timestamp']
        logging.info(f'Loading checkpoint into current state at time t = {t:.4e} (index {i})')
else:
    t = 0.

t_final = 5. * params['inflow_period']
nt = np.int32(np.round((t_final - t) / control['dt']))
logging.info(f'running simulation up to time {t_final:.5f}, from {t:.5f}')

# set t_thin to observation interval
t_thin = 0.4
thin = np.int32(np.round(t_thin // control['dt']))
nt_thin = len([i for i in range(nt) if i % thin == 0])
logging.info(f'storing every {t_thin:.5f} tu (every {nt_thin:d} timesteps)')

# store attributes as needed (write attributes to root of file)
if args.write_checkpoint:
    logging.info(f'Writing checkpoint to disk at {args.checkpoint_file}')
    swe.setup_checkpoint(args.checkpoint_file)
    swe.checkpoint.write(swe.mesh, 'mesh')

    attrs = swe.checkpoint.attributes('/')
    metadata = {**params, **control}
    for name, val in metadata.items():
        attrs[name] = val

# for output storage
n_dofs = len(swe.W.tabulate_dof_coordinates())
t_out = np.zeros((nt_thin))
du_out = np.zeros((nt_thin, n_dofs))

i_save = 0
for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.inlet_velocity.t = t
        swe.solve()
    except RuntimeError as e:
        raise e
        print(f'failed at iteration {i}, time {t:.5f}, exiting')
        break

    # checkpoint (optional: flagged at runtime)
    if i % thin == 0 and args.write_checkpoint:
        swe.checkpoint_save(t)

    du_out[i, :] = swe.du.vector().get_local()
    swe.set_prev()

if args.write_checkpoint:
    swe.checkpoint_close()

with h5py.File(OUTPUT_FILE, 'w') as f:
    metadata = {**params, **control}
    for name, val in metadata.items():
        f.attrs.create(name, val)

    f.create_dataset('t', data=t_out)
    f.create_dataset('du', data=du_out)
