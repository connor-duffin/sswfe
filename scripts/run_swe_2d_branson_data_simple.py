import logging
import time

import fenics as fe
import numpy as np
import xarray as xr

from argparse import ArgumentParser
from mpi4py.MPI import COMM_WORLD
from swfe.swe_2d import ShallowTwoEnsembleVanilla
from scipy.sparse import vstack, diags
from statfenics.utils import build_observation_operator
from tqdm import tqdm

comm = fe.MPI.comm_self
rank = COMM_WORLD.Get_rank()
size = COMM_WORLD.Get_size()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('--compute_post', action='store_true')
args = parser.parse_args()

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

# dimensionless parameters
params = dict(nu=1 / Re, g=g * length_ref / u_ref**2, C=0., H=0.053 / H_ref,
              length=20., width=5., cylinder_centre=(10, 2.5), cylinder_radius=0.5,
              u_inflow=0.004 / u_ref, inflow_period=period / time_ref)

L = params['length']
B = params['width']
r = params['cylinder_radius']

dt = 0.2
n_ens = 192

obs_skip = 10
error_process_noise = 1e-4
sigma_y = 2e-2

mesh_file = 'mesh/branson-mesh-nondim.xdmf'
data_file = 'data/Run08_G0_dt4.0_de1.0_ne5_velocity.nc'
control = dict(dt=dt, theta=0.5, simulation='laminar', use_imex=False, use_les=False)

# swe = ShallowTwoFilter(MESH_FILE, params, control, comm)
# swe = ShallowTwoFilterPETSc(MESH_FILE, params, control, comm)
swe = ShallowTwoEnsembleVanilla(mesh_file, params, control, comm)

swe.setup_form()
swe.setup_solver(use_ksp=False)
t = 0.
t_final = 5 * period / time_ref
nt = np.int32(np.round(t_final / control['dt']))

# #######################################
# step 1: read in and preprocess the data
ds = xr.open_dataset(data_file)

# first we get the locations of the observation points wrt grid
ds_zero = ds.isel(time=0)
ds_zero_missing = ds_zero.where(ds_zero.isnull(), drop=True)

# scale to our coordinates
x = ds_zero.coords['x'].to_numpy() / length_ref
y = ds_zero.coords['y'].to_numpy() / length_ref

x_missing = ds_zero_missing.coords['x'].to_numpy() / length_ref
y_missing = ds_zero_missing.coords['y'].to_numpy() / length_ref

# compute global `x` coordinates
x_displacement = np.min(x[x > np.max(x_missing)])
x_center_displacement = r - x_displacement
x_global_displacement = L / 2 + x_center_displacement
x_global = x + x_global_displacement

# compute global `y` coordinates
y_mid = y[len(y) // 2]
y_global = B / 2 + (y - y_mid)

# and create meshgrid according to our current setup
x_mg, y_mg = np.meshgrid(x_global, y_global, indexing='ij')

# then we average the data spatially
u_spatial_averaged = np.mean(ds.U, axis=(1, 2, 3))
v_spatial_averaged = np.mean(ds.V, axis=(1, 2, 3))

# start from the flow being nearest to 0.
idx_start = np.argmin(np.abs(u_spatial_averaged.values - 0.))
idx_end = len(u_spatial_averaged.values)

u_depth_averaged = np.mean(ds.U, axis=-1)
v_depth_averaged = np.mean(ds.V, axis=-1)

u_depth_averaged = u_depth_averaged.isel(time=slice(idx_start, idx_end))
v_depth_averaged = v_depth_averaged.isel(time=slice(idx_start, idx_end))

u_depth_averaged['time'].dtype == np.datetime64
u_depth_averaged = u_depth_averaged.assign_coords(
    dict(time_rel=((u_depth_averaged['time'] - u_depth_averaged['time'][0]) * 1e-9).astype(float)))
v_depth_averaged = v_depth_averaged.assign_coords(
    dict(time_rel=((v_depth_averaged['time'] - v_depth_averaged['time'][0]) * 1e-9).astype(float)))

# reference values: scaled as needed
# to start we get the data for the velocity fields
# and check for sensibility
ud = u_depth_averaged.to_numpy() / u_ref
vd = v_depth_averaged.to_numpy() / u_ref
nrows, ncols = vd[0, :, :].shape
nt_obs = ud.shape[0]

# mask values 'close' to the cylinder, avoiding values outside the mesh
c = np.sqrt((x_mg - L / 2)**2 + (y_mg - B / 2)**2)
mask = (c <= 0.6)

t_obs = u_depth_averaged.coords['time_rel'].to_numpy() / time_ref
x_obs = np.vstack((x_mg[~mask][::obs_skip], y_mg[~mask][::obs_skip])).T
u_obs = ud[:, ~mask][:, ::obs_skip]
v_obs = vd[:, ~mask][:, ::obs_skip]

# checking sanity
period_ref = period / time_ref
assert t_obs[-1] / period_ref >= 5.
assert x_obs.shape[1] == 2
assert np.all(np.logical_and(x_obs[:, 0] >= 10., x_obs[:, 0] <= 20.))
assert np.all(np.logical_and(x_obs[:, 1] >= 0., x_obs[:, 1] <= 5.))
nx_obs = x_obs.shape[0]
logger.info(f'Taking in {nx_obs} observations each time point')
assert np.all(~np.isnan(x_obs))
assert np.any(np.isnan(u_obs)) == False
assert np.any(np.isnan(v_obs)) == False

assert ud.shape == vd.shape
assert ud[0, :, :].shape == x_mg.shape
assert vd[0, :, :].shape == y_mg.shape

# read in data and set up assimilation routine
Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
H = vstack((Hu, Hv), format='csr')
logger.info('Observation operator shape is %s', H.shape)

# then setup filter (basically compute prior additive noise covariance)
stat_params = dict(error_process_noise=error_process_noise, n_ens=n_ens, H=H, sigma_y=sigma_y)
swe.setup_filter(stat_params)
np.testing.assert_allclose(H @ swe.mean, np.zeros((2 * nx_obs, )))

# checking things are sound
np.testing.assert_allclose(swe.mean[swe.u_dofs], 0.)
np.testing.assert_allclose(swe.mean[swe.v_dofs], 0.)
np.testing.assert_allclose(swe.mean[swe.h_dofs], 0.)
if args.compute_post:
    logger.info('Starting posterior computations...')
else:
    logger.info('Starting prior computations...')

# start after first data point
i_dat = 1
for i in range(nt):
    if rank == 0:
        t_start = time.time()

    t += swe.dt
    swe.inlet_velocity.t = t
    swe.prediction_step(t)
    if rank == 0:
        var = np.var(swe.du_ens, axis=0)
        logger.info('Var = %.5e', np.linalg.norm(var[swe.u_dofs]))

    if np.isclose(t_obs[i_dat], t):
        y_obs = np.concatenate((u_obs[i_dat, :], v_obs[i_dat, :]))

        if rank == 0:
            logger.info(f'Observation time reached: {t_obs[i_dat]:.4e}')
            logger.info(f'Data/obs. time: {t_obs[i_dat]:.5f}, ' + f'Simulation time {t:.5f}')

            theta = np.linalg.norm(y_obs - H @ swe.mean) / np.sqrt(len(y_obs))
            logger.info('t = %.5f, theta = %.5e', t, theta)

        if args.compute_post:
            swe.update_step(y_obs, compute_lml=False)

            if rank == 0:
                rmse = np.linalg.norm(y_obs - H @ swe.mean) / np.sqrt(len(y_obs))
                logger.info('t = %.5f, rmse = %.5e', t, rmse)

        i_dat += 1

    # progress onto next iteration
    if rank == 0:
        t_finish = time.time()
        logger.info(f'Iteration {i + 1} / {nt} took {t_finish - t_start:.5f} s')

    swe.set_prev()
