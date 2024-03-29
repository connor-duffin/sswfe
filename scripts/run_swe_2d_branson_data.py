import h5py
import logging

import fenics as fe
import numpy as np
import xarray as xr

from argparse import ArgumentParser
from tqdm import tqdm
from scipy.sparse import vstack, diags
from scipy.signal import wiener

from swfe.swe_2d import ShallowTwoFilter
from statfenics.utils import build_observation_operator

comm = fe.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()


def run_statfem(data, params, control, stat_params, tqdm_offset=0, load_from_checkpoint=False):
    swe = ShallowTwoFilter(MESH_FILE, params, control, comm)
    swe.setup_form()
    swe.setup_solver(use_ksp=False)

    t_obs = data['t_obs']
    x_obs = data['x_obs']
    u_obs = data['u_obs']
    v_obs = data['v_obs']

    t = 0.
    t_final = 5 * params['inflow_period']
    nt = np.int32(np.round(t_final / control['dt']))
    nt_obs = len(t_obs)

    Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
    Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
    H = vstack((Hu, Hv), format='csr')
    stat_params['H'] = H

    # get indices from the operator for zeta calculation
    H_rows, H_cols = H.nonzero()
    n_approx_obs = len(H_cols)
    n_approx_unobs = H.shape[1] - n_approx_obs
    assert n_approx_obs + n_approx_unobs == len(swe.W.dofmap().dofs())

    H_identity = diags(np.ones(n_approx_obs), shape=(n_approx_obs, len(swe.W.dofmap().dofs())))
    H_identity_flip = diags(np.ones(n_approx_unobs), shape=(n_approx_unobs, len(swe.W.dofmap().dofs())))
    print(H_identity.shape, H_identity_flip.shape)

    swe.setup_filter(stat_params)
    np.testing.assert_allclose(H @ swe.mean, np.zeros((2 * nx_obs, )))

    # check that H_identity hack works
    print((H_identity * swe.cov_sqrt).shape)

    # checking things are sound
    np.testing.assert_allclose(swe.mean[swe.u_dofs], 0.)
    np.testing.assert_allclose(swe.mean[swe.v_dofs], 0.)
    np.testing.assert_allclose(swe.mean[swe.h_dofs], 0.)

    checkpoint_interval = 25
    output = h5py.File(OUTPUT_FILE, mode='w')
    t_checkpoint = output.create_dataset('t_checkpoint', shape=(1, ))
    mean_checkpoint = output.create_dataset('mean_checkpoint', shape=swe.mean.shape)
    cov_sqrt_checkpoint = output.create_dataset('cov_sqrt_checkpoint', shape=swe.cov_sqrt.shape)

    # check this is an integer, then cast to an integer
    i_dat = 1
    obs_interval = t_obs[1] / control['dt']
    assert obs_interval % 1 <= 1e-14
    obs_interval = np.int32(t_obs[1] / control['dt'])
    logger.info(f'Observing every {obs_interval:d} timesteps')

    ts = np.zeros((nt, ))
    means = np.zeros((nt, swe.mean.shape[0]))
    variances = np.zeros((nt, swe.mean.shape[0]))
    eff_rank = np.zeros((nt, ))

    error = np.zeros((nt_obs, ))
    y = np.zeros((nt, H.shape[0]))

    if load_from_checkpoint:
        with h5py.File(CHECKPOINT_FILE, mode='r') as f:
            t = f['/t_checkpoint'][:].item()
            i_dat = np.argwhere(t <= t_obs)[-1].item()
            logger.info(f'Loading checkpointed data at t = {t:.5f}, starting data at index {i_dat}')

            swe.mean[:] = f['/mean_checkpoint'][:]
            swe.du.vector().set_local(f['/mean_checkpoint'][:])
            swe.du_prev.vector().set_local(f['/mean_checkpoint'][:])

            swe.cov_sqrt[:] = f['/cov_sqrt_checkpoint'][:]
            swe.cov_sqrt_prev[:] = f['/cov_sqrt_checkpoint'][:]

    lml = np.zeros((nt_obs, ))
    correction = np.zeros((nt_obs, swe.mean.shape[0]))
    logger.info('Starting posterior computations...')

    # start after first data point
    # TODO(connor): remove tqdm_offset requirement via multiprocessing
    # inflation_factor = 5e-2
    # swe.cov_sqrt_prev *= (1. + inflation_factor)
    progress_bar = tqdm(total=nt, position=tqdm_offset, ncols=80)
    for i in range(nt):
        t += swe.dt
        swe.inlet_velocity.t = t

        try:
            swe.prediction_step(t)
            eff_rank[i] = swe.eff_rank
            logger.info('Effective rank: %.5f', swe.eff_rank)
        except Exception as e:
            t_checkpoint[:] = t
            mean_checkpoint[:] = swe.mean
            cov_sqrt_checkpoint[:] = swe.cov_sqrt
            logger.info(e)
            logger.error('Solver (prediction_step) failed at time %.5f, exiting timestepping', t)
            break

        # assimilation step
        # if (i + 1) % obs_interval == 0:
        # assert np.isclose(t_obs[i_dat], t)
        if np.isclose(t_obs[i_dat], t):
            logger.info(f'Observation time reached: {t_obs[i_dat]:.4f}')
            logger.info(f'Data/obs. time: {t_obs[i_dat]:.5f}, ' + f'Simulation time {t:.5f}')
            y_obs = np.concatenate((u_obs[i_dat, :], v_obs[i_dat, :]))
            y[i_dat, :] = y_obs

            try:
                theta = np.linalg.norm(y_obs - H @ swe.du.vector().get_local()) / np.linalg.norm(y_obs)
                logger.info('t = %.5f, theta = %.5e', t, theta)
                lml[i_dat], correction[i_dat, :] = swe.update_step(y_obs, compute_lml=True)
                error[i_dat] = np.linalg.norm(y_obs - H @ swe.du.vector().get_local()) / np.linalg.norm(y_obs)
                logger.info('t = %.5f, theta_post = %.5e', t, error[i_dat])
            except Exception as e:
                t_checkpoint[:] = t
                mean_checkpoint[:] = swe.mean
                cov_sqrt_checkpoint[:] = swe.cov_sqrt
                logger.info(e)
                logger.error('Update step failed at time %.5f, exiting timestepping', t)
                break
            i_dat += 1

        zeta = np.linalg.norm((H_identity @ swe.cov_sqrt) @ (H_identity_flip @ swe.cov_sqrt).T, ord='fro')
        logger.info('Zeta (cross-cov): %.5e', zeta)

        # checkpoint every so often
        if i % checkpoint_interval:
            t_checkpoint[:] = t
            mean_checkpoint[:] = swe.mean
            cov_sqrt_checkpoint[:] = swe.cov_sqrt

        # store things for outputs
        ts[i] = t
        means[i, :] = swe.du.vector().get_local()
        variances[i, :] = np.sum(swe.cov_sqrt**2, axis=1)
        swe.set_prev()

        # and update progress bar
        progress_bar.update(1)

    # store all outputs
    logger.info('now saving output to %s', OUTPUT_FILE)
    metadata = {**params, **control, **stat_params}
    for name, val in metadata.items():
        if name == 'H':
            output.attrs.create('x_obs', x_obs)
            output.attrs.create('observed_vars', ('u', 'v'))
        else:
            output.attrs.create(name, val)

    output.create_dataset('t', data=ts)
    output.create_dataset('mean', data=means)
    output.create_dataset('variance', data=variances)
    output.create_dataset('eff_rank', data=eff_rank)
    output.create_dataset('rmse', data=error)
    output.create_dataset('correction', data=correction)
    output.create_dataset('lml', data=lml)
    output.close()


# ##########################
# get command line arguments
parser = ArgumentParser()
parser.add_argument('--rho', type=np.float64, default=1e-2)
parser.add_argument('--dt', type=np.float64, default=5e-2)
parser.add_argument('--k_approx', type=int, default=100)
parser.add_argument('--k_full', type=int, default=200)
parser.add_argument('--obs_skip', type=int, default=1)
parser.add_argument('--tqdm_offset', type=int, default=0)
parser.add_argument('--compute_posterior', action='store_true', help='Compute posterior; else compute prior')
parser.add_argument('--load_from_checkpoint', action='store_true', help='Load from checkpoint file')
args = parser.parse_args()

# ##########################################
# step 0: relevant constants and setup model
output_file_base = ('branson-run08-coarse-theta-1.0-'
                    + f'dt-{args.dt:.4f}-'
                    + f'rho-{args.rho:.4e}-'
                    + f'k-approx-{args.k_approx:d}-'
                    + f'k-full-{args.k_full:d}-'
                    + f'obs-skip-{args.obs_skip:d}-')
output_file_base += 'post'

OUTPUT_FILE = 'outputs/' + output_file_base + '.h5'
CHECKPOINT_FILE = 'outputs/' + output_file_base + '-checkpoint.h5'
LOG_FILE = 'log/' + output_file_base + '.log'
# MESH_FILE = 'mesh/branson-mesh-nondim.xdmf'
MESH_FILE = 'mesh/branson-mesh-nondim-coarse.xdmf'
DATA_FILE = 'data/Run08_G0_dt4.0_de1.0_ne5_velocity.nc'

logging.basicConfig(filename=LOG_FILE, encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# physical settings
period = 120.
nu = 5e-6
g = 9.8

# reference values
u_ref = 0.01  # cm/s
length_ref = 0.1  # cylinder
time_ref = length_ref / u_ref
H_ref = length_ref

# compute reynolds number
Re = u_ref * length_ref / nu

# parameters and numerical control settings
params = dict(nu=1 / Re, g=g * length_ref / u_ref**2, C=0., H=0.053 / H_ref,
              length=20., width=5., cylinder_centre=(10, 2.5), cylinder_radius=0.5,
              u_inflow=0.004 / u_ref, inflow_period=period / time_ref)
control = dict(dt=args.dt, theta=1., simulation='cylinder', use_imex=False, use_les=False)

L = params['length']
B = params['width']
r = params['cylinder_radius']

# #######################################
# step 1: read in and preprocess the data
ds = xr.open_dataset(DATA_FILE)

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
x_obs = np.vstack((x_mg[~mask][::args.obs_skip], y_mg[~mask][::args.obs_skip])).T
u_obs = ud[:, ~mask][:, ::args.obs_skip]
v_obs = vd[:, ~mask][:, ::args.obs_skip]

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

# HACK(connor) instead of using this we just hardcode sigma_y
# filter_width = 3
# sigma_u_est = np.zeros((nrows, ncols))
# sigma_v_est = np.zeros((nrows, ncols))

# for i in range(nrows):
#     for j in range(ncols):
#         ud_filtered = wiener(ud[:, i, j], filter_width)
#         vd_filtered = wiener(vd[:, i, j], filter_width)

#         sigma_u_est[i, j] = np.std(ud[:, i, j] - ud_filtered)
#         sigma_v_est[i, j] = np.std(vd[:, i, j] - vd_filtered)

# print(np.amin(sigma_u_est[~np.isnan(sigma_u_est)]),
#       np.amax(sigma_u_est[~np.isnan(sigma_u_est)]))
# print(np.amin(sigma_v_est[~np.isnan(sigma_v_est)]),
#       np.amax(sigma_v_est[~np.isnan(sigma_v_est)]))
# sigma_y = np.mean(sigma_u_est[~np.isnan(sigma_u_est)])

rho = args.rho
ell = 2.
k_approx = args.k_approx
k_full = args.k_full

sigma_y = 1e-2
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=rho, ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k_approx, k_init_v=k_approx, k_init_h=k_approx, k=k_full, sigma_y=sigma_y)
data = dict(t_obs=t_obs, x_obs=x_obs, u_obs=u_obs, v_obs=v_obs)

run_statfem(data, params, control, stat_params,
            tqdm_offset=args.tqdm_offset, load_from_checkpoint=args.load_from_checkpoint)
