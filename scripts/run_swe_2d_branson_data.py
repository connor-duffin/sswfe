import h5py
import logging

import fenics as fe
import numpy as np
import xarray as xr

from argparse import ArgumentParser
from tqdm import trange
from scipy.sparse import vstack
from scipy.signal import wiener
from petsc4py import PETSc

from swfe.swe_2d import ShallowTwoFilter
from statfenics.utils import build_observation_operator

comm = fe.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# get command line arguments
parser = ArgumentParser()
parser.add_argument("output_file", type=str)
parser.add_argument("--compute_posterior", action="store_true")
parser.add_argument("--use_petsc", action="store_true")
parser.add_argument("--log_file", type=str, default="swe-filter-run.log")
args = parser.parse_args()

# setup logging
logging.basicConfig(filename=args.log_file,
                    encoding="utf-8",
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

if args.use_petsc:
    logger.error("PETSc not yet configured")
    raise ValueError

# ##########################################
# step 0: relevant constants and setup model
MESH_FILE = "mesh/branson-mesh-nondim.xdmf"
DATA_FILE = "data/Run08_G0_dt4.0_de1.0_ne5_velocity.nc"

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
params = dict(
    nu=2 / Re,
    g=g * length_ref / u_ref**2,
    C=0.,
    H=0.053 / H_ref,
    length=20.,
    width=5.,
    cylinder_centre=(10, 2.5),
    cylinder_radius=0.5,
    u_inflow=0.004 / u_ref,
    inflow_period=period / time_ref)

# and numerical settings
control = dict(
    dt=5e-2,
    theta=0.5,
    simulation="laminar",
    use_imex=False,
    use_les=False)

# TODO(connor): refactor to use `params` dict
L = params["length"]
B = params["width"]
r = params["cylinder_radius"]

swe = ShallowTwoFilter(MESH_FILE, params, control, comm)
swe.setup_form()
swe.setup_solver(use_ksp=False)

# #######################################
# step 1: read in and preprocess the data
ds = xr.open_dataset(DATA_FILE)

# first we get the locations of the observation points with respect to our grid
ds_zero = ds.isel(time=0)
ds_zero_missing = ds_zero.where(ds_zero.isnull(), drop=True)

# scale to our coordinates
x = ds_zero.coords["x"].to_numpy() / length_ref
y = ds_zero.coords["y"].to_numpy() / length_ref

x_missing = ds_zero_missing.coords["x"].to_numpy() / length_ref
y_missing = ds_zero_missing.coords["y"].to_numpy() / length_ref

# compute global `x` coordinates
x_displacement = np.min(x[x > np.max(x_missing)])
x_center_displacement = r - x_displacement
x_global_displacement = L / 2 + x_center_displacement
x_global = x + x_global_displacement
print(x_global)

# compute global `y` coordinates
y_mid = y[len(y) // 2]
y_global = B / 2 + (y - y_mid)
print(y_global)

# and create meshgrid according to our current setup
x_mg, y_mg = np.meshgrid(x_global, y_global, indexing="ij")

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

u_depth_averaged["time"].dtype == np.datetime64
u_depth_averaged = u_depth_averaged.assign_coords(
    dict(time_rel=((u_depth_averaged["time"]
                    - u_depth_averaged["time"][0]) * 1e-9).astype(float)))
v_depth_averaged = v_depth_averaged.assign_coords(
    dict(time_rel=((v_depth_averaged["time"]
                    - v_depth_averaged["time"][0]) * 1e-9).astype(float)))

# reference values: scaled as needed
# to start we get the data for the velocity fields
# and check for sensibility
ud = u_depth_averaged.to_numpy() / u_ref
vd = v_depth_averaged.to_numpy() / u_ref
nrows, ncols = vd[0, :, :].shape

# TODO(connor): NEED TO FILTER OUT NaNs
nt_obs = ud.shape[0]
t_obs = u_depth_averaged.coords["time_rel"].to_numpy() / time_ref
u_obs = ud.reshape((nt_obs, nrows * ncols))
v_obs = vd.reshape((nt_obs, nrows * ncols))

assert ud.shape == vd.shape
assert ud[0, :, :].shape == x_mg.shape
assert vd[0, :, :].shape == y_mg.shape

filter_width = 3
sigma_u_est = np.zeros((nrows, ncols))
sigma_v_est = np.zeros((nrows, ncols))

for i in range(nrows):
    for j in range(ncols):
        ud_filtered = wiener(ud[:, i, j], filter_width)
        vd_filtered = wiener(vd[:, i, j], filter_width)

        sigma_u_est[i, j] = np.std(ud[:, i, j] - ud_filtered)
        sigma_v_est[i, j] = np.std(vd[:, i, j] - vd_filtered)

# check that the min/max variances are sound
print(np.amin(sigma_u_est[~np.isnan(sigma_u_est)]),
      np.amax(sigma_u_est[~np.isnan(sigma_u_est)]))

# HACK(connor): setting to minimum for compatibility
sigma_y = np.amin(sigma_u_est[~np.isnan(sigma_u_est)])

# and set the meshgrid as needed
x_mg = x_mg.reshape((nrows * ncols, 1))
y_mg = y_mg.reshape((nrows * ncols, 1))

# at this stage the data is ready to be assimilated into the model
x_obs = np.hstack((x_mg, y_mg))
assert x_obs.shape[1] == 2
assert np.all(np.logical_and(x_obs[:, 0] >= 10., x_obs[:, 0] <= 20.))
assert np.all(np.logical_and(x_obs[:, 1] >= 0., x_obs[:, 1] <= 5.))
nx_obs = x_obs.shape[0]
print(f"Taking in {nx_obs} observations each time point")

print(np.unique(x_obs[:, 0]))
print(np.unique(x_obs[:, 1]))
assert np.all(~np.isnan(x_obs))

# set the observation operator
Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
H = vstack((Hu, Hv), format="csr")
np.testing.assert_allclose(H @ swe.mean, np.zeros((2 * nx_obs, )))
logger.info("Observation operator shape is %s", H.shape)

# then setup filter (basically compute prior additive noise covariance)
rho = 1e-2  # approx 1% of the state should be off
ell = 2.  # characteristic lengthscale from cylinder
k_approx = 32
k_full = 100
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0.,
                   ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k_approx, k_init_v=k_approx, k_init_h=0, k=k_full,
                   H=H, sigma_y=0.05 * params["u_inflow"])
swe.setup_filter(stat_params)
