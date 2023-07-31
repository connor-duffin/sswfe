import h5py
import logging

import numpy as np
import fenics as fe

from argparse import ArgumentParser
from tqdm import trange
from scipy.sparse import vstack

from swfe.swe_2d import ShallowTwoFilter
from statfenics.utils import build_observation_operator

comm = fe.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# get command line arguments
parser = ArgumentParser()
parser.add_argument("output_file", type=str)
parser.add_argument("--log_file", type=str, default="swe-filter-run.log")
args = parser.parse_args()

# setup logging
logging.basicConfig(filename=args.log_file,
                    encoding="utf-8",
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# set parameters
params = dict(
    nu=2 / Re,
    g=g * length_ref / u_ref**2,
    C=0.,
    H=0.053 / H_ref,
    u_inflow=0.004 / u_ref,
    inflow_period=period / time_ref)
logger.info("Inflow: %.4f, Re: %.2f", params["u_inflow"], Re)

# and numerical settings
control = dict(
    dt=5e-2,
    theta=0.5,
    simulation="laminar",
    use_imex=False,
    use_les=False)
sigma_y = 0.05 * params["u_inflow"]

# and setup filters as needed
mesh_file = "mesh/branson-mesh-nondim.xdmf"
swe = ShallowTwoFilter(mesh_file, params, control, comm)

# setup appropriate FEM things
swe.setup_form()
swe.setup_solver(use_ksp=False)

# setup the data read-in
data_file = "data/run-8-synthetic-data-nondim.h5"
with h5py.File(data_file, "r") as f:
    t_data = f["t"][:]
    x_data = f["x"][:]
    u_data = f["u"][:]
    v_data = f["v"][:]

spatial_obs_freq = 20
mask_obs = x_data[:, 0] >= 13.
x_obs = x_data[mask_obs, :][::spatial_obs_freq]

Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
H = vstack((Hu, Hv))
logger.info("Observation operator shape is %s", H.shape)

# setup filter (basically compute prior additive noise covariance)
rho = 1e-3  # approx 1% of the state should be off
ell = 2.  # characteristic lengthscale from cylinder
k_approx = 32
k_full = 100
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0.,
                   ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k_approx, k_init_v=k_approx, k_init_h=0, k=k_full,
                   H=H, sigma_y=sigma_y)
swe.setup_filter(stat_params)

t = 0.
t_final = 5 * period / time_ref
nt = np.int32(np.round(t_final / control["dt"]))

checkpoint_interval = 10
output = h5py.File(args.output_file, mode="w")
t_checkpoint = output.create_dataset(
    "t_checkpoint", shape=(1, ))
mean_checkpoint = output.create_dataset(
    "mean_checkpoint", shape=swe.mean.shape)
cov_sqrt_checkpoint = output.create_dataset(
    "cov_sqrt_checkpoint", shape=swe.cov_sqrt.shape)

obs_interval = 2
nt_obs = len([i for i in range(nt) if i % obs_interval == 0])

ts = np.zeros((nt, ))
means = np.zeros((nt, swe.mean.shape[0]))
variances = np.zeros((nt, swe.mean.shape[0]))
eff_rank = np.zeros((nt, ))

lml = np.zeros((nt_obs, ))
rmse = np.zeros((nt_obs, ))

i_dat = 0
for i in trange(nt):
    t += swe.dt
    swe.inlet_velocity.t = t

    # push models forward
    try:
        swe.prediction_step(t)
        eff_rank[i] = swe.eff_rank
        logger.info("Effective rank: %.5f", swe.eff_rank)
    except Exception as e:
        t_checkpoint[:] = t
        mean_checkpoint[:] = swe.mean
        cov_sqrt_checkpoint[:] = swe.cov_sqrt
        logger.error("Failed at time %t:.5f, exiting...", t)
        break

    if i % obs_interval == 0:
        assert np.isclose(t_data[i_dat], t)
        y_obs = np.concatenate((u_data[i_dat, mask_obs][::spatial_obs_freq],
                                v_data[i_dat, mask_obs][::spatial_obs_freq]))

        # assimilate data
        try:
            lml[i_dat] = swe.update_step(y_obs, compute_lml=True)
            rmse[i_dat] = (
                np.linalg.norm(y_obs - H @ swe.du.vector().get_local())
                / np.sqrt(len(y_obs)))
            logger.info("t = %.5f, rmse = %.5e", t, rmse[i_dat])
        except Exception as e:
            t_checkpoint[:] = t
            mean_checkpoint[:] = swe.mean
            cov_sqrt_checkpoint[:] = swe.cov_sqrt
            logger.error("Failed at time %t:.5f, exiting...", t)
            break
        i_dat += 1

    # checkpoint every so often
    if i % checkpoint_interval:
        t_checkpoint[:] = t
        mean_checkpoint[:] = swe.mean
        cov_sqrt_checkpoint[:] = swe.cov_sqrt

    # and store things for outputs
    ts[i] = t
    means[i, :] = swe.du.vector().get_local()
    variances[i, :] = np.sum(swe.cov_sqrt**2, axis=1)
    swe.set_prev()

# store all outputs storage
logger.info("now saving output to %s", args.output_file)
metadata = {**params, **control, **stat_params}
for name, val in metadata.items():
    if name == "H":
        # don't create H, just store x-outputs
        output.attrs.create("x_obs", x_obs)
        output.attrs.create("observed_vars", ("u", "v"))
    else:
        # store values into wherever
        output.attrs.create(name, val)

output.create_dataset("t", data=ts)
output.create_dataset("mean", data=means)
output.create_dataset("variance", data=variances)
output.create_dataset("rmse", data=rmse)
output.create_dataset("lml", data=lml)
output.close()
