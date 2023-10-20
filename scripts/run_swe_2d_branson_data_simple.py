import logging

import fenics as fe
import numpy as np
import xarray as xr

from argparse import ArgumentParser
from tqdm import tqdm
from scipy.sparse import vstack, diags
from scipy.signal import wiener

from swfe.swe_2d import ShallowTwoEnsemble
from statfenics.utils import build_observation_operator

comm = fe.MPI.comm_self
rank = comm.Get_rank()
size = comm.Get_size()

logging.basicConfig(level=logging.INFO)
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

# dimensionless parameters
params = dict(
    nu=1 / Re,
    g=g * length_ref / u_ref**2,
    C=0.,
    H=0.053 / H_ref,
    length=20.,
    width=5.,
    cylinder_centre=(10, 2.5),
    cylinder_radius=0.5,
    u_inflow=0.004 / u_ref,
    inflow_period=period / time_ref)

# TODO(connor): refactor to use `params` dict
L = params["length"]
B = params["width"]
r = params["cylinder_radius"]
dt = 5e-2
rho = 1.
ell = 2.

sigma_y = 2e-2
k_approx = 64
k_full = 128

MESH_FILE = "mesh/branson-mesh-nondim.xdmf"
control = dict(dt=dt, theta=0.5, simulation="laminar", use_imex=False, use_les=False)

# swe = ShallowTwoFilter(MESH_FILE, params, control, comm)
# swe = ShallowTwoFilterPETSc(MESH_FILE, params, control, comm)
swe = ShallowTwoEnsemble(MESH_FILE, params, control, comm)

swe.setup_form()
swe.setup_solver(use_ksp=False)
t = 0.
t_final = 5 * period / time_ref
nt = np.int32(np.round(t_final / control["dt"]))

# then setup filter (basically compute prior additive noise covariance)
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0., ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k_approx, k_init_v=k_approx, k_init_h=0, k=k_full, n_ens=64)

swe.setup_filter(stat_params)

# checking things are sound
np.testing.assert_allclose(swe.mean[swe.u_dofs], 0.)
np.testing.assert_allclose(swe.mean[swe.v_dofs], 0.)
np.testing.assert_allclose(swe.mean[swe.h_dofs], 0.)

logger.info("Starting prior computations...")

# start after first data point
# TODO(connor): remove tqdm_offset requirement via multiprocessing
progress_bar = tqdm(total=nt, ncols=80)
for i in range(nt):
    t += swe.dt
    swe.inlet_velocity.t = t

    try:
        swe.prediction_step(t)
        var = np.var(swe.du_ens, axis=1)
        logger.info("Variance: = %.5e", np.linalg.norm(var[swe.u_dofs]))
    except Exception as e:
        logger.info(e)
        logger.error("Solver (prediction_step) failed at time %.5f, exiting timestepping", t)
        raise

    # progress onto next iteration
    swe.set_prev()
    progress_bar.update(1)
