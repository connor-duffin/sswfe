import h5py
import logging

import numpy as np
import fenics as fe

from argparse import ArgumentParser
from tqdm import trange
from scipy.sparse import vstack
from petsc4py import PETSc

from swfe.swe_2d import ShallowTwoFilter, ShallowTwoFilterPETSc
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

# set the files
MESH_FILE = "mesh/branson-mesh-nondim.xdmf"
DATA_FILE = "data/run-8-synthetic-data-nondim.h5"

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
    length=20.,
    width=5.,
    cylinder_centre=(10, 2.5),
    cylinder_radius=0.5,
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

# and setup filters and appropriate FEM stuff
if args.use_petsc:
    swe = ShallowTwoFilterPETSc(MESH_FILE, params, control, comm)
else:
    swe = ShallowTwoFilter(MESH_FILE, params, control, comm)

swe.setup_form()
swe.setup_solver(use_ksp=False)

# setup the data read-in
with h5py.File(DATA_FILE, "r") as f:
    t_data = f["t"][:]
    x_data = f["x"][:]
    u_data = f["u"][:]
    v_data = f["v"][:]

spatial_obs_freq = 20
mask_obs = np.logical_and(x_data[:, 0] >= 11, x_data[:, 0] <= 13.)
x_obs = x_data[mask_obs, :][::spatial_obs_freq]
nx_obs = x_obs.shape[0]
assert x_obs.shape[1] == 2

if args.use_petsc:
    Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0), out="petsc")
    Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1), out="petsc")
    Hu_size = Hu.getSize()
    Hv_size = Hv.getSize()
    assert Hu_size[1] == Hv_size[1]

    Hu_csr = Hu.getValuesCSR()
    Hu_rowptr = Hu_csr[0]
    Hu_cols = Hu_csr[1]
    Hu_vals = Hu_csr[2]

    Hv_csr = Hv.getValuesCSR()
    Hv_rowptr = Hv_csr[0]
    Hv_cols = Hv_csr[1]
    Hv_vals = Hv_csr[2]

    H_vals = np.concatenate((Hu_vals, Hv_vals))
    H_cols = np.concatenate((Hu_cols, Hv_cols))
    H_rowptr = np.concatenate((Hu_rowptr[:-1], nx_obs + Hv_rowptr))

    # just a test to see if things are OK
    from scipy.sparse import csr_matrix
    H = csr_matrix((H_vals, H_cols, H_rowptr),
                   shape=(Hu_size[0] + Hv_size[0], Hu_size[1]))

    Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
    Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
    H_true = vstack((Hu, Hv), format="csr")

    # check that our construction is correct
    np.testing.assert_allclose(H.data, H_true.data)

    # same no. of columns
    assert Hu_size[1] == Hv_size[1]

    H = PETSc.Mat().create(comm=comm)
    H.setSizes([Hu_size[0] + Hv_size[0], Hu_size[1]])
    H.setType("aij")
    H.setUp()
    H.assemble()
else:
    Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
    Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
    H = vstack((Hu, Hv), format="csr")
    np.testing.assert_allclose(H @ swe.mean, np.zeros((2 * nx_obs, )))
    logger.info("Observation operator shape is %s", H.shape)

# setup filter (basically compute prior additive noise covariance)
rho = 1e-2  # approx 1% of the state should be off
ell = 2.  # characteristic lengthscale from cylinder
k_approx = 32
k_full = 100
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0.,
                   ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k_approx, k_init_v=k_approx, k_init_h=0, k=k_full,
                   H=H, sigma_y=0.05 * params["u_inflow"])
swe.setup_filter(stat_params)

if args.use_petsc:
    swe.setup_covariance()

# hack(connor): conditional while developing
if not args.use_petsc:
    # checking things are sound
    np.testing.assert_allclose(swe.mean[swe.u_dofs], 0.)
    np.testing.assert_allclose(swe.mean[swe.v_dofs], 0.)
    np.testing.assert_allclose(swe.mean[swe.h_dofs], 0.)

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

    rmse = np.zeros((nt_obs, ))
    y = np.zeros((nt, H.shape[0]))

    if args.compute_posterior:
        lml = np.zeros((nt_obs, ))
        logger.info("Starting posterior computations...")
    else:
        logger.info("Starting prior computations...")

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
            y[i_dat, :] = y_obs

            # assimilate data
            try:
                if args.compute_posterior:
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
    output.create_dataset("eff_rank", data=eff_rank)
    output.create_dataset("rmse", data=rmse)

    if args.compute_posterior:
        output.create_dataset("lml", data=lml)

    output.close()
