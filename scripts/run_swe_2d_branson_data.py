import h5py
import logging

import fenics as fe
import numpy as np
import xarray as xr

from argparse import ArgumentParser
from tqdm import tqdm
from scipy.sparse import vstack
from scipy.signal import wiener

from swfe.swe_2d import ShallowTwoFilter, ShallowTwoFilterPETSc
from statfenics.utils import build_observation_operator

comm = fe.MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()


def run_model(dt, rho, ell, sigma_y, k_approx, k_full,
              use_petsc=False, compute_posterior=False, tqdm_offset=0, dry_run=False):
    # numerical settings
    control = dict(
        dt=dt,
        theta=0.5,
        simulation="laminar",
        use_imex=False,
        use_les=False)

    if use_petsc:
        swe = ShallowTwoFilterPETSc(MESH_FILE, params, control, comm)
    else:
        swe = ShallowTwoFilter(MESH_FILE, params, control, comm)

    swe.setup_form()
    swe.setup_solver(use_ksp=False)
    t = 0.
    t_final = 5 * period / time_ref
    nt = np.int32(np.round(t_final / control["dt"]))
    nt_obs = len(t_obs)

    # HACK(connor): currently running conditionally
    # as PETSc has much less bells and whistles
    if use_petsc:
        stat_params = dict(
            rho_u=rho, rho_v=rho, rho_h=0.,
            ell_u=ell, ell_v=ell, ell_h=ell,
            k_init_u=k_approx, k_init_v=k_approx, k_init_h=0,
            k=k_full)

        swe.setup_filter(stat_params)
        swe.setup_prior_covariance()

        eff_rank = np.zeros((nt, ))
        for i in tqdm(nt):
            t += swe.dt
            swe.inlet_velocity.t = t

            swe.prediction_step(t)
            swe.set_prev()
    else:
        Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
        Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
        H = vstack((Hu, Hv), format="csr")
        logger.info("Observation operator shape is %s", H.shape)

        # then setup filter (basically compute prior additive noise covariance)
        stat_params = dict(
            rho_u=rho, rho_v=rho, rho_h=0.,
            ell_u=ell, ell_v=ell, ell_h=ell,
            k_init_u=k_approx, k_init_v=k_approx, k_init_h=0,
            k=k_full,
            H=H, sigma_y=sigma_y)

        swe.setup_filter(stat_params)
        np.testing.assert_allclose(H @ swe.mean, np.zeros((2 * nx_obs, )))

        # checking things are sound
        np.testing.assert_allclose(swe.mean[swe.u_dofs], 0.)
        np.testing.assert_allclose(swe.mean[swe.v_dofs], 0.)
        np.testing.assert_allclose(swe.mean[swe.h_dofs], 0.)

        checkpoint_interval = 10
        output = h5py.File(OUTPUT_FILE, mode="w")
        t_checkpoint = output.create_dataset(
            "t_checkpoint", shape=(1, ))
        mean_checkpoint = output.create_dataset(
            "mean_checkpoint", shape=swe.mean.shape)
        cov_sqrt_checkpoint = output.create_dataset(
            "cov_sqrt_checkpoint", shape=swe.cov_sqrt.shape)

        # check this is an integer, then cast to an integer
        obs_interval = t_obs[1] / control["dt"]
        assert obs_interval % 1 <= 1e-14
        obs_interval = np.int32(t_obs[1] / control["dt"])
        logger.info(f"Observing every {obs_interval:d} timesteps")

        ts = np.zeros((nt, ))
        means = np.zeros((nt, swe.mean.shape[0]))
        variances = np.zeros((nt, swe.mean.shape[0]))
        eff_rank = np.zeros((nt, ))

        rmse = np.zeros((nt_obs, ))
        y = np.zeros((nt, H.shape[0]))

        if compute_posterior:
            lml = np.zeros((nt_obs, ))
            logger.info("Starting posterior computations...")
        else:
            logger.info("Starting prior computations...")

        # start after first data point
        # TODO(connor): remove tqdm_offset requirement: should be OK
        i_dat = 1
        progress_bar = tqdm(total=nt, position=tqdm_offset)
        for i in range(nt):
            t += swe.dt
            swe.inlet_velocity.t = t

            # prediction step
            try:
                if not dry_run:
                    swe.prediction_step(t)
                    eff_rank[i] = swe.eff_rank
                    logger.info("Effective rank: %.5f", swe.eff_rank)
            except Exception as e:
                t_checkpoint[:] = t
                mean_checkpoint[:] = swe.mean
                cov_sqrt_checkpoint[:] = swe.cov_sqrt
                logger.error("Failed at time %.5f, exiting...", t)
                break

            # assimilation step
            # if (i + 1) % obs_interval == 0:
            # assert np.isclose(t_obs[i_dat], t)
            if np.isclose(t_obs[i_dat], t):
                logger.info(f"Observation time reached: {t_obs[i_dat]:.4e}")
                logger.info(f"Data/obs. time: {t_obs[i_dat]:.5f}, "
                            + f"Simulation time {t:.5f}")
                y_obs = np.concatenate((u_obs[i_dat, :], v_obs[i_dat, :]))
                y[i_dat, :] = y_obs

                try:
                    if compute_posterior and not dry_run:
                        lml[i_dat] = swe.update_step(y_obs, compute_lml=True)

                    rmse[i_dat] = (
                        np.linalg.norm(y_obs - H @ swe.du.vector().get_local())
                        / np.sqrt(len(y_obs)))
                    logger.info("t = %.5f, rmse = %.5e", t, rmse[i_dat])
                except Exception as e:
                    t_checkpoint[:] = t
                    mean_checkpoint[:] = swe.mean
                    cov_sqrt_checkpoint[:] = swe.cov_sqrt
                    logger.error("Failed at time %.5f, exiting...", t)
                    break
                i_dat += 1

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

        # store all outputs storage
        logger.info("now saving output to %s", OUTPUT_FILE)
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

        if compute_posterior:
            output.create_dataset("lml", data=lml)

        output.close()


if __name__ == "__main__":
    print("MODULARITY")

    # get command line arguments
    parser = ArgumentParser()
    parser.add_argument("--rho", type=np.float64, default=1e-2)
    parser.add_argument("--dt", type=np.float64, default=5e-2)
    parser.add_argument("--k_approx", type=int, default=100)
    parser.add_argument("--k_full", type=int, default=200)
    parser.add_argument("--obs_skip", type=int, default=1)
    parser.add_argument("--tqdm_offset", type=int, default=0)
    parser.add_argument("--compute_posterior", action="store_true", help="Compute posterior; else compute prior")
    parser.add_argument("--dry_run", action="store_true", help="Run setup only; no prior/post calculations")
    parser.add_argument("--use_petsc", action="store_true", help="Compute using PETSc not numpy/scipy")
    args = parser.parse_args()

    # ##########################################
    # step 0: relevant constants and setup model
    output_file_base = (f"branson-run08-dt-{args.dt:.4f}-"
                        + f"rho-{args.rho:.4e}-"
                        + f"k-approx-{args.k_approx:d}-"
                        + f"k-full-{args.k_full:d}-"
                        + f"obs-skip-{args.obs_skip:d}-")

    if args.compute_posterior:
        output_file_base += "post"
    else:
        output_file_base += "prior"

    OUTPUT_FILE = "outputs/" + output_file_base + ".h5"
    LOG_FILE = "log/" + output_file_base + ".log"
    MESH_FILE = "mesh/branson-mesh-nondim.xdmf"
    DATA_FILE = "data/Run08_G0_dt4.0_de1.0_ne5_velocity.nc"

    logging.basicConfig(filename=LOG_FILE,
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

    # #######################################
    # step 1: read in and preprocess the data
    ds = xr.open_dataset(DATA_FILE)

    # first we get the locations of the observation points wrt grid
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

    # compute global `y` coordinates
    y_mid = y[len(y) // 2]
    y_global = B / 2 + (y - y_mid)

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
        dict(time_rel=((u_depth_averaged["time"] - u_depth_averaged["time"][0]) * 1e-9).astype(float)))
    v_depth_averaged = v_depth_averaged.assign_coords(
        dict(time_rel=((v_depth_averaged["time"] - v_depth_averaged["time"][0]) * 1e-9).astype(float)))

    # reference values: scaled as needed
    # to start we get the data for the velocity fields
    # and check for sensibility
    ud = u_depth_averaged.to_numpy() / u_ref
    vd = v_depth_averaged.to_numpy() / u_ref
    nrows, ncols = vd[0, :, :].shape
    nt_obs = ud.shape[0]

    # mask values "close" to the cylinder, avoiding values outside the mesh
    c = np.sqrt((x_mg - L / 2)**2 + (y_mg - B / 2)**2)
    mask = (c <= 0.6)

    t_obs = u_depth_averaged.coords["time_rel"].to_numpy() / time_ref
    x_obs = np.vstack((x_mg[~mask][::args.obs_skip],
                       y_mg[~mask][::args.obs_skip])).T
    u_obs = ud[:, ~mask][:, ::args.obs_skip]
    v_obs = vd[:, ~mask][:, ::args.obs_skip]

    # checking sanity
    period_ref = period / time_ref
    assert t_obs[-1] / period_ref >= 5.
    assert x_obs.shape[1] == 2
    assert np.all(np.logical_and(x_obs[:, 0] >= 10., x_obs[:, 0] <= 20.))
    assert np.all(np.logical_and(x_obs[:, 1] >= 0., x_obs[:, 1] <= 5.))
    nx_obs = x_obs.shape[0]
    logger.info(f"Taking in {nx_obs} observations each time point")
    assert np.all(~np.isnan(x_obs))
    assert np.any(np.isnan(u_obs)) == False
    assert np.any(np.isnan(v_obs)) == False

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
    # print(np.amin(sigma_u_est[~np.isnan(sigma_u_est)]),
    #       np.amax(sigma_u_est[~np.isnan(sigma_u_est)]))
    # print(np.amin(sigma_v_est[~np.isnan(sigma_v_est)]),
    #       np.amax(sigma_v_est[~np.isnan(sigma_v_est)]))
    # sigma_y = np.mean(sigma_u_est[~np.isnan(sigma_u_est)])

    run_model(dt=args.dt, rho=args.rho, ell=2., sigma_y=0.02, k_approx=args.k_approx, k_full=args.k_full,
              use_petsc=args.use_petsc, compute_posterior=args.compute_posterior,
              tqdm_offset=args.tqdm_offset, dry_run=args.dry_run)
