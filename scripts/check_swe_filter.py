import h5py
import time

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe

from argparse import ArgumentParser
from tqdm import trange
from numpy.testing import assert_allclose
from scipy.sparse import vstack

from swe_2d import ShallowTwoFilter, ShallowTwoFilterPETSc
from statfenics.utils import build_observation_operator

fe.set_log_level(50)

parser = ArgumentParser()
parser.add_argument("--use_numpy", action="store_true")
args = parser.parse_args()

comm = fe.MPI.comm_world
rank = comm.Get_rank()

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

mesh = fe.RectangleMesh(
    comm, fe.Point(0, 0), fe.Point(10., 5.), 32, 16)

params = dict(
    nu=1 / Re,
    g=g * length_ref / u_ref**2,
    C=0.,
    H=0.053 / H_ref,
    u_inflow=0.004 / u_ref,
    inflow_period=period / time_ref)
control = dict(
    dt=1e-2,
    theta=0.5,
    simulation="laminar",
    use_imex=False,
    use_les=False)

print(params["u_inflow"], params["H"])
start_time = time.time()
if args.use_numpy:
    swe = ShallowTwoFilter(mesh, params, control, comm)
else:
    swe = ShallowTwoFilterPETSc(mesh, params, control, comm)

swe.setup_form()
swe.setup_solver(use_ksp=True)

# setup filter (basically compute prior additive noise covariance)
rho = 1e-2  # approx 1% of the state should be off
ell = 2.  # characteristic lengthscale from cylinder
k = 16
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0.1,
                   ell_u=ell, ell_v=ell, ell_h=ell,
                   k_init_u=k, k_init_v=k, k_init_h=k, k=2*k)
swe.setup_filter(stat_params)

if not args.use_numpy:
    swe.setup_prior_covariance()

print(f"Took {time.time() - start_time} s to setup")

t = 0.
t_final = 0.1
nt = np.int32(np.round(t_final / control["dt"]))

i_dat = 0
for i in trange(nt):
    t += swe.dt
    swe.inlet_velocity.t = t
    swe.prediction_step(t)
    swe.set_prev()

# get first variance of the column vector
if args.use_numpy:
    cov_vec = swe.cov_sqrt[:, 1]
    cov_vec_norm = np.sqrt(cov_vec @ cov_vec)
else:
    cov_vec = swe.cov_sqrt.getColumnVector(1)
    cov_vec_norm = np.sqrt(cov_vec.tDot(cov_vec))

if rank == 0:
    print("First column of covariance norm: ", cov_vec_norm)

# var_v = np.sqrt(np.sum(swe.cov_sqrt.getDenseArray()**2, axis=1))
# var_f = fe.Function(swe.W)
# var_f.vector().set_local(var_v)

# vel, h = var_f.split()
# u, v = vel.split()

# if rank == 0:
#     im = fe.plot(u)
#     plt.colorbar(im)
#     plt.savefig("figures/variance-test-u-MPI.pdf")
#     plt.close()

#     im = fe.plot(h)
#     plt.colorbar(im)
#     plt.savefig("figures/variance-test-h-MPI.pdf")
#     plt.close()

#     if i % 2 == 0:
#         assert np.isclose(t_data[i_dat], t)
#         y_obs = np.concatenate((u_data[i_dat, mask_obs][::20], 
#                                 v_data[i_dat, mask_obs][::20]))
#         swe.update_step(y_obs, compute_lml=False)
#         i_dat += 1

# data_file = "data/run-8-synthetic-data.h5"

# with h5py.File(data_file, "r") as f:
#     t_data = f["t"][:]
#     x_data = f["x"][:]
#     u_data = f["u"][:]
#     v_data = f["v"][:]

# mask_obs = x_data[:, 0] >= 1.3
# x_obs = x_data[mask_obs, :][::20]
# print(x_obs.shape)

# Hu = build_observation_operator(x_obs, swe.W, sub=(0, 0))
# Hv = build_observation_operator(x_obs, swe.W, sub=(0, 1))
# H = vstack((Hu, Hv))
# print(H.shape)
# as on the same fcn space
# assert_allclose(swe.Ku_vals, swe.Kv_vals)

# plt.semilogy(swe.Ku_vals, ".-", label="$u$")
# plt.semilogy(swe.Kv_vals, ".-", label="$v$")
# plt.legend()
# plt.show()

# vel, h = swe.du.split()
# u, v = vel.split()

# fe.plot(u)
# plt.show()

# im = fe.plot(v)
# plt.colorbar(im)
# plt.show()
