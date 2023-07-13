import h5py

import numpy as np
import matplotlib.pyplot as plt
import fenics as fe

from tqdm import trange
from numpy.testing import assert_allclose
from scipy.sparse import vstack

from swe_2d import ShallowTwoFilter, ShallowTwoFilterPETSc
from statfenics.utils import build_observation_operator

fe.set_log_level(50)


mesh = "./mesh/branson.xdmf"
# mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(2., 1.), 32, 16)
params = {"nu": 1e-5, "C": 0., "H": 0.053, "u_inflow": 0.004, "inflow_period": 120}
control = {"dt": 5e-2,
           "theta": 0.5,
           "simulation": "cylinder",
           "use_imex": False,
           "use_les": False}
# swe = ShallowTwoFilterPETSc(mesh, params, control)
swe = ShallowTwoFilter(mesh, params, control)
assert swe.L == 2.
assert swe.B == 1.

# check that all the dofs line up
assert_allclose(np.unique(swe.W.dofmap().dofs()),
                np.unique(np.concatenate((swe.u_dofs,
                                          swe.v_dofs,
                                          swe.h_dofs))))
swe.setup_form()
swe.setup_solver(use_ksp=True)

# setup filter (basically compute prior additive noise covariance)
rho = 1e-2
ell = 0.1  # characteristic lengthscale from cylinder
stat_params = dict(rho_u=rho, rho_v=rho, rho_h=0.,
                   ell_u=ell, ell_v=ell, ell_h=0.5,
                   k_init_u=32, k_init_v=32, k_init_h=16, k=32)
swe.setup_filter(stat_params)

t = 0.
t_final = 30.
nt = np.int32(np.round(t_final / control["dt"]))

i_dat = 0
for i in trange(nt):
    t += swe.dt
    swe.inlet_velocity.t = t
    swe.prediction_step(t)
    swe.set_prev()

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
