import h5py
import os

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from tqdm import tqdm
from swe import ShallowOne

import sys
sys.path.insert(0, "./scripts")


def ds_from_output_file(output_file, field="u"):
    checkpoint = h5py.File(output_file)
    nu = checkpoint["/"].attrs["nu"]
    dt = checkpoint["/"].attrs["dt"]
    nt_out = checkpoint["/du"].attrs["count"]
    x_grid = checkpoint["/du/x_cell_dofs"][:]
    nx = x_grid.shape[0]

    control = {"nx": nx - 1, "dt": dt, "theta": 1.0, "simulation": "tidal_flow"}
    params = {"nu": nu}
    swe = ShallowOne(control, params)
    checkpoint.close()

    t = np.zeros((nt_out, ))
    field_out = np.zeros((nt_out, nx))

    fe_checkpoint = fe.HDF5File(fe.MPI.comm_world, output_file, "r")
    for i in range(nt_out):
        vec_idx = f"/du/vector_{i}"

        fe_checkpoint.read(swe.du, vec_idx)  # read into du
        u_curr, h_curr = swe.get_vertex_values()

        t[i] = fe_checkpoint.attributes(vec_idx)["timestamp"]
        if field == "u":
            field_out[i, :] = u_curr
        else:
            field_out[i, :] = h_curr

    fe_checkpoint.close()
    ds = xr.DataArray(data=field_out[:, :, np.newaxis], coords=dict(t=t, x=x_grid, nu=np.array([nu])), name=field)
    return ds


# read in all outputs from `output_dir`
output_dir = "./outputs/swe-tidal/"
output_files = [output_dir + f for f in os.listdir(output_dir)]
output_files = output_files[::-1]

u_ds = []
h_ds = []
for output_file in tqdm(output_files):
    u_ds.append(ds_from_output_file(output_file, field="u"))
    h_ds.append(ds_from_output_file(output_file, field="h"))

# u/h series
du = xr.merge(u_ds)
dh = xr.merge(h_ds)

# first we plt to get an idea of dims
du.u[100, :, :].plot.line(hue="nu")
plt.show()

du.u[100, :, 2:].plot.line(hue="nu")
plt.show()


def vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(np.linalg.norm, x,
                          input_core_dims=[[dim]],
                          kwargs={'ord': ord, 'axis': -1})


du_lowpass = du.where(du < 1e2)

# and we now compute the energy by proxy
fig, ax = plt.subplots(1, 1, constrained_layout=True)
energy = np.power(vector_norm(du_lowpass, "x", ord=2), 2)
energy.u[:500, :2].plot.line(ax=ax, hue="nu")
# ax.set_ylim([-0.5, 7])
plt.show()

# now we look at an individual instance of the model to check the
# values of the derivatives
idx_select = -1
x = du.coords["x"]
u = du.u[:, :, idx_select].copy()
h = dh.h[:, :, idx_select].copy()
nu = du.coords["nu"][idx_select]
ux = np.gradient(u, x, axis=1)
hx = np.gradient(h, x, axis=1)
uxx = np.gradient(ux, x, axis=1)

nt = len(du.coords["t"])
fig, axs = plt.subplots(2, 5, sharex=True, constrained_layout=True, figsize=(15, 5))
axs = axs.flatten()
for i, idx in enumerate(np.int64(np.linspace(1, nt - 1, 10))):
    axs[i].plot(x, ux[idx, :], label=r"$u_x$")
    axs[i].plot(x, u[idx, :] * ux[idx, :], label=r"$u u_x$")
    axs[i].plot(x, 9.8 * hx[idx, :], label=r"$g h_x$")
    axs[i].plot(x, uxx[idx, :], label=r"$u_{xx}$")

plt.legend()
fig.suptitle(f"Values of PDE terms in SWE, with nu: {nu:.4e}")
plt.show()
