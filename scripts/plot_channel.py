import h5py

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from swe import ShallowTwo


def plot_fields_curr(swe, t, output_dir):
    x_vertices = swe.mesh.coordinates()
    n_vertices = len(x_vertices)
    du_vec = swe.du.compute_vertex_values()
    u1, u2, h = (du_vec[:n_vertices], du_vec[n_vertices:(2 * n_vertices)],
                 du_vec[(2 * n_vertices):])

    x, y = x_vertices[:, 0], x_vertices[:, 1]

    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(9, 6))
    fig.suptitle(f"SWE solution fields at time t = {t:.5f}")

    im = axs[0].tricontourf(x, y, u1, 64)
    axs[0].set_title(r"$u_1$")
    cbar = fig.colorbar(im, ax=axs[0])

    im = axs[1].tricontourf(x, y, u2, 64)
    axs[1].set_title(r"$u_2$")
    cbar = fig.colorbar(im, ax=axs[1])

    im = axs[2].tricontourf(x, y, h, 64, cmap="coolwarm")
    axs[2].set_title(r"$h$")
    cbar = fig.colorbar(im, ax=axs[2])

    for ax in axs:
        ax.set_ylabel(r"$y$")
    axs[-1].set_xlabel(r"$x$")


def plot_quiver_curr(swe, t):
    x_vertices = swe.mesh.coordinates()
    n_vertices = len(x_vertices)
    du_vec = swe.du.compute_vertex_values()
    u1, u2, h = (du_vec[:n_vertices], du_vec[n_vertices:(2 * n_vertices)],
                 du_vec[(2 * n_vertices):])

    x, y = x_vertices[:, 0], x_vertices[:, 1]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.quiver(x, y, u1, u2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(f"Velocity field at time t = {t:.5f}")


parser = ArgumentParser()
parser.add_argument("model_output_file", type=str)
parser.add_argument("figure_output_dir", type=str)
args = parser.parse_args()

output = h5py.File(args.model_output_file, "r")

mesh_file = "mesh/channel-piggott.xdmf"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": 1,
                     "integrate_continuity_by_parts": True,
                     "simulation": "cylinder",
                     "laplacian": False,
                     "les": True
                 })

x_vertices = swe.mesh.coordinates()
x, y = x_vertices[:, 0], x_vertices[:, 1]
n_vertices = len(x_vertices)

vec_name = "/du/vector_100"
checkpoint = fe.HDF5File(swe.mesh.mpi_comm(), args.model_output_file, "r")
checkpoint.read(swe.du, vec_name)  # read into du
t = checkpoint.attributes(vec_name)["timestamp"]

# def extract_fields(swe, n_vertices):
#     du_vec = swe.du.compute_vertex_values()
#     u1, u2, h = (du_vec[:n_vertices], du_vec[n_vertices:(2 * n_vertices)],
#                  du_vec[(2 * n_vertices):])
#     return (u1, u2, h)

# u1_curr, u2_curr, h_curr = extract_fields(swe, n_vertices)

# t = checkpoint.attributes(vec_name)["timestamp"]
# u1[i, :] = u1_curr
# u2[i, :] = u2_curr
# h[i, :] = h_curr

plot_quiver_curr(swe, t)
plt.savefig(args.figure_output_dir + "quiver.png", dpi=600)
plt.close()

# plot the mesh
fe.plot(swe.mesh)
plt.savefig(args.figure_output_dir + "mesh.png", dpi=600)
plt.close()

# plot the solution field
plot_fields_curr(swe, t, args.figure_output_dir)
plt.savefig(args.figure_output_dir + "solution-fields.png", dpi=600)
plt.close()
