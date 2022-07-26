import numpy as np
import matplotlib.pyplot as plt
import fenics as fe

from swe import ShallowTwo


def plot_fields_curr(swe, t, output_file):
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

    plt.savefig(output_file, dpi=600)
    plt.close()


mesh_file = "mesh/channel-piggott.xdmf"
checkpoint_file = "outputs/swe-channel-checkpoint.h5"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 5e-4,
                     "theta": 1,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": True
                 })

checkpoint = fe.HDF5File(swe.mesh.mpi_comm(), checkpoint_file, "r")
vec_name = f"/du/vector_{100}"
checkpoint.read(swe.du, vec_name)  # read into du
t = checkpoint.attributes(vec_name)["timestamp"]
plot_fields_curr(swe, t, "figures/fields-MPI.png")
