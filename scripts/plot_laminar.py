import h5py

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from swe import ShallowTwo
from utils import dolfin_to_csr


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

    # cbar.ax.set_ylabel('verbosity coefficient')
    # Make a colorbar for the ContourSet returned by the contourf call.
    # plt.savefig(output_dir + "solution-fields.png", dpi=600)
    # plt.close()


parser = ArgumentParser()
parser.add_argument("--integrate_continuity_by_parts", action="store_true")
parser.add_argument("mesh_file", type=str)
parser.add_argument("model_output_file", type=str)
parser.add_argument("figure_output_dir", type=str)
args = parser.parse_args()

output = h5py.File(args.model_output_file, "r")
t_final = output["t"][-1]

swe = ShallowTwo(mesh=args.mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": 1,
                     "simulation": "laminar",
                     "integrate_continuity_by_parts": args.integrate_continuity_by_parts
                 })

swe.set_curr_vector(output["du"][-1, :])
swe.set_prev_vector(output["du"][-2, :])

# plot the mesh
fe.plot(swe.mesh)
plt.show()

# plot the solution field
plot_fields_curr(swe, t_final, args.figure_output_dir)
plt.show()

# investigate the Jacobian matrix
J = fe.assemble(swe.J)
J_scipy = dolfin_to_csr(J)
from scipy.sparse.linalg import eigs
vals, vecs = eigs(J_scipy, k=50)
plt.semilogy(vals)
plt.show()
