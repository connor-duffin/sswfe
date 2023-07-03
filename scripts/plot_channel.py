import h5py

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from swe_2d import ShallowTwo


def plot_fields_curr(swe, t, output_dir):
    x_vertices = swe.mesh.coordinates()
    n_vertices = len(x_vertices)
    du_vec = swe.du.compute_vertex_values()
    u1, u2, h = (du_vec[:n_vertices], du_vec[n_vertices:(2 * n_vertices)],
                 du_vec[(2 * n_vertices):])

    x, y = x_vertices[:, 0], x_vertices[:, 1]

    fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(9, 9))
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


def plot_end_surface_heights(swe, checkpoint_file):
    gamma_left = fe.CompiledSubDomain("near(x[0], 0.0)")
    gamma_right = fe.CompiledSubDomain("near(x[0], 2.0)")

    # 1d facet's
    facet_marker = fe.MeshFunction("size_t", swe.mesh, 1)
    facet_marker.set_all(0)
    gamma_left.mark(facet_marker, 1)
    gamma_right.mark(facet_marker, 2)

    du = fe.Function(swe.H_space)
    du.interpolate(fe.Expression("sin(pi * x[1])", degree=4))
    ds = fe.Measure('ds', domain=swe.mesh, subdomain_data=facet_marker)
    np.testing.assert_almost_equal(
        fe.assemble(fe.Constant(1.0) * ds(1)), 1.)

    # TODO(connor): fix up this hard-coded s#!t
    nt_final = 3000
    indices = np.linspace(0, nt_final,
                          num=256, endpoint=False, dtype=np.int32)

    t_obs = np.zeros((len(indices), ))
    h_left = np.zeros((len(indices), ))
    h_right = np.zeros((len(indices), ))

    for i, idx in enumerate(indices):
        vec_id = f"/du/vector_{idx:d}"
        t_obs[i] = checkpoint.attributes(vec_id)["timestamp"]
        checkpoint.read(swe.du, vec_id)
        u, h = swe.du.split()

        h_left[i, ] = fe.assemble(h * ds(1))
        h_right[i, ] = fe.assemble(h * ds(2))

    # plot everything all together
    plt.plot(t_obs / 60, h_left, label="Left")
    plt.plot(t_obs / 60, h_right, label="Right")
    plt.xlabel(r"$t / T$")
    plt.title(r"$\bar{h}(t)$")
    plt.legend()


output_file = "outputs/branson-swe-bottom-friction.h5"
figure_output_dir = "figures/"
output = h5py.File(output_file, "r")

mesh_file = "mesh/branson.xdmf"
params = {"nu": 1e-5, "H": 0.053, "C": 0.}
simulation = "cylinder"
control = {"dt": 1e-2,
           "theta": 1,
           "simulation": simulation,
           "laplacian": True,
           "les": False,
           "integrate_continuity_by_parts": True}

swe = ShallowTwo(mesh=mesh_file, params=params, control=control)

x_vertices = swe.mesh.coordinates()
x, y = x_vertices[:, 0], x_vertices[:, 1]
n_vertices = len(x_vertices)

vec_name = "/du/vector_100"
checkpoint = fe.HDF5File(swe.mesh.mpi_comm(), output_file, "r")
checkpoint.read(swe.du, vec_name)  # read into du
t = checkpoint.attributes(vec_name)["timestamp"]

# plot surface heights
plot_end_surface_heights(swe, checkpoint)
plt.show()
plt.close()

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

# # quiver plot
# plot_quiver_curr(swe, t)
# plt.show()
# plt.close()

# # plot the mesh
# fe.plot(swe.mesh)
# plt.show()
# plt.close()

# # plot the solution field
# plot_fields_curr(swe, t, figure_output_dir)
# plt.show()
# plt.close()
