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

    plt.savefig(output_file, dpi=600)
    plt.close()


class LeftBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14  # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 0.) < tol


class RightBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14  # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1.) < tol


mesh_file = "mesh/channel-piggott.xdmf"
checkpoint_file = "outputs/swe-channel-outflow-checkpoint.h5"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 5e-4,
                     "theta": 1,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": True
                 })

Gamma_left = LeftBoundary()
Gamma_left.mark(swe.boundaries, 1)  # mark with tag 1 for RHS
Gamma_right = RightBoundary()
Gamma_right.mark(swe.boundaries, 2)  # mark with tag 2 for RHS

n = fe.FacetNormal(swe.mesh)
ds = fe.Measure('ds', domain=swe.mesh, subdomain_data=swe.boundaries)

for i in range(100):
    checkpoint = fe.HDF5File(swe.mesh.mpi_comm(), checkpoint_file, "r")
    vec_name = f"/du/vector_{i + 1}"
    checkpoint.read(swe.du, vec_name)  # read into du
    t = checkpoint.attributes(vec_name)["timestamp"]

    # compute average h value
    u, h = fe.split(swe.du)
    # print(
    #     fe.assemble(h * fe.dx) / fe.assemble(fe.Constant(1) * fe.dx(swe.mesh)))
    print(fe.assemble(fe.inner(u, n) * ds(2)))

    # plot fields
    print(f"plotting at time {t:.5f}")
    plot_fields_curr(swe, t, f"figures/fields-outflow/field-{i:05}.png")
