import matplotlib.pyplot as plt
from tqdm import tqdm
from swe import ShallowTwo


def plot_fields_curr(swe, t):
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



mesh_file = "mesh/channel-hole.xdmf"
checkpoint_file = "outputs/swe-cylinder-checkpoint.h5"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": 1,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": False
                 })

load_checkpoint = True
if load_checkpoint:
    t = swe.checkpoint_load(checkpoint_file)
    print(f"loading from checkpoint at time t = {t}")
else:
    t = 0.
    swe.setup_checkpoint(checkpoint_file)

nt = 1001
nt_thin = 10
for i in tqdm(range(nt)):
    # store outputs
    if i % nt_thin == 0:
        plot_fields_curr(swe, t)
        plt.show()
        swe.checkpoint_save(t)

    t += swe.dt
    try:
        swe.solve()
    except RuntimeError:
        print(f"failed at iteration {i}, exiting")
        break

swe.checkpoint_close()
