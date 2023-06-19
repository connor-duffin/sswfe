import os
import h5py
import tqdm
import numpy as np

from itertools import product
from multiprocessing import Pool
from ns_2d_fenicsx import NSSplit


def run_ns_2d(mesh_file, mu, output_file):
    t = 0
    T = 60
    dt = 1e-3
    nt = int(T / dt)
    thin = 100
    nt_save = nt // thin

    if nt % thin >= 1:
        nt_save += 1

    params = dict(mu=mu, rho=1000)
    # mesh_file = "mesh/branson-refined.msh"
    ns = NSSplit(mesh_file, dt, params)
    ns.setup_form()

    # output_file = "outputs/branson-testing.h5"
    output = h5py.File(output_file, "w")

    metadata = params
    for name, val in metadata.items():
        output.attrs.create(name, val)

    n_dofs_u = ns.V_dof_coordinates.shape[0]
    n_dofs_p = ns.Q_dof_coordinates.shape[0]

    t_out = output.create_dataset("t", shape=(nt_save, ), dtype=np.float64)
    u_out = output.create_dataset("u_mean", shape=(nt_save, 2 * n_dofs_u), dtype=np.float64)
    p_out = output.create_dataset("p_mean", shape=(nt_save, n_dofs_p), dtype=np.float64)

    i_save = 0
    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=nt)
    for i in range(nt):
        progress.update(1)

        t += dt
        ns.solve(t)

        # check for NaNs
        if (np.any(np.isnan(ns.u_.vector.array)) and np.any(np.isnan(ns.p_.vector.array))):
            print(f"Simulation failed at t = {t:.4e}")
            break

        # store outputs
        if i % thin == 0:
            t_out[i_save] = t
            u_out[i_save, :] = ns.u_.vector.array.copy()
            p_out[i_save, :] = ns.p_.vector.array.copy()
            i_save += 1

    output.close()


if __name__ == "__main__":
    mus = [1e-3, 1e-2, 1e-1, 1.]
    mesh_files = [f"mesh/branson-{i}.msh" for i in range(4)]

    model_args = []
    for a in product(mesh_files, mus):
        mesh_file, mu = a
        output_file = f"outputs/ns-fx-mu-{mu}-{os.path.basename(mesh_file)[:-4]}.h5"

        model_args.append((*a, output_file))

    print(model_args)
    n_threads = 16
    p = Pool(n_threads)
    p.starmap(run_ns_2d, model_args)
