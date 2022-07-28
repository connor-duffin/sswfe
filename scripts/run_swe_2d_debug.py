import logging

import fenics as fe
from swe import ShallowTwo

logging.basicConfig(level=logging.INFO)

comm = fe.MPI.comm_world
rank = comm.rank

mesh_file = "mesh/channel-piggott.xdmf"
checkpoint_file = "outputs/swe-channel-outflow-checkpoint.h5"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 5e-4,
                     "theta": 1,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": True
                 })
swe.setup_checkpoint(checkpoint_file)

t = 0.
nt = 30_001
nt_thin = 100
for i in range(nt):
    if i % nt_thin == 0:
        _, h = fe.split(swe.du)
        h_average = fe.assemble(h * fe.dx) / fe.assemble(fe.Constant(1) * fe.dx(swe.mesh))
        if rank == 0:
            logging.info("average height: %.5f", h_average)
            logging.info("storing solution at time %.5f, iteration %d of %d complete",
                         t, i + 1, nt)
        swe.checkpoint_save(t)

    try:
        swe.solve()
        t += swe.dt
    except RuntimeError:
        print(f"SOLVER FAILED AT TIME {t:.5f}")
        swe.checkpoint_save(t)
        break

swe.checkpoint_close()
