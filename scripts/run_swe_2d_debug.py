import logging

from fenics import MPI
from swe import ShallowTwo

logging.basicConfig(level=logging.INFO)

comm = MPI.comm_world
rank = comm.rank

mesh_file = "mesh/channel-piggott.xdmf"
checkpoint_file = "outputs/swe-channel-checkpoint.h5"
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
        if rank == 0:
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
