import logging

from fenics import MPI
from swe import ShallowTwo

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
print("using smaller timestep")

t = 0.
nt = 15_001
for i in range(nt):
    t += swe.dt

    try:
        swe.solve()
    except RuntimeError:
        print(f"SOLVER FAILED AT TIME {t:.5f}")
        break
