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
                     "theta": 0.5,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": True,
                     "laplacian": False,
                     "les": True
                 })

t = 0.
nt = 1001
nt_thin = 100
imex = False

# solves for du_prev
if imex:
    F, J = swe.setup_form(swe.du_prev, swe.du_prev_prev, imex=False)
    bcs, F = swe.setup_bcs(F)
    solver = swe.setup_solver(F, swe.du_prev, bcs, J)
    if swe.use_les:
        swe.les.solve()

    solver.solve()

# F, J = swe.setup_form(swe.du, swe.du_prev, swe.du_prev_prev, imex=False)
F, J = swe.setup_form(swe.du, swe.du_prev, swe.du_prev_prev, imex=imex)
bcs, F = swe.setup_bcs(F)
solver = swe.setup_solver(F, swe.du, bcs, J)
for i in range(nt):
    print(i)
    if swe.use_les:
        swe.les.solve()

    solver.solve()

    # set previous terms
    if imex:
        fe.assign(swe.du_prev_prev, swe.du_prev)
    fe.assign(swe.du_prev, swe.du)
