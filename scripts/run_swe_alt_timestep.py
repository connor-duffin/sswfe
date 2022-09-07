import logging

from argparse import ArgumentParser
import fenics as fe
from swe import ShallowTwo

logging.basicConfig(level=logging.INFO)

comm = fe.MPI.comm_world
rank = comm.rank

IMEX = False

parser = ArgumentParser()
parser.add_argument("--theta", type=float, default=0.5)
args = parser.parse_args()

mesh_file = "mesh/channel-piggott.xdmf"
swe = ShallowTwo(mesh=mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": args.theta,
                     "simulation": "cylinder",
                     "integrate_continuity_by_parts": True,
                     "laplacian": False,
                     "les": True
                 })
checkpoint_file = f"outputs/swe-channel-alt-ts-theta-{args.theta:.1f}-checkpoint.h5"
logging.info("storing solution at %s", checkpoint_file)
swe.setup_checkpoint(checkpoint_file)

t = 0.
nt = 1001
nt_thin = 100

# solves for du_prev, from du_prev_prev
if IMEX:
    F, J = swe.setup_form(swe.du_prev, swe.du_prev_prev, imex=False)
    bcs, F = swe.setup_bcs(F)
    solver = swe.setup_solver(F, swe.du_prev, bcs, J)
    if swe.use_les:
        swe.les.solve()

    solver.solve()
    t += swe.dt

F, J = swe.setup_form(swe.du, swe.du_prev, swe.du_prev_prev, imex=IMEX)
bcs, F = swe.setup_bcs(F)
solver = swe.setup_solver(F, swe.du, bcs, J)

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
        if swe.use_les:
            swe.les.solve()

        solver.solve()

        # set previous terms
        if IMEX:
            fe.assign(swe.du_prev_prev, swe.du_prev)

        fe.assign(swe.du_prev, swe.du)
        t += swe.dt
    except RuntimeError:
        print(f"SOLVER FAILED AT TIME {t:.5f}")
        swe.checkpoint_save(t)
        break

