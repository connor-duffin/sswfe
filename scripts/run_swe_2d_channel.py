import h5py
import logging

from argparse import ArgumentParser
from swe import ShallowTwo
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(relativeCreated)d ms - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("--integrate_continuity_by_parts", action="store_true")
parser.add_argument("--cylinder", action="store_true")
parser.add_argument("mesh_file", type=str)
parser.add_argument("output_file", type=str)
args = parser.parse_args()

if args.cylinder:
    simulation = "cylinder"
else:
    simulation = "laminar"

logging.info("using %s as the simulation settings", simulation)

swe = ShallowTwo(mesh=args.mesh_file,
                 control={
                     "dt": 1e-3,
                     "theta": 1,
                     "simulation": simulation,
                     "integrate_continuity_by_parts": args.integrate_continuity_by_parts
                 })

if args.integrate_continuity_by_parts:
    logging.info("integrating continuity equation by parts")

nt = 100_001
nt_thin = 100
# HACK: because I am lazy at the moment
nt_out = len([i for i in range(nt) if i % nt_thin == 0])
n_dofs = len(swe.du.vector().get_local())
logging.info("saving %i DOFs at %i timesteps", n_dofs, nt_out)
logging.info("running simulation up to time %f.5f", nt * swe.dt)

output = h5py.File(args.output_file, "w")
output.create_dataset("x_vertices", data=swe.mesh.coordinates())
t_out = output.create_dataset("t", (nt_out, ))
du_out = output.create_dataset("du", (nt_out, n_dofs))

t = 0.
i_save = 0
for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.solve()
    except RuntimeError:
        print(f"failed at iteration {i}, time {t:.5f} exiting")
        break

    if i % nt_thin == 0:
        logging.info("%d of %d timesteps done", i + 1, nt)
        logging.info("saving output at time %.5f", t)
        t_out[i_save] = t
        du_out[i_save, :] = swe.du.vector().get_local()
        i_save += 1

output.close()
