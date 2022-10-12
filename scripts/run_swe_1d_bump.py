import logging

import numpy as np
import fenics as fe

from tqdm import tqdm
from argparse import ArgumentParser
from swe import ShallowOne, ShallowOneLinear

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
fe.set_log_level(40)

parser = ArgumentParser()
# sim settings
parser.add_argument("--output_file", type=str)
parser.add_argument("--nu", type=float, default=0.)

# numerical settings
parser.add_argument("--nx", type=int)
parser.add_argument("--dt", type=float)
parser.add_argument("--nt_save", type=int)
parser.add_argument("--linear", action="store_true")
args = parser.parse_args()

control = {"nx": args.nx,
           "dt": args.dt,
           "theta": 0.6,
           "simulation": "immersed_bump"}

if args.linear:
    logger.info("using linear SWE (no diffusion)")
    swe = ShallowOneLinear(control, {"nu": args.nu})
else:
    logger.info("using nonlinear SWE (w/diffusion)")
    swe = ShallowOne(control, {"nu": args.nu})

logger.info("setting up checkpoint at %s", args.output_file)
swe.setup_checkpoint(args.output_file)
attrs = swe.checkpoint.attributes("/")
attrs["linear"] = args.linear
attrs["nu"] = args.nu
attrs["dt"] = args.dt

t = 0.
t_final = 5 * 60.
nt = np.int64(t_final / control["dt"])

for i in tqdm(range(nt)):
    t += swe.dt
    try:
        swe.solve(t)
        if i % args.nt_save == 0:
            swe.checkpoint_save(t)
    except RuntimeError:
        logger.error("SWE, nu = %.5f failed at t= %.5f, exiting", args.nu, t)
        swe.checkpoint_save(t)
        break

swe.checkpoint_close()
